use std::cmp::min;
use std::env;
use std::os::raw::c_void;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::backend::Backend;
use crate::comm::Comm;
use crate::constants::{CKPT_MSG, INIT_MSG, MAX_FILE_NUM, RESTORE_MSG, SHM_SIZE, STAGING_BUF_SIZE};
use crate::copy;
use crate::cuda::{self, CudaMemcpyKind, Stream};
use crate::{Error, Result};

pub struct Runtime {
    comm: Comm,
    backend: Option<Backend>,
}

unsafe impl Send for Runtime {}

impl Runtime {
    pub fn new(pid: libc::pid_t) -> Result<Self> {
        Ok(Self {
            comm: Comm::for_pid(pid)?,
            backend: None,
        })
    }

    pub fn handle_control_message(&mut self) -> Result<()> {
        let msg = self.comm.recv_msg();
        let result = match msg {
            INIT_MSG => self.init_checkpoint(),
            CKPT_MSG => self.checkpoint(),
            RESTORE_MSG => self.restore(),
            other => Err(Error::Protocol(format!(
                "unexpected control message {other}"
            ))),
        };
        if result.is_ok() {
            self.comm.send_finish();
        } else {
            self.comm.send_error();
        }
        result
    }

    fn init_checkpoint(&mut self) -> Result<()> {
        cuda::ensure_context()?;
        let path = self.comm.controls().checkpoint_path();
        let mut backend = Backend::setup_for_current_pid(&path)?;
        backend.try_register_staging_with_cuda();
        self.backend = Some(backend);
        Ok(())
    }

    fn backend_mut(&mut self) -> Result<&mut Backend> {
        self.backend
            .as_mut()
            .ok_or_else(|| Error::Protocol("checkpoint backend is not initialized".to_string()))
    }

    fn checkpoint(&mut self) -> Result<()> {
        let backend = self.backend_mut()?;
        checkpoint_backend(backend)
    }

    fn restore(&mut self) -> Result<()> {
        let backend = self.backend_mut()?;
        restore_backend(backend)
    }
}

pub struct NvidiaController {
    backend: Backend,
    pid: libc::pid_t,
}

impl NvidiaController {
    pub fn init_for_current_pid(path: impl AsRef<Path>) -> Result<Self> {
        cuda::ensure_context()?;
        let path = path.as_ref().to_string_lossy();
        let mut backend = Backend::setup_for_current_pid(&path)?;
        backend.try_register_staging_with_cuda();
        Ok(Self {
            backend,
            pid: unsafe { libc::getpid() },
        })
    }

    pub fn checkpoint_and_suspend(&mut self) -> Result<()> {
        checkpoint_backend(&mut self.backend)?;
        run_cuda_checkpoint_toggle(self.pid)
    }

    pub fn resume_and_restore(&mut self) -> Result<()> {
        run_cuda_checkpoint_toggle(self.pid)?;
        restore_backend(&mut self.backend)
    }
}

fn checkpoint_backend(backend: &mut Backend) -> Result<()> {
    cuda::device_synchronize()?;
    backend.fs_mut().clear();
    let snapshot = cuda::memory_snapshot()?;
    if snapshot.len() > MAX_FILE_NUM {
        return Err(Error::Protocol(format!(
            "too many GPU allocations: {} > {MAX_FILE_NUM}",
            snapshot.len()
        )));
    }

    let stream = Stream::create()?;
    let mut offset = 0usize;
    for (idx, allocation) in snapshot.iter().enumerate() {
        if offset
            .checked_add(allocation.size)
            .is_none_or(|end| end > SHM_SIZE)
        {
            return Err(Error::Protocol(format!(
                "checkpoint storage too small: allocation {idx} exceeds {SHM_SIZE} bytes"
            )));
        }
        let mut copied = 0usize;
        while copied < allocation.size {
            let chunk = min(STAGING_BUF_SIZE, allocation.size - copied);
            unsafe {
                let device_src = (allocation.ptr + copied) as *const c_void;
                let host_dst = backend.staging().cast::<c_void>();
                stream.copy_async(host_dst, device_src, chunk, CudaMemcpyKind::DeviceToHost)?;
                stream.synchronize()?;
                copy::copy_nonoverlapping(
                    backend.checkpoint_storage().add(offset + copied),
                    backend.staging(),
                    chunk,
                );
            }
            copied += chunk;
        }

        let file = &mut backend.fs_mut().files[idx];
        file.ptr = allocation.ptr as *mut c_void;
        file.size = allocation.size as u64;
        file.offset = offset as u64;
        offset += allocation.size;
    }
    backend.fs_mut().file_num = snapshot.len() as u64;
    backend.fs_mut().tot_size = offset as u64;

    for allocation in snapshot {
        cuda::release_physical(allocation.ptr)?;
    }
    Ok(())
}

fn restore_backend(backend: &mut Backend) -> Result<()> {
    let stream = Stream::create()?;
    let file_num = backend.fs().file_num as usize;
    if file_num > MAX_FILE_NUM {
        return Err(Error::Protocol(format!(
            "checkpoint metadata contains too many files: {file_num}"
        )));
    }
    let storage_len = checkpoint_storage_len(backend.fs().tot_size)?;

    for idx in 0..file_num {
        let file = backend.fs().files[idx];
        let (offset, size) = checkpoint_file_range(idx, file.offset, file.size, storage_len)?;
        let ptr = file.ptr as usize;
        cuda::remap_physical(ptr)?;

        let mut copied = 0usize;
        while copied < size {
            let chunk = min(STAGING_BUF_SIZE, size - copied);
            unsafe {
                copy::copy_nonoverlapping(
                    backend.staging(),
                    backend.checkpoint_storage().add(offset + copied),
                    chunk,
                );
                let device_dst = (ptr + copied) as *mut c_void;
                let host_src = backend.staging().cast::<c_void>();
                stream.copy_async(device_dst, host_src, chunk, CudaMemcpyKind::HostToDevice)?;
                stream.synchronize()?;
            }
            copied += chunk;
        }
    }
    Ok(())
}

fn checkpoint_storage_len(tot_size: u64) -> Result<usize> {
    let len = usize::try_from(tot_size).map_err(|_| {
        Error::Protocol(format!(
            "checkpoint metadata total size is too large: {tot_size} bytes"
        ))
    })?;
    if len > SHM_SIZE {
        return Err(Error::Protocol(format!(
            "checkpoint metadata total size exceeds storage: {len} > {SHM_SIZE}"
        )));
    }
    Ok(len)
}

fn checkpoint_file_range(
    idx: usize,
    offset: u64,
    size: u64,
    storage_len: usize,
) -> Result<(usize, usize)> {
    let offset = usize::try_from(offset).map_err(|_| {
        Error::Protocol(format!(
            "checkpoint metadata offset is too large for file {idx}: {offset}"
        ))
    })?;
    let size = usize::try_from(size).map_err(|_| {
        Error::Protocol(format!(
            "checkpoint metadata size is too large for file {idx}: {size}"
        ))
    })?;
    if offset.checked_add(size).is_none_or(|end| end > storage_len) {
        return Err(Error::Protocol(format!(
            "checkpoint metadata range is out of bounds for file {idx}: offset {offset}, size {size}, storage {storage_len}"
        )));
    }
    Ok((offset, size))
}

pub fn run_cuda_checkpoint_toggle(pid: libc::pid_t) -> Result<()> {
    let binary = cuda_checkpoint_binary();
    let status = Command::new(&binary)
        .args(["--toggle", "--pid", &pid.to_string()])
        .status()
        .map_err(Error::Io)?;
    if status.success() {
        Ok(())
    } else {
        Err(Error::Process(format!(
            "cuda-checkpoint --toggle --pid {pid} failed with {status}"
        )))
    }
}

fn cuda_checkpoint_binary() -> PathBuf {
    if let Ok(path) = env::var("GPUCR_CUDA_CHECKPOINT") {
        return PathBuf::from(path);
    }
    let repo_binary = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint");
    if repo_binary.exists() {
        repo_binary
    } else {
        PathBuf::from("cuda-checkpoint")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_protocol_error(result: Result<impl Sized>) {
        assert!(matches!(result, Err(Error::Protocol(_))));
    }

    #[test]
    fn accepts_checkpoint_range_inside_declared_storage() {
        assert_eq!(
            checkpoint_file_range(0, 128, 256, 1024).unwrap(),
            (128, 256)
        );
    }

    #[test]
    fn rejects_checkpoint_range_that_exceeds_declared_storage() {
        assert_protocol_error(checkpoint_file_range(0, 900, 200, 1024));
    }

    #[test]
    fn rejects_checkpoint_range_overflow() {
        assert_protocol_error(checkpoint_file_range(0, u64::MAX, 2, SHM_SIZE));
    }

    #[test]
    fn rejects_checkpoint_storage_larger_than_mapping() {
        assert_protocol_error(checkpoint_storage_len((SHM_SIZE as u64) + 1));
    }
}
