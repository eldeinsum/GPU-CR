use std::ffi::CString;
use std::fs;
use std::io;
use std::path::Path;
use std::ptr;

use crate::constants::{CKPT_DIR, SHM_SIZE, STAGING_BUF_NUM, STAGING_BUF_SIZE};
use crate::cuda;
use crate::layout::SharedMemFs;
use crate::{Error, Result};

pub struct Backend {
    fs: *mut SharedMemFs,
    fs_len: usize,
    ckpt_storage: *mut u8,
    ckpt_storage_len: usize,
    staging: *mut u8,
    staging_len: usize,
    staging_registered: bool,
}

unsafe impl Send for Backend {}

impl Backend {
    pub fn setup(path: &str) -> Result<Self> {
        fs::create_dir_all(CKPT_DIR)?;
        let path = if path.is_empty() {
            format!("{CKPT_DIR}/default")
        } else {
            path.to_string()
        };
        let c_path = CString::new(path)?;
        let fd = unsafe {
            libc::open(
                c_path.as_ptr(),
                libc::O_RDWR | libc::O_CREAT,
                libc::S_IRUSR
                    | libc::S_IWUSR
                    | libc::S_IRGRP
                    | libc::S_IWGRP
                    | libc::S_IROTH
                    | libc::S_IWOTH,
            )
        };
        if fd < 0 {
            return Err(Error::Io(io::Error::last_os_error()));
        }
        if unsafe { libc::ftruncate(fd, SHM_SIZE as libc::off_t) } != 0 {
            let err = Error::Io(io::Error::last_os_error());
            unsafe {
                libc::close(fd);
            }
            return Err(err);
        }

        let ckpt_storage = unsafe {
            libc::mmap(
                ptr::null_mut(),
                SHM_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        unsafe {
            libc::close(fd);
        }
        if ckpt_storage == libc::MAP_FAILED {
            return Err(Error::Io(io::Error::last_os_error()));
        }

        let fs_len = std::mem::size_of::<SharedMemFs>();
        let fs = unsafe {
            libc::mmap(
                ptr::null_mut(),
                fs_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if fs == libc::MAP_FAILED {
            unsafe {
                libc::munmap(ckpt_storage, SHM_SIZE);
            }
            return Err(Error::Io(io::Error::last_os_error()));
        }

        let staging_len = STAGING_BUF_SIZE * STAGING_BUF_NUM;
        let staging = unsafe {
            libc::mmap(
                ptr::null_mut(),
                staging_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if staging == libc::MAP_FAILED {
            unsafe {
                libc::munmap(fs, fs_len);
                libc::munmap(ckpt_storage, SHM_SIZE);
            }
            return Err(Error::Io(io::Error::last_os_error()));
        }

        let mut backend = Self {
            fs: fs.cast::<SharedMemFs>(),
            fs_len,
            ckpt_storage: ckpt_storage.cast::<u8>(),
            ckpt_storage_len: SHM_SIZE,
            staging: staging.cast::<u8>(),
            staging_len,
            staging_registered: false,
        };
        backend.fs_mut().clear();
        Ok(backend)
    }

    pub fn setup_for_current_pid(path: &str) -> Result<Self> {
        let resolved = if path.is_empty() {
            format!("{CKPT_DIR}/{}", unsafe { libc::getpid() })
        } else {
            path.to_string()
        };
        Self::setup(&resolved)
    }

    pub fn fs(&self) -> &SharedMemFs {
        unsafe { &*self.fs }
    }

    pub fn fs_mut(&mut self) -> &mut SharedMemFs {
        unsafe { &mut *self.fs }
    }

    pub fn checkpoint_storage(&self) -> *mut u8 {
        self.ckpt_storage
    }

    pub fn staging(&self) -> *mut u8 {
        self.staging
    }

    pub fn try_register_staging_with_cuda(&mut self) {
        if self.staging_registered {
            return;
        }
        if unsafe {
            cuda::register_host_memory(self.staging.cast::<libc::c_void>(), self.staging_len)
        }
        .is_ok()
        {
            self.staging_registered = true;
        }
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        if self.staging_registered {
            let _ = unsafe { cuda::unregister_host_memory(self.staging.cast::<libc::c_void>()) };
        }
        unsafe {
            libc::munmap(self.fs.cast::<libc::c_void>(), self.fs_len);
            libc::munmap(
                self.ckpt_storage.cast::<libc::c_void>(),
                self.ckpt_storage_len,
            );
            libc::munmap(self.staging.cast::<libc::c_void>(), self.staging_len);
        }
    }
}

pub fn default_checkpoint_path_for_pid(pid: libc::pid_t) -> String {
    format!("{CKPT_DIR}/{pid}")
}

pub fn ensure_checkpoint_dir() -> Result<()> {
    fs::create_dir_all(Path::new(CKPT_DIR))?;
    Ok(())
}
