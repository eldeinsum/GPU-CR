use crate::constants::MAX_FILE_NUM;
use crate::{Error, Result};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SharedMemFile {
    pub size: u64,
    pub ptr: *mut libc::c_void,
    pub offset: u64,
}

impl Default for SharedMemFile {
    fn default() -> Self {
        Self {
            size: 0,
            ptr: std::ptr::null_mut(),
            offset: 0,
        }
    }
}

#[repr(C)]
pub struct SharedMemFs {
    pub tot_size: u64,
    pub file_num: u64,
    pub files: [SharedMemFile; MAX_FILE_NUM],
}

impl SharedMemFs {
    pub fn clear(&mut self) {
        self.tot_size = 0;
        self.file_num = 0;
        self.files.fill(SharedMemFile::default());
    }
}

pub const CONTROL_PATH_CAPACITY: usize = 256;
pub const CONTROL_PATH_MAX_LEN: usize = CONTROL_PATH_CAPACITY - 1;

#[repr(C)]
pub struct SignalControls {
    pub msg: libc::c_int,
    pub status: libc::c_int,
    pub restore_path: [u8; CONTROL_PATH_CAPACITY],
    pub checkpoint_path: [u8; CONTROL_PATH_CAPACITY],
}

impl SignalControls {
    pub fn checkpoint_path(&self) -> String {
        nul_terminated(&self.checkpoint_path)
    }

    pub fn restore_path(&self) -> String {
        nul_terminated(&self.restore_path)
    }

    pub fn set_checkpoint_path(&mut self, path: &str) -> Result<()> {
        set_buf(&mut self.checkpoint_path, path)
    }

    pub fn set_restore_path(&mut self, path: &str) -> Result<()> {
        set_buf(&mut self.restore_path, path)
    }
}

fn nul_terminated(buf: &[u8]) -> String {
    let len = buf.iter().position(|byte| *byte == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..len]).into_owned()
}

fn set_buf(buf: &mut [u8], value: &str) -> Result<()> {
    if value.len() >= buf.len() {
        return Err(Error::Protocol(format!(
            "control path is too long: {} bytes, maximum is {}",
            value.len(),
            buf.len() - 1
        )));
    }
    buf.fill(0);
    let bytes = value.as_bytes();
    buf[..bytes.len()].copy_from_slice(bytes);
    Ok(())
}
