use crate::constants::MAX_FILE_NUM;

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

#[repr(C)]
pub struct SignalControls {
    pub msg: libc::c_int,
    pub status: libc::c_int,
    pub restore_path: [u8; 256],
    pub checkpoint_path: [u8; 256],
}

impl SignalControls {
    pub fn checkpoint_path(&self) -> String {
        nul_terminated(&self.checkpoint_path)
    }

    pub fn restore_path(&self) -> String {
        nul_terminated(&self.restore_path)
    }

    pub fn set_checkpoint_path(&mut self, path: &str) {
        set_buf(&mut self.checkpoint_path, path);
    }

    pub fn set_restore_path(&mut self, path: &str) {
        set_buf(&mut self.restore_path, path);
    }
}

fn nul_terminated(buf: &[u8]) -> String {
    let len = buf.iter().position(|byte| *byte == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..len]).into_owned()
}

fn set_buf(buf: &mut [u8], value: &str) {
    buf.fill(0);
    let bytes = value.as_bytes();
    let len = bytes.len().min(buf.len().saturating_sub(1));
    buf[..len].copy_from_slice(&bytes[..len]);
}
