use std::ffi::CString;
use std::fs::{self, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicI32, Ordering};
use std::thread;
use std::time::Duration;

use crate::constants::{CONTROL_FILE_PREFIX, CONTROL_PATH, FINISH_MSG, HUGE_PAGE_SIZE};
use crate::layout::SignalControls;
use crate::{Error, Result};

pub struct Comm {
    ptr: *mut SignalControls,
    len: usize,
}

unsafe impl Send for Comm {}

impl Comm {
    pub fn for_pid(pid: libc::pid_t) -> Result<Self> {
        fs::create_dir_all(Path::new(CONTROL_PATH).parent().unwrap())?;
        let path = format!("{CONTROL_FILE_PREFIX}{pid}");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let len = HUGE_PAGE_SIZE;
        file.set_len(len as u64)?;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(Error::Io(io::Error::last_os_error()));
        }
        Ok(Self {
            ptr: ptr.cast::<SignalControls>(),
            len,
        })
    }

    pub fn controls(&self) -> &SignalControls {
        unsafe { &*self.ptr }
    }

    pub fn controls_mut(&mut self) -> &mut SignalControls {
        unsafe { &mut *self.ptr }
    }

    pub fn recv_msg(&self) -> i32 {
        unsafe { std::ptr::read_volatile(&(*self.ptr).msg) }
    }

    pub fn send_msg(&self, msg: i32) {
        unsafe {
            std::ptr::write_volatile(&mut (*self.ptr).msg, msg);
        }
    }

    pub fn send_finish(&self) {
        self.send_msg(FINISH_MSG);
    }

    pub fn wait_for_finish(&self) {
        while self.recv_msg() != FINISH_MSG {
            thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Comm {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.cast::<libc::c_void>(), self.len);
        }
    }
}

pub fn next_control_id() -> Result<i32> {
    fs::create_dir_all(Path::new(CONTROL_PATH).parent().unwrap())?;
    let path = CString::new(CONTROL_PATH)?;
    let fd = unsafe {
        libc::open(
            path.as_ptr(),
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

    let len = std::mem::size_of::<AtomicI32>();
    let truncate = unsafe { libc::ftruncate(fd, len as libc::off_t) };
    if truncate != 0 {
        let err = Error::Io(io::Error::last_os_error());
        unsafe {
            libc::close(fd);
        }
        return Err(err);
    }

    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        )
    };
    unsafe {
        libc::close(fd);
    }
    if ptr == libc::MAP_FAILED {
        return Err(Error::Io(io::Error::last_os_error()));
    }

    let id = unsafe {
        let atomic = &*(ptr.cast::<AtomicI32>());
        atomic.fetch_add(1, Ordering::SeqCst)
    };
    unsafe {
        libc::munmap(ptr, len);
    }
    Ok(id)
}
