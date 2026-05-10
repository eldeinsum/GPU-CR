pub const CONTROL_PATH: &str = "/mnt/huge-ckpt/control";
pub const CONTROL_FILE_PREFIX: &str = "/mnt/huge-ckpt/control-";
pub const CKPT_DIR: &str = "/mnt/huge-ckpt";

pub const INIT_MSG: i32 = 10;
pub const CKPT_MSG: i32 = 11;
pub const RESTORE_MSG: i32 = 12;
pub const FINISH_MSG: i32 = 0;

pub const MAX_FILE_NUM: usize = 4096;
pub const SHM_SIZE: usize = 50 * 1024 * 1024 * 1024;
pub const STAGING_BUF_SIZE: usize = 1024 * 1024 * 1024;
pub const STAGING_BUF_NUM: usize = 2;

pub const TWO_MB: usize = 2 * 1024 * 1024;
pub const HUGE_PAGE_SIZE: usize = TWO_MB;
pub const NUM_COPY_THREADS: usize = 4;
pub const COPY_THRESHOLD: usize = 1 << 29;

pub fn round_up_2mb(size: usize) -> usize {
    (size + TWO_MB - 1) & !(TWO_MB - 1)
}

pub mod signals {
    pub fn cr_init() -> i32 {
        libc::SIGRTMAX()
    }

    pub fn cr_checkpoint() -> i32 {
        libc::SIGUSR1
    }

    pub fn cr_restore() -> i32 {
        libc::SIGUSR2
    }
}
