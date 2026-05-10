use std::os::raw::{c_int, c_void};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread;

use gpucr::constants::signals;
use gpucr::cuda;
use gpucr::runtime::Runtime;

static SIGNAL_READ_FD: AtomicI32 = AtomicI32::new(-1);
static SIGNAL_WRITE_FD: AtomicI32 = AtomicI32::new(-1);
static ACTIVE_PID: AtomicI32 = AtomicI32::new(0);
static RUNTIME_PID: AtomicI32 = AtomicI32::new(0);
static INIT_LOCK: Mutex<()> = Mutex::new(());
static RUNTIME: OnceLock<Mutex<Option<Runtime>>> = OnceLock::new();

#[used]
#[cfg_attr(target_os = "linux", link_section = ".init_array")]
static INIT_ARRAY: extern "C" fn() = init_library;

extern "C" fn init_library() {
    ensure_signal_thread();
}

fn ensure_signal_thread() {
    let pid = unsafe { libc::getpid() };
    if ACTIVE_PID.load(Ordering::SeqCst) == pid {
        return;
    }

    let Ok(_guard) = INIT_LOCK.lock() else {
        return;
    };
    if ACTIVE_PID.load(Ordering::SeqCst) == pid {
        return;
    }

    close_signal_pipe();
    let mut fds = [0; 2];
    let pipe_rc = unsafe { libc::pipe(fds.as_mut_ptr()) };
    if pipe_rc != 0 {
        eprintln!(
            "gpucr: failed to create signal pipe: {}",
            std::io::Error::last_os_error()
        );
        return;
    }
    SIGNAL_READ_FD.store(fds[0], Ordering::SeqCst);
    SIGNAL_WRITE_FD.store(fds[1], Ordering::SeqCst);

    unsafe {
        let handler = signal_handler as *const () as libc::sighandler_t;
        libc::signal(signals::cr_init(), handler);
        libc::signal(signals::cr_checkpoint(), handler);
        libc::signal(signals::cr_restore(), handler);
    }

    ACTIVE_PID.store(pid, Ordering::SeqCst);
    thread::spawn(signal_loop);
}

fn close_signal_pipe() {
    let read_fd = SIGNAL_READ_FD.swap(-1, Ordering::SeqCst);
    let write_fd = SIGNAL_WRITE_FD.swap(-1, Ordering::SeqCst);
    unsafe {
        if read_fd >= 0 {
            libc::close(read_fd);
        }
        if write_fd >= 0 && write_fd != read_fd {
            libc::close(write_fd);
        }
    }
}

extern "C" fn signal_handler(signum: c_int) {
    let fd = SIGNAL_WRITE_FD.load(Ordering::SeqCst);
    if fd < 0 {
        return;
    }
    let bytes = signum.to_ne_bytes();
    unsafe {
        libc::write(fd, bytes.as_ptr().cast::<c_void>(), bytes.len());
    }
}

fn signal_loop() {
    let fd = SIGNAL_READ_FD.load(Ordering::SeqCst);
    if fd < 0 {
        return;
    }

    loop {
        let mut bytes = [0u8; std::mem::size_of::<c_int>()];
        let mut read_len = 0usize;
        while read_len < bytes.len() {
            let rc = unsafe {
                libc::read(
                    fd,
                    bytes[read_len..].as_mut_ptr().cast::<c_void>(),
                    bytes.len() - read_len,
                )
            };
            if rc <= 0 {
                return;
            }
            read_len += rc as usize;
        }
        let signum = c_int::from_ne_bytes(bytes);
        if signum == signals::cr_init()
            || signum == signals::cr_checkpoint()
            || signum == signals::cr_restore()
        {
            if let Err(err) = handle_signal() {
                eprintln!("gpucr: control signal failed: {err}");
            }
        }
    }
}

fn handle_signal() -> gpucr::Result<()> {
    let pid = unsafe { libc::getpid() };
    let runtime = RUNTIME.get_or_init(|| Mutex::new(None));
    let mut guard = runtime
        .lock()
        .map_err(|_| gpucr::Error::Protocol("runtime mutex poisoned".to_string()))?;
    if RUNTIME_PID.load(Ordering::SeqCst) != pid {
        *guard = None;
        RUNTIME_PID.store(pid, Ordering::SeqCst);
    }
    if guard.is_none() {
        *guard = Some(Runtime::new(pid)?);
    }
    let Some(runtime) = guard.as_mut() else {
        return Err(gpucr::Error::Protocol(
            "runtime failed to initialize".to_string(),
        ));
    };
    runtime.handle_control_message()
}

#[no_mangle]
/// CUDA runtime allocation hook.
///
/// # Safety
///
/// `dev_ptr` must satisfy CUDA's `cudaMalloc` output-pointer contract.
pub unsafe extern "C" fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int {
    ensure_signal_thread();
    cuda::vmm_alloc(dev_ptr, size)
}

#[no_mangle]
/// CUDA runtime free hook.
///
/// # Safety
///
/// `ptr` must be a CUDA device pointer or null, matching CUDA's `cudaFree`
/// contract.
pub unsafe extern "C" fn cudaFree(ptr: *mut c_void) -> c_int {
    ensure_signal_thread();
    let code = cuda::vmm_free(ptr);
    if code == cuda::CUDA_SUCCESS_RT {
        return code;
    }
    fallback_cuda_free(ptr).unwrap_or(code)
}

type CudaFree = unsafe extern "C" fn(*mut c_void) -> c_int;

fn fallback_cuda_free(ptr: *mut c_void) -> Option<c_int> {
    let name = b"cudaFree\0";
    let sym = unsafe { libc::dlsym(libc::RTLD_NEXT, name.as_ptr().cast()) };
    if sym.is_null() {
        return None;
    }
    let func: CudaFree = unsafe { std::mem::transmute(sym) };
    Some(unsafe { func(ptr) })
}
