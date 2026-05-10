use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uint, c_ulonglong, c_void};
use std::sync::{Mutex, OnceLock};

use crate::constants::round_up_2mb;
use crate::{Error, Result};

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUdeviceptr = u64;
type CUmemGenericAllocationHandle = u64;
type CudaError = c_int;
type CudaStream = *mut c_void;

const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_SUCCESS_RT: CudaError = 0;
pub const CUDA_ERROR_INVALID_VALUE: CudaError = 1;
pub const CUDA_ERROR_MEMORY_ALLOCATION: CudaError = 2;
pub const CUDA_ERROR_INITIALIZATION_ERROR: CudaError = 3;

const CU_MEM_HANDLE_TYPE_NONE: c_int = 0;
const CU_MEM_ACCESS_FLAGS_PROT_READWRITE: c_int = 0x3;
const CU_MEM_LOCATION_TYPE_DEVICE: c_int = 0x1;
const CU_MEM_ALLOCATION_TYPE_PINNED: c_int = 0x1;

#[repr(C)]
#[derive(Clone, Copy)]
struct CUmemLocation {
    type_: c_int,
    id: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CUmemAllocationPropAllocFlags {
    compression_type: u8,
    gpu_direct_rdma_capable: u8,
    usage: u16,
    reserved: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CUmemAllocationProp {
    type_: c_int,
    requested_handle_types: c_int,
    location: CUmemLocation,
    win32_handle_meta_data: *mut c_void,
    alloc_flags: CUmemAllocationPropAllocFlags,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CUmemAccessDesc {
    location: CUmemLocation,
    flags: c_int,
}

#[repr(i32)]
#[derive(Clone, Copy)]
pub enum CudaMemcpyKind {
    HostToDevice = 1,
    DeviceToHost = 2,
}

#[derive(Clone, Debug)]
pub struct AllocationSnapshot {
    pub ptr: usize,
    pub size: usize,
}

#[derive(Clone, Debug)]
struct Allocation {
    size: usize,
    aligned_size: usize,
    handle: Option<CUmemGenericAllocationHandle>,
}

#[derive(Default)]
struct CudaState {
    initialized: bool,
    context: usize,
    device: CUdevice,
    allocations: HashMap<usize, Allocation>,
}

static CUDA_STATE: OnceLock<Mutex<CudaState>> = OnceLock::new();

fn state() -> &'static Mutex<CudaState> {
    CUDA_STATE.get_or_init(|| Mutex::new(CudaState::default()))
}

#[link(name = "cuda")]
extern "C" {
    fn cuGetErrorString(error: CUresult, pstr: *mut *const c_char) -> CUresult;
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
    fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult;
    fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    fn cuMemAddressReserve(
        ptr: *mut CUdeviceptr,
        size: usize,
        alignment: usize,
        addr: CUdeviceptr,
        flags: c_ulonglong,
    ) -> CUresult;
    fn cuMemAddressFree(ptr: CUdeviceptr, size: usize) -> CUresult;
    fn cuMemCreate(
        handle: *mut CUmemGenericAllocationHandle,
        size: usize,
        prop: *const CUmemAllocationProp,
        flags: c_ulonglong,
    ) -> CUresult;
    fn cuMemRelease(handle: CUmemGenericAllocationHandle) -> CUresult;
    fn cuMemMap(
        ptr: CUdeviceptr,
        size: usize,
        offset: usize,
        handle: CUmemGenericAllocationHandle,
        flags: c_ulonglong,
    ) -> CUresult;
    fn cuMemUnmap(ptr: CUdeviceptr, size: usize) -> CUresult;
    fn cuMemSetAccess(
        ptr: CUdeviceptr,
        size: usize,
        desc: *const CUmemAccessDesc,
        count: usize,
    ) -> CUresult;
}

#[link(name = "cudart")]
extern "C" {
    #[link_name = "cudaGetErrorString"]
    fn cudart_get_error_string(error: CudaError) -> *const c_char;
    #[link_name = "cudaStreamCreate"]
    fn cudart_stream_create(stream: *mut CudaStream) -> CudaError;
    #[link_name = "cudaStreamSynchronize"]
    fn cudart_stream_synchronize(stream: CudaStream) -> CudaError;
    #[link_name = "cudaStreamDestroy"]
    fn cudart_stream_destroy(stream: CudaStream) -> CudaError;
    #[link_name = "cudaMemcpyAsync"]
    fn cudart_memcpy_async(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: CudaMemcpyKind,
        stream: CudaStream,
    ) -> CudaError;
    #[link_name = "cudaDeviceSynchronize"]
    fn cudart_device_synchronize() -> CudaError;
    #[link_name = "cudaHostRegister"]
    fn cudart_host_register(ptr: *mut c_void, size: usize, flags: c_uint) -> CudaError;
    #[link_name = "cudaHostUnregister"]
    fn cudart_host_unregister(ptr: *mut c_void) -> CudaError;
}

pub fn cuda_error_string(code: CudaError) -> String {
    unsafe {
        let ptr = cudart_get_error_string(code);
        if ptr.is_null() {
            return "unknown cuda runtime error".to_string();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

fn driver_error_string(code: CUresult) -> String {
    unsafe {
        let mut ptr = std::ptr::null();
        if cuGetErrorString(code, &mut ptr) != CUDA_SUCCESS || ptr.is_null() {
            return "unknown cuda driver error".to_string();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

fn check_driver(code: CUresult) -> Result<()> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda {
            code,
            message: driver_error_string(code),
        })
    }
}

fn check_runtime(code: CudaError) -> Result<()> {
    if code == CUDA_SUCCESS_RT {
        Ok(())
    } else {
        Err(Error::Cuda {
            code,
            message: cuda_error_string(code),
        })
    }
}

fn allocation_prop(device: CUdevice) -> CUmemAllocationProp {
    CUmemAllocationProp {
        type_: CU_MEM_ALLOCATION_TYPE_PINNED,
        requested_handle_types: CU_MEM_HANDLE_TYPE_NONE,
        location: CUmemLocation {
            type_: CU_MEM_LOCATION_TYPE_DEVICE,
            id: device,
        },
        win32_handle_meta_data: std::ptr::null_mut(),
        alloc_flags: CUmemAllocationPropAllocFlags {
            compression_type: 0,
            gpu_direct_rdma_capable: 0,
            usage: 0,
            reserved: [0; 4],
        },
    }
}

fn access_desc(device: CUdevice) -> CUmemAccessDesc {
    CUmemAccessDesc {
        location: CUmemLocation {
            type_: CU_MEM_LOCATION_TYPE_DEVICE,
            id: device,
        },
        flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    }
}

pub fn ensure_context() -> Result<()> {
    let mut guard = state()
        .lock()
        .map_err(|_| Error::Protocol("cuda state mutex poisoned".to_string()))?;
    ensure_context_locked(&mut guard)
}

fn ensure_context_locked(state: &mut CudaState) -> Result<()> {
    if state.initialized {
        return Ok(());
    }
    unsafe {
        check_driver(cuInit(0))?;
        let mut ctx: CUcontext = std::ptr::null_mut();
        check_driver(cuCtxGetCurrent(&mut ctx))?;
        let mut device: CUdevice = 0;
        if ctx.is_null() {
            check_driver(cuDeviceGet(&mut device, 0))?;
            check_driver(cuCtxCreate_v2(&mut ctx, 0, device))?;
        } else {
            check_driver(cuCtxGetDevice(&mut device))?;
        }
        state.context = ctx as usize;
        state.device = device;
        state.initialized = true;
    }
    Ok(())
}

/// Allocate GPU memory through CUDA VMM and write the resulting device pointer.
///
/// # Safety
///
/// `dev_ptr` must be valid for writes of one pointer-sized value, matching the
/// safety contract of CUDA's `cudaMalloc`.
pub unsafe fn vmm_alloc(dev_ptr: *mut *mut c_void, size: usize) -> CudaError {
    if dev_ptr.is_null() {
        return CUDA_ERROR_MEMORY_ALLOCATION;
    }
    if size == 0 {
        *dev_ptr = std::ptr::null_mut();
        return CUDA_SUCCESS_RT;
    }
    match vmm_alloc_result(size) {
        Ok(ptr) => {
            *dev_ptr = ptr as *mut c_void;
            CUDA_SUCCESS_RT
        }
        Err(err) => {
            eprintln!("gpucr cudaMalloc failed: {err}");
            CUDA_ERROR_MEMORY_ALLOCATION
        }
    }
}

fn vmm_alloc_result(size: usize) -> Result<usize> {
    let aligned_size = round_up_2mb(size);
    let mut guard = state()
        .lock()
        .map_err(|_| Error::Protocol("cuda state mutex poisoned".to_string()))?;
    ensure_context_locked(&mut guard)?;
    let device = guard.device;
    let prop = allocation_prop(device);
    let desc = access_desc(device);

    unsafe {
        let mut ptr: CUdeviceptr = 0;
        let mut handle: CUmemGenericAllocationHandle = 0;
        check_driver(cuMemAddressReserve(&mut ptr, aligned_size, 0, 0, 0))?;
        if let Err(err) = check_driver(cuMemCreate(&mut handle, aligned_size, &prop, 0)) {
            let _ = cuMemAddressFree(ptr, aligned_size);
            return Err(err);
        }
        if let Err(err) = check_driver(cuMemMap(ptr, aligned_size, 0, handle, 0)) {
            let _ = cuMemRelease(handle);
            let _ = cuMemAddressFree(ptr, aligned_size);
            return Err(err);
        }
        if let Err(err) = check_driver(cuMemSetAccess(ptr, aligned_size, &desc, 1)) {
            let _ = cuMemUnmap(ptr, aligned_size);
            let _ = cuMemRelease(handle);
            let _ = cuMemAddressFree(ptr, aligned_size);
            return Err(err);
        }
        guard.allocations.insert(
            ptr as usize,
            Allocation {
                size,
                aligned_size,
                handle: Some(handle),
            },
        );
        Ok(ptr as usize)
    }
}

pub fn vmm_free(ptr: *mut c_void) -> CudaError {
    if ptr.is_null() {
        return CUDA_SUCCESS_RT;
    }
    match vmm_free_result(ptr as usize) {
        Ok(()) => CUDA_SUCCESS_RT,
        Err(Error::Protocol(_)) => CUDA_ERROR_INVALID_VALUE,
        Err(err) => {
            eprintln!("gpucr cudaFree failed: {err}");
            CUDA_ERROR_MEMORY_ALLOCATION
        }
    }
}

fn vmm_free_result(ptr: usize) -> Result<()> {
    let mut guard = state()
        .lock()
        .map_err(|_| Error::Protocol("cuda state mutex poisoned".to_string()))?;
    let Some(allocation) = guard.allocations.remove(&ptr) else {
        return Err(Error::Protocol(format!(
            "cannot free untracked allocation {ptr:#x}"
        )));
    };
    unsafe {
        if let Some(handle) = allocation.handle {
            check_driver(cuMemUnmap(ptr as CUdeviceptr, allocation.aligned_size))?;
            check_driver(cuMemRelease(handle))?;
        }
        check_driver(cuMemAddressFree(
            ptr as CUdeviceptr,
            allocation.aligned_size,
        ))?;
    }
    Ok(())
}

pub fn memory_snapshot() -> Vec<AllocationSnapshot> {
    let mut snapshot: Vec<_> = state()
        .lock()
        .ok()
        .map(|guard| {
            guard
                .allocations
                .iter()
                .map(|(ptr, allocation)| AllocationSnapshot {
                    ptr: *ptr,
                    size: allocation.size,
                })
                .collect()
        })
        .unwrap_or_default();
    snapshot.sort_by_key(|allocation| allocation.ptr);
    snapshot
}

pub fn release_physical(ptr: usize) -> Result<()> {
    let mut guard = state()
        .lock()
        .map_err(|_| Error::Protocol("cuda state mutex poisoned".to_string()))?;
    let Some(allocation) = guard.allocations.get_mut(&ptr) else {
        return Err(Error::Protocol(format!(
            "cannot release unknown allocation {ptr:#x}"
        )));
    };
    let Some(handle) = allocation.handle.take() else {
        return Ok(());
    };
    unsafe {
        check_driver(cuMemUnmap(ptr as CUdeviceptr, allocation.aligned_size))?;
        check_driver(cuMemRelease(handle))?;
    }
    Ok(())
}

pub fn remap_physical(ptr: usize) -> Result<()> {
    let mut guard = state()
        .lock()
        .map_err(|_| Error::Protocol("cuda state mutex poisoned".to_string()))?;
    ensure_context_locked(&mut guard)?;
    let device = guard.device;
    let Some(allocation) = guard.allocations.get_mut(&ptr) else {
        return Err(Error::Protocol(format!(
            "cannot remap unknown allocation {ptr:#x}"
        )));
    };
    if allocation.handle.is_some() {
        return Ok(());
    }
    let prop = allocation_prop(device);
    let desc = access_desc(device);
    unsafe {
        let mut handle = 0;
        check_driver(cuMemCreate(&mut handle, allocation.aligned_size, &prop, 0))?;
        if let Err(err) = check_driver(cuMemMap(
            ptr as CUdeviceptr,
            allocation.aligned_size,
            0,
            handle,
            0,
        )) {
            let _ = cuMemRelease(handle);
            return Err(err);
        }
        if let Err(err) = check_driver(cuMemSetAccess(
            ptr as CUdeviceptr,
            allocation.aligned_size,
            &desc,
            1,
        )) {
            let _ = cuMemUnmap(ptr as CUdeviceptr, allocation.aligned_size);
            let _ = cuMemRelease(handle);
            return Err(err);
        }
        allocation.handle = Some(handle);
    }
    Ok(())
}

pub struct Stream {
    raw: CudaStream,
}

pub fn device_synchronize() -> Result<()> {
    unsafe { check_runtime(cudart_device_synchronize()) }
}

/// Register a host memory range with CUDA for faster transfers.
///
/// # Safety
///
/// `ptr` must refer to a live host allocation of at least `size` bytes.
pub unsafe fn register_host_memory(ptr: *mut c_void, size: usize) -> Result<()> {
    unsafe { check_runtime(cudart_host_register(ptr, size, 0)) }
}

/// Unregister a host memory range previously registered with CUDA.
///
/// # Safety
///
/// `ptr` must be the base pointer passed to `register_host_memory`.
pub unsafe fn unregister_host_memory(ptr: *mut c_void) -> Result<()> {
    unsafe { check_runtime(cudart_host_unregister(ptr)) }
}

impl Stream {
    pub fn create() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        unsafe {
            check_runtime(cudart_stream_create(&mut raw))?;
        }
        Ok(Self { raw })
    }

    /// Copy memory asynchronously on this CUDA stream.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must be valid for `count` bytes for the requested
    /// direction and must satisfy CUDA runtime pointer requirements.
    pub unsafe fn copy_async(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: CudaMemcpyKind,
    ) -> Result<()> {
        unsafe { check_runtime(cudart_memcpy_async(dst, src, count, kind, self.raw)) }
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { check_runtime(cudart_stream_synchronize(self.raw)) }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = cudart_stream_destroy(self.raw);
            }
        }
    }
}
