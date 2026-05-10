use crate::constants::{COPY_THRESHOLD, NUM_COPY_THREADS};

/// Copy host bytes with the same coarse-grained split used by the C++ path.
///
/// # Safety
///
/// `dst` and `src` must be valid for `size` bytes and must not overlap.
pub unsafe fn copy_nonoverlapping(dst: *mut u8, src: *const u8, size: usize) {
    if size < COPY_THRESHOLD || NUM_COPY_THREADS <= 1 {
        std::ptr::copy_nonoverlapping(src, dst, size);
        return;
    }

    let chunk_size = size.div_ceil(NUM_COPY_THREADS);
    let dst_addr = dst as usize;
    let src_addr = src as usize;
    std::thread::scope(|scope| {
        for thread_id in 0..NUM_COPY_THREADS {
            let offset = thread_id * chunk_size;
            if offset >= size {
                break;
            }
            let this_chunk = chunk_size.min(size - offset);
            scope.spawn(move || {
                let dst = (dst_addr + offset) as *mut u8;
                let src = (src_addr + offset) as *const u8;
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst, this_chunk);
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn copies_small_buffer() {
        let src = vec![0x5au8; 4096];
        let mut dst = vec![0u8; src.len()];
        unsafe {
            super::copy_nonoverlapping(dst.as_mut_ptr(), src.as_ptr(), src.len());
        }
        assert_eq!(src, dst);
    }
}
