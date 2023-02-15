use super::*;

use std::alloc::{AllocError, Allocator};
use std::ffi::c_void;
use std::ptr::{addr_of_mut, NonNull};

// use gpu_ffi::bindings::{bc_free_host, bc_malloc_host};

#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CudaAllocator;

unsafe impl Allocator for CudaAllocator {
    fn allocate(&self, layout: std::alloc::Layout) -> Result<NonNull<[u8]>, AllocError> {
        let size = layout.size();
        let mut raw_ptr: *mut u8 = std::ptr::null_mut();
        unsafe {
            if bc_malloc_host(addr_of_mut!(raw_ptr) as *mut *mut c_void, size as u64) != 0 {
                return Err(AllocError);
            }
        }
        let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(ptr, size))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: std::alloc::Layout) {
        if bc_free_host(ptr.as_ptr() as *mut c_void) != 0 {
            panic!("can't deallocate")
        }
    }
}

mod test_allocator {

    #[ignore]
    #[test]
    fn test_cuda_allocator() {
        use super::*;
        let log_degree = if let Ok(log_base) = std::env::var("LOG_BASE") {
            log_base.parse::<usize>().unwrap()
        } else {
            4usize
        };

        let size = 1 << log_degree;
        println!("start");
        let mut v = Vec::with_capacity_in(size, CudaAllocator);
        println!("push");
        for idx in 0..size {
            v.push(idx as u8);
        }
        println!("sleep");
        std::thread::sleep(std::time::Duration::from_secs(5));
        println!("end");
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}
