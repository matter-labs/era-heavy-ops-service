mod cuda_allocator;
mod device_buf;

pub use cuda_allocator::*;
use gpu_ffi::*;
pub use device_buf::*;
use std::{ffi::c_void, ptr::addr_of_mut, sync::Arc};
