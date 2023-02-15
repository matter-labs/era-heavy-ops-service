use super::*;
use bellman::Engine;
pub use gpu_ffi::bindings::*;
use std::{
    ffi::c_void,
    ptr::addr_of_mut,
    sync::{Arc, Mutex},
};
// use super::bindings::*;

#[cfg(feature = "allocator")]
pub mod cuda_allocator;
#[cfg(feature = "allocator")]
pub use cuda_allocator::*;
pub mod async_vec;
mod context;
mod device_arithmetic;
mod device_buf;
mod device_heavy_ops;
mod error;
mod event;
mod stream;

pub use context::*;
pub use device_arithmetic::*;
pub use device_buf::*;
pub use error::*;
pub use event::*;
pub use stream::*;

#[derive(Clone, Debug, Default)]
pub struct DeviceMemoryInfo {
    pub free: u64,
    pub total: u64,
}

pub fn set_device(device_id: usize) -> GpuResult<usize> {
    let result = unsafe { bc_set_device(device_id as i32) };
    if result != 0 {
        return Err(GpuError::SetDeviceErr(result));
    }
    Ok(device_id)
}

pub fn devices() -> GpuResult<i32> {
    let mut count = 0;
    let result = unsafe { bc_get_device_count(std::ptr::addr_of_mut!(count)) };
    if result != 0 {
        return Err(GpuError::DeviceGetCountErr(result));
    }
    Ok(count)
}

pub fn device_info(device_id: i32) -> GpuResult<DeviceMemoryInfo> {
    let mut free = 0;
    let mut total = 0;
    let result = unsafe {
        let result = bc_set_device(device_id);
        assert_eq!(result, 0);
        bc_mem_get_info(std::ptr::addr_of_mut!(free), std::ptr::addr_of_mut!(total))
    };
    if result != 0 {
        return Err(GpuError::DeviceGetDeviceMemoryInfoErr(result));
    }
    Ok(DeviceMemoryInfo { free, total })
}
