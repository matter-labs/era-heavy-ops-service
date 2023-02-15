use super::*;

unsafe extern "C" fn callback_wrapper<F: FnMut()>(closure: *mut ::std::os::raw::c_void) {
    let user_data = &mut *(closure as *mut F);
    user_data();
}

pub fn call_host_fn<F: FnMut()>(stream: bc_stream, cb: &F) -> Result<(), GpuError> {
    let callback_data = cb as *const _ as *mut ::std::os::raw::c_void;

    unsafe {
        if bc_launch_host_fn(
            stream,
            Some(callback_wrapper::<F>),
            callback_data,
        ) != 0
        {
            return Err(GpuError::SchedulingErr);
        }
    }
    return Ok(());
}

pub fn malloc_from_pool_async(
    ptr: *mut *mut ::std::os::raw::c_void,
    size: usize,
    pool: bc_mem_pool,
    stream: bc_stream,
) -> Result<(), GpuError> {
    if unsafe { bc_malloc_from_pool_async(ptr, size as size_t, pool, stream) } != 0 {
        return Err(GpuError::AsyncPoolMallocErr);
    }
    Ok(())
}

pub fn device_disable_peer_access(
    device_id: usize,
) -> Result<(), GpuError> {
    if unsafe { bc_device_disable_peer_access(device_id as i32) } != 0 {
        return Err(GpuError::DevicePeerAccessErr);
    }
    Ok(())
}

pub fn device_enable_peer_access(
    device_id: i32,
) -> Result<(), GpuError> {
    if unsafe { bc_device_enable_peer_access(device_id) } != 0 {
        return Err(GpuError::DevicePeerAccessErr);
    }
    Ok(())
}

pub fn mem_pool_disable_peer_access(
    pool: bc_mem_pool,
    device_id: usize,
) -> Result<(), GpuError> {
    if unsafe { bc_mem_pool_disable_peer_access(pool, device_id as i32) } != 0 {
        return Err(GpuError::MemPoolPeerAccessErr);
    }
    Ok(())
}

pub fn mem_pool_enable_peer_access(
    pool: bc_mem_pool,
    device_id: i32,
) -> Result<(), GpuError> {
    if unsafe { bc_mem_pool_enable_peer_access(pool, device_id) } != 0 {
        return Err(GpuError::MemPoolPeerAccessErr);
    }
    Ok(())
}

pub fn memcpy_async(
    dst: *mut ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    size: usize,
    stream: bc_stream,
) -> Result<(), GpuError> {
    if unsafe { bc_memcpy_async(dst, src, size as u64, stream) } != 0 {
        return Err(GpuError::AsyncMemcopyErr);
    }
    Ok(())
}

pub fn free_async(ptr: *mut ::std::os::raw::c_void, stream: bc_stream) -> Result<(), GpuError> {
    if unsafe { bc_free_async(ptr, stream) } != 0 {
        return Err(GpuError::AsyncMemcopyErr);
    }
    Ok(())
}

pub fn alloc_and_copy(
    ctx: &GpuContext,
    h_values: &[u8],
    stream: bc_stream,
) -> Result<*mut c_void, GpuError> {
    let len = h_values.len();

    let mut d_values = std::ptr::null_mut();
    malloc_from_pool_async(addr_of_mut!(d_values), len, ctx.get_mem_pool(), stream)?;
    memcpy_async(d_values, h_values.as_ptr() as *const c_void, len, stream)?;

    Ok(d_values)
}
pub fn copy_and_free(
    h_values: &mut [u8],
    d_values: *mut c_void,
    stream: bc_stream,
) -> Result<(), GpuError> {
    let len = h_values.len();
    memcpy_async(h_values.as_ptr() as *mut c_void, d_values, len, stream)?;
    free_async(d_values, stream)?;
    Ok(())
}

pub fn run_ntt(
    ctx: &GpuContext,
    inputs: *mut c_void,
    outputs: *mut c_void,
    log_values_count: u32,
    bits_reversed: bool,
    inverse: bool,
) -> Result<(), GpuError> {
    let cfg = ntt_configuration::new(
        ctx,
        inputs,
        outputs,
        log_values_count,
        bits_reversed,
        inverse,
    );
    if unsafe { ntt_execute_async(cfg) } != 0 {
        return Err(GpuError::NttExecErr);
    }

    Ok(())
}

impl bc_mem_pool {
    pub fn new(device_id: usize) -> Result<bc_mem_pool, GpuError> {
        let mut mem_pool = Self::null();
        let result = unsafe { bc_mem_pool_create(addr_of_mut!(mem_pool), device_id as i32) } == 0;
        if !result {
            return Err(GpuError::MemPoolCreateErr);
        }

        Ok(mem_pool)
    }

    pub fn null() -> bc_mem_pool {
        bc_mem_pool {
            handle: std::ptr::null_mut() as *mut c_void,
        }
    }
}

impl bc_stream {
    pub fn new() -> Result<bc_stream, GpuError> {
        let mut new = Self::null();
        if unsafe { bc_stream_create(new.as_mut_ptr(), true) } != 0 {
            return Err(GpuError::StremCreateErr);
        };

        Ok(new)
    }
    pub fn destroy(self) -> Result<(), GpuError> {
        if unsafe { bc_stream_destroy(self) } != 0 {
            return Err(GpuError::StreamDestroyErr);
        }

        Ok(())
    }

    pub fn wait(self, event: bc_event) -> Result<(), GpuError> {
        if unsafe { bc_stream_wait_event(self, event) } != 0 {
            return Err(GpuError::StreamWaitEventErr);
        }

        Ok(())
    }

    pub fn sync(self) -> Result<(), GpuError> {
        if unsafe { bc_stream_synchronize(self) } != 0 {
            return Err(GpuError::StreamSyncErr);
        }
        Ok(())
    }

    pub fn null() -> bc_stream {
        bc_stream {
            handle: std::ptr::null_mut() as *mut c_void,
        }
    }

    fn as_mut_ptr(&mut self) -> *mut bc_stream {
        addr_of_mut!(*self)
    }
}

impl bc_event {
    pub fn new() -> Result<bc_event, GpuError> {
        let mut event = bc_event::null();
        if unsafe { bc_event_create(addr_of_mut!(event), true, true) } != 0 {
            return Err(GpuError::EventCreateErr);
        }
        Ok(event)
    }

    pub fn record(self, stream: bc_stream) -> Result<(), GpuError> {
        if unsafe { bc_event_record(self, stream) } != 0 {
            return Err(GpuError::EventRecordErr);
        }

        Ok(())
    }
    pub fn destroy(self) -> Result<(), GpuError> {
        if unsafe { bc_event_destroy(self) } != 0 {
            return Err(GpuError::EventDestroyErr);
        }

        Ok(())
    }

    pub fn null() -> bc_event {
        bc_event {
            handle: std::ptr::null_mut() as *mut c_void,
        }
    }

    pub fn sync(self) ->  Result<(), GpuError> {
        if unsafe { bc_event_synchronize(self) } != 0 {
            return Err(GpuError::EventSyncErr);
        }

        Ok(())
    }
}

impl ntt_configuration {
    pub fn new_for_lde(
        ctx: &GpuContext,
        inputs: *const c_void,
        outputs: *mut c_void,
        log_values_count: u32,
        coset_index: usize,
        lde_factor: u32,
    ) -> Self {
        let log_extension_degree = log_2(lde_factor as usize);
        let coset_index = bitreverse(coset_index, log_extension_degree as usize);
        let mut this = Self::new(ctx, inputs as *mut c_void, outputs, log_values_count, false, false);
        this.coset_index = coset_index as u32;
        this.log_extension_degree = log_extension_degree;

        this
    }

    pub fn new(
        ctx: &GpuContext,
        inputs: *mut c_void,
        outputs: *mut c_void,
        log_values_count: u32,
        bits_reversed: bool,
        inverse: bool,
    ) -> Self {
        ntt_configuration {
            mem_pool: ctx.get_mem_pool(),
            stream: ctx.get_exec_stream(),
            inputs: inputs,
            outputs: outputs,
            log_values_count,
            bit_reversed_inputs: bits_reversed,
            inverse,
            h2d_copy_finished: bc_event::null(),
            h2d_copy_finished_callback: None,
            h2d_copy_finished_callback_data: std::ptr::null_mut() as *mut c_void,
            d2h_copy_finished: bc_event::null(),
            d2h_copy_finished_callback: None,
            d2h_copy_finished_callback_data: std::ptr::null_mut() as *mut c_void,
            can_overwrite_inputs: false,
            coset_index: 0,
            log_extension_degree: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DeviceMemoryInfo {
    pub free: u64,
    pub total: u64,
}
