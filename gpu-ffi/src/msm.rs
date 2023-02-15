use super::*;

// GPU is smart enough that can determine memory residence from some attributes
pub fn msm(ctx: &GpuContext, scalars: &[u8]) -> Result<Vec<u8>, GpuError> {
    let len = scalars.len();
    let d_scalars = alloc_and_copy(ctx, scalars, ctx.get_h2d_stream())?;

    let h2d_finished = bc_event::new()?;
    h2d_finished.record(ctx.get_h2d_stream())?;
    ctx.get_exec_stream().wait(h2d_finished)?;

    raw_msm(ctx, d_scalars, len)
}

pub fn raw_msm(ctx: &GpuContext, d_scalars: *mut c_void, len: usize) -> Result<Vec<u8>, GpuError> {
    let log_scalars_count = log_2((len / FIELD_ELEMENT_LEN) as usize);
    let result_buf_len = 3 * FIELD_ELEMENT_LEN * 256;
    let mut d_result = std::ptr::null_mut();
    malloc_from_pool_async(
        addr_of_mut!(d_result),
        result_buf_len,
        ctx.get_mem_pool(),
        ctx.get_h2d_stream(),
    )?;

    let cfg = msm_configuration {
        mem_pool: ctx.get_mem_pool(),
        stream: ctx.get_exec_stream(),
        bases: ctx.get_bases_ptr_mut(),
        scalars: d_scalars,
        results: d_result,
        log_scalars_count: log_scalars_count,
        h2d_copy_finished: bc_event::null(),
        h2d_copy_finished_callback: None,
        h2d_copy_finished_callback_data: std::ptr::null_mut() as *mut c_void,
        d2h_copy_finished: bc_event::null(),
        d2h_copy_finished_callback: None,
        d2h_copy_finished_callback_data: std::ptr::null_mut() as *mut c_void,
    };

    unsafe {
        if msm_execute_async(cfg) != 0 {
            return Err(GpuError::SchedulingErr);
        };
    }

    let exec_finished = bc_event::new()?;
    exec_finished.record(ctx.get_exec_stream())?;
    ctx.get_d2h_stream().wait(exec_finished)?;

    let mut result = vec![0u8; result_buf_len];

    copy_and_free(&mut result[..], d_result, ctx.get_d2h_stream())?;
    
    Ok(result)
}
