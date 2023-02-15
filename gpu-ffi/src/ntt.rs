use super::*;

pub fn ntt(
    ctx: &GpuContext,
    scalars: &mut [u8],
    bits_reversed: bool,
    inverse: bool,
) -> Result<(), GpuError> {
    let len = scalars.len();
    let d_scalars = alloc_and_copy(ctx, scalars, ctx.get_h2d_stream())?;
    ctx.wait_h2d()?;

    raw_ntt(ctx, d_scalars, len, bits_reversed, inverse,)
}

pub fn raw_ntt(
    ctx: &GpuContext,
    d_scalars: *mut c_void,
    len: usize,
    bits_reversed: bool,
    inverse: bool,
) -> Result<(), GpuError> {
    let log_scalars_count = log_2(len/FIELD_ELEMENT_LEN);
    let cfg = ntt_configuration::new(
        ctx,
        d_scalars,
        d_scalars,
        log_scalars_count,
        bits_reversed,
        inverse,
    );
    if unsafe { ntt_execute_async(cfg) } != 0 {
        return Err(GpuError::SchedulingErr);
    };

    Ok(())
}

pub fn ifft_then_msm(
    ctx: &GpuContext,
    scalars: &[u8],
    bits_reversed: bool,
) -> Result<Vec<u8>, GpuError> {
    let len = scalars.len();

    let d_scalars = alloc_and_copy(ctx, scalars, ctx.get_h2d_stream())?;

    let h2d_finished = bc_event::new()?;
    h2d_finished.record(ctx.get_h2d_stream())?;
    ctx.get_exec_stream().wait(h2d_finished)?;

    raw_ntt(ctx, d_scalars, len, bits_reversed, true,)?;
    let result = raw_msm(ctx, d_scalars, len)?;

    let exec_finished = bc_event::new()?;
    exec_finished.record(ctx.get_exec_stream())?;
    ctx.get_d2h_stream().wait(exec_finished)?;

    ctx.get_d2h_stream().sync()?;

    Ok(result)
}

pub fn coset_fft(
    ctx: &GpuContext,
    scalars: &mut [u8],
    lde_factor: u32,
    coset_idx: usize,
) -> Result<(), GpuError> {
    let len = scalars.len();
    let d_scalars = alloc_and_copy(ctx, scalars, ctx.get_h2d_stream())?;
    ctx.wait_h2d()?;

    raw_coset_ntt(ctx, d_scalars, len, lde_factor, coset_idx, false)?;

    copy_and_free(scalars, d_scalars, ctx.get_d2h_stream())?;

    Ok(())
}

pub fn raw_coset_ntt(
    ctx: &GpuContext,
    d_scalars: *mut c_void,
    len: usize,
    lde_factor: u32,
    coset_idx: usize,
    inverse: bool,
) -> Result<(), GpuError> {
    let log_scalars_count = log_2(len/FIELD_ELEMENT_LEN);
    let mut cfg = ntt_configuration::new_for_lde(
        ctx,
        d_scalars,
        d_scalars,
        log_scalars_count,
        coset_idx,
        lde_factor,
    );
    cfg.inverse = inverse;
    if unsafe { ntt_execute_async(cfg) } != 0 {
        return Err(GpuError::SchedulingErr);
    }

    Ok(())
}

pub fn coset_ifft(
    ctx: &GpuContext,
    scalars: &mut [u8],
    lde_factor: u32,
    coset_idx: usize,
) -> Result<(), GpuError> {
    let len = scalars.len();
    let d_scalars = alloc_and_copy(ctx, scalars, ctx.get_h2d_stream())?;
    ctx.wait_h2d()?;

    raw_coset_ntt(ctx, d_scalars, len, lde_factor, coset_idx, true)?;

    copy_and_free(scalars, d_scalars, ctx.get_d2h_stream())?;

    Ok(())
}

pub fn lde(
    ctx: &GpuContext,
    scalars: &[u8],
    log_scalars_count: u32,
    lde_factor: u32,
) -> Result<Vec<u8>, GpuError> {
    let len = scalars.len();

    let d_scalars = alloc_and_copy(ctx, scalars, ctx.get_h2d_stream())?;
    let h2d_finished = bc_event::new()?;
    h2d_finished.record(ctx.get_h2d_stream())?;
    ctx.get_exec_stream().wait(h2d_finished)?;

    let new_size = lde_factor as usize * len;
    let mut result = Vec::with_capacity(new_size);
    unsafe {
        result.set_len(new_size);
    }

    for (coset_idx, h_result) in result.chunks_mut(len).enumerate() {
        let mut d_result = std::ptr::null_mut();
        malloc_from_pool_async(
            addr_of_mut!(d_result),
            len,
            ctx.get_mem_pool(),
            ctx.get_h2d_stream(),
        )?;

        let cfg = ntt_configuration::new_for_lde(
            ctx,
            d_scalars,
            d_result,
            log_scalars_count,
            coset_idx as usize,
            lde_factor,
        );

        if unsafe { ntt_execute_async(cfg) } != 0 {
            return Err(GpuError::SchedulingErr);
        }

        let exec_finished = bc_event::new()?;
        exec_finished.record(ctx.get_exec_stream())?;
        ctx.get_d2h_stream().wait(exec_finished)?;

        copy_and_free(h_result, d_result, ctx.get_d2h_stream())?;
    }

    Ok(result)
}

pub fn fft(
    ctx: &GpuContext,
    scalars: &mut [u8],
    bits_reversed: bool,    
) -> Result<(), GpuError> {
    ntt(ctx, scalars, bits_reversed, false)
}

pub fn multi_fft(
    ctx: &GpuContext,
    multi_scalars: &mut [&mut [u8]],
    bits_reversed: bool,
) -> Result<(), GpuError> {
    multi_ntt(ctx, multi_scalars, bits_reversed, false)
}

pub fn multi_ifft(
    ctx: &GpuContext,
    multi_scalars: &mut [&mut [u8]],
    bits_reversed: bool,
) -> Result<(), GpuError> {
    multi_ntt(ctx, multi_scalars, bits_reversed, true)
}

pub fn multi_ntt(
    ctx: &GpuContext,
    multi_scalars: &mut [&mut [u8]],
    bits_reversed: bool,
    inverse: bool,
) -> Result<(), GpuError> {
    if multi_scalars.len() == 1 {
        fft(ctx, multi_scalars[0], bits_reversed)?;
        return Ok(());
    }
    let len = multi_scalars[0].len();

    for scalars in multi_scalars {
        assert_eq!(len, scalars.len());
        let d_scalars = alloc_and_copy(ctx, scalars, ctx.get_h2d_stream())?;
        let h2d_finished = bc_event::new()?;
        h2d_finished.record(ctx.get_h2d_stream())?;
        ctx.get_exec_stream().wait(h2d_finished)?;

        raw_ntt(ctx, d_scalars, len,  bits_reversed, inverse)?;

        let exec_finished = bc_event::new()?;
        exec_finished.record(ctx.get_exec_stream())?;
        ctx.get_d2h_stream().wait(exec_finished)?;

        copy_and_free(scalars, d_scalars, ctx.get_d2h_stream())?;
    }

    ctx.get_d2h_stream().sync()?;
    Ok(())
}

pub fn ifft(
    ctx: &GpuContext,
    scalars: &mut [u8],
    bits_reversed: bool,
) -> Result<(), GpuError> {
    ntt(ctx, scalars, bits_reversed, true)
}
