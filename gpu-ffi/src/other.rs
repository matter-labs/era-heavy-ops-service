use super::*;

pub fn get_powers_of_alpha(
    ctx: &GpuContext,
    d_scalars: *mut c_void,
    log_degree: usize,
    offset: usize,
    count: usize,
    inverse: bool,
) -> Result<(), GpuError> {
    unsafe {
        if ff_get_powers_of_w(
            d_scalars,
            log_degree as u32,
            offset as u32,
            count as u32,
            inverse,
            false,
            ctx.get_exec_stream(),
        ) != 0
        {
            return Err(GpuError::SchedulingErr);
        }
    }

    Ok(())
}

pub fn get_powers_of_coset_gen(
    ctx: &GpuContext,
    d_scalars: *mut c_void,
    offset: usize,
    count: usize,
    inverse: bool,
) -> Result<(), GpuError> {
    unsafe {
        if ff_get_powers_of_g(
            d_scalars,
            offset as u32,
            count as u32,
            inverse,
            ctx.get_exec_stream(),
        ) != 0
        {
            return Err(GpuError::SchedulingErr);
        }
    }

    Ok(())
}

pub fn shift(
    ctx: &GpuContext,
    d_scalars: *mut c_void,
    log_degree: usize,
    shift: usize,
    offset: usize,
    count: usize,
    inverse: bool,
) -> Result<(), GpuError> {
    unsafe {
        if ff_shift(
            d_scalars,
            d_scalars,
            log_degree as u32,
            shift as u32,
            offset as u32,
            count as u32,
            inverse,
            ctx.get_exec_stream(),
        ) != 0
        {
            return Err(GpuError::SchedulingErr);
        }
    }

    Ok(())
}
