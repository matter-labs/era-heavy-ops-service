use super::*;

enum BinOpAssign {
    Add,
    AddScaled(*const c_void),
    Sub,
    SubScaled(*const c_void),
    Mul,
}

fn binop_constant(
    ctx: &GpuContext,
    this: &mut [u8],
    constant: &[u8],
    op: BinOpAssign,
) -> Result<(), GpuError> {
    assert_eq!(constant.len(), FIELD_ELEMENT_LEN);
    let len = this.len();
    let d_this = alloc_and_copy(ctx, this, ctx.get_h2d_stream())?;
    let h2d_finished = bc_event::new()?;
    h2d_finished.record(ctx.get_h2d_stream())?;
    ctx.get_exec_stream().wait(h2d_finished)?;

    let h_constant = constant.as_ptr() as *const c_void;

    raw_binop_constant(ctx, d_this, len, h_constant, op)?;

    copy_and_free(this, d_this, ctx.get_d2h_stream())?;

    Ok(())
}

fn raw_binop_constant(
    ctx: &GpuContext,
    d_this: *mut c_void,
    len: usize,
    h_constant: *const c_void,
    op: BinOpAssign,
) -> Result<(), GpuError> {
    let count = (len / FIELD_ELEMENT_LEN) as u32;

    match op {
        BinOpAssign::Add => {
            unsafe { ff_a_plus_x(h_constant, d_this, d_this, count, ctx.get_exec_stream()) };
        }
        BinOpAssign::Mul => {
            unsafe { ff_ax(h_constant, d_this, d_this, count, ctx.get_exec_stream()) };
        }
        _ => unimplemented!(),
    };

    Ok(())
}

fn binop_assign(
    ctx: &GpuContext,
    this: &mut [u8],
    other: &[u8],
    op: BinOpAssign,
) -> Result<(), GpuError> {
    let len = this.len();

    let d_this = alloc_and_copy(ctx, this, ctx.get_h2d_stream())?;
    let d_other = alloc_and_copy(ctx, other, ctx.get_h2d_stream())?;

    let h2d_finished = bc_event::new()?;
    h2d_finished.record(ctx.get_h2d_stream())?;
    ctx.get_exec_stream().wait(h2d_finished)?;

    raw_binop_assign(ctx, d_this, d_other, len, op)?;

    copy_and_free(this, d_this, ctx.get_d2h_stream())?;
    free_async(d_this, ctx.get_d2h_stream())?;
    free_async(d_other, ctx.get_d2h_stream())?;

    Ok(())
}

fn raw_binop_assign(
    ctx: &GpuContext,
    d_this: *mut c_void,
    d_other: *const c_void,
    len: usize,
    op: BinOpAssign,
) -> Result<(), GpuError> {
    let count = (len / FIELD_ELEMENT_LEN) as u32;
    let ret = unsafe {
        match op {
            BinOpAssign::Add => ff_x_plus_y(d_this, d_other, d_this, count, ctx.get_exec_stream()),
            BinOpAssign::AddScaled(scaler) => {
                let ret = ff_ax_plus_y(
                    scaler as *const c_void,
                    d_other,
                    d_this,
                    d_this,
                    count,
                    ctx.get_exec_stream(),
                );
                ret
            }
            BinOpAssign::Sub => ff_x_minus_y(d_this, d_other, d_this, count, ctx.get_exec_stream()),
            BinOpAssign::SubScaled(scaler) => {
                let ret = ff_x_minus_ay(
                    scaler as *const c_void,
                    d_this,
                    d_other,
                    d_this,
                    count,
                    ctx.get_exec_stream(),
                );
                ret
            }
            BinOpAssign::Mul => ff_x_mul_y(d_this, d_other, d_this, count, ctx.get_exec_stream()),
        }
    };

    if ret != 0 {
        return Err(GpuError::SchedulingErr);
    }

    Ok(())
}

pub fn add_constant(
    ctx: &GpuContext,
    h_this: &mut [u8],
    h_constant: &[u8],
) -> Result<(), GpuError> {
    assert_eq!(h_constant.len(), FIELD_ELEMENT_LEN);
    binop_constant(ctx, h_this, h_constant, BinOpAssign::Add)
}

pub fn raw_add_constant(
    ctx: &GpuContext,
    d_this: *mut c_void,
    len: usize,
    h_constant: *const c_void,
) -> Result<(), GpuError> {
    raw_binop_constant(ctx, d_this, len, h_constant, BinOpAssign::Add)
}

pub fn sub_constant(
    ctx: &GpuContext,
    h_this: &mut [u8],
    h_constant: &[u8],
) -> Result<(), GpuError> {
    assert_eq!(h_constant.len(), FIELD_ELEMENT_LEN);
    binop_constant(ctx, h_this, h_constant, BinOpAssign::Sub)
}

pub fn raw_sub_constant(
    ctx: &GpuContext,
    d_this: *mut c_void,
    len: usize,
    h_constant: *const c_void,
) -> Result<(), GpuError> {
    raw_binop_constant(ctx, d_this, len, h_constant, BinOpAssign::Sub)
}

pub fn mul_constant(
    ctx: &GpuContext,
    h_this: &mut [u8],
    h_constant: &[u8],
) -> Result<(), GpuError> {
    assert_eq!(h_constant.len(), FIELD_ELEMENT_LEN);
    binop_constant(ctx, h_this, h_constant, BinOpAssign::Mul)
}

pub fn raw_mul_constant(
    ctx: &GpuContext,
    d_this: *mut c_void,
    len: usize,
    h_constant: *const c_void,
) -> Result<(), GpuError> {
    raw_binop_constant(ctx, d_this, len, h_constant, BinOpAssign::Mul)
}

pub fn add_assign(ctx: &GpuContext, h_this: &mut [u8], h_other: &[u8]) -> Result<(), GpuError> {
    binop_assign(ctx, h_this, h_other, BinOpAssign::Add)
}

pub fn raw_add_assign(
    ctx: &GpuContext,
    d_this: *mut c_void,
    d_other: *const c_void,
    len: usize,
) -> Result<(), GpuError> {
    raw_binop_assign(ctx, d_this, d_other, len, BinOpAssign::Add)
}

pub fn sub_assign(ctx: &GpuContext, h_this: &mut [u8], h_other: &[u8]) -> Result<(), GpuError> {
    binop_assign(ctx, h_this, h_other, BinOpAssign::Sub)
}

pub fn raw_sub_assign(
    ctx: &GpuContext,
    d_this: *mut c_void,
    d_other: *const c_void,
    len: usize,
) -> Result<(), GpuError> {
    raw_binop_assign(ctx, d_this, d_other, len, BinOpAssign::Sub)
}

pub fn mul_assign(ctx: &GpuContext, this: &mut [u8], other: &[u8]) -> Result<(), GpuError> {
    binop_assign(ctx, this, other, BinOpAssign::Mul)
}

pub fn raw_mul_assign(
    ctx: &GpuContext,
    d_this: *mut c_void,
    d_other: *const c_void,
    len: usize,
) -> Result<(), GpuError> {
    raw_binop_assign(ctx, d_this, d_other, len, BinOpAssign::Mul)
}

pub fn add_assign_scaled(
    ctx: &GpuContext,
    this: &mut [u8],
    other: &[u8],
    scaler: &[u8],
) -> Result<(), GpuError> {
    assert_eq!(scaler.len(), FIELD_ELEMENT_LEN);
    binop_assign(
        ctx,
        this,
        other,
        BinOpAssign::AddScaled(scaler.as_ptr() as *const c_void),
    )
}

pub fn raw_add_assign_scaled(
    ctx: &GpuContext,
    d_this: *mut c_void,
    d_other: *const c_void,
    len: usize,
    h_constant: *const c_void,
) -> Result<(), GpuError> {
    raw_binop_assign(
        ctx,
        d_this,
        d_other,
        len,
        BinOpAssign::AddScaled(h_constant),
    )
}

pub fn sub_assign_scaled(
    ctx: &GpuContext,
    this: &mut [u8],
    other: &[u8],
    scaler: &[u8],
) -> Result<(), GpuError> {
    assert_eq!(scaler.len(), FIELD_ELEMENT_LEN);
    binop_assign(
        ctx,
        this,
        other,
        BinOpAssign::SubScaled(scaler.as_ptr() as *const c_void),
    )
}

pub fn raw_sub_assign_scaled(
    ctx: &GpuContext,
    d_this: *mut c_void,
    d_other: *const c_void,
    len: usize,
    h_constant: *const c_void,
) -> Result<(), GpuError> {
    raw_binop_assign(
        ctx,
        d_this,
        d_other,
        len,
        BinOpAssign::SubScaled(h_constant),
    )
}

pub fn shifted_grand_product(ctx: &GpuContext, h_this: &[u8]) -> Result<Vec<u8>, GpuError> {
    let len = h_this.len();

    let d_this = alloc_and_copy(ctx, h_this, ctx.get_h2d_stream())?;

    let h2d_finished = bc_event::new()?;
    h2d_finished.record(ctx.get_h2d_stream())?;
    ctx.get_exec_stream().wait(h2d_finished)?;

    raw_shifted_grand_product(ctx, d_this, len)?;

    let count = (len / FIELD_ELEMENT_LEN) as u32;

    let new_len = FIELD_ELEMENT_LEN * (count as usize + 1).next_power_of_two();
    let mut result = vec![0u8; new_len];
    copy_and_free(
        &mut result[FIELD_ELEMENT_LEN..len + FIELD_ELEMENT_LEN],
        d_this,
        ctx.get_d2h_stream(),
    )?;

    Ok(result)
}

pub fn raw_shifted_grand_product(
    ctx: &GpuContext,
    d_this: *mut c_void,
    len: usize,
) -> Result<(), GpuError> {
    let count = (len / FIELD_ELEMENT_LEN) as u32;
    let cfg = ff_grand_product_configuration {
        mem_pool: ctx.get_mem_pool(),
        stream: ctx.get_exec_stream(),
        inputs: d_this,
        outputs: d_this,
        count,
    };
    if unsafe { ff_grand_product(cfg) } != 0 {
        return Err(GpuError::FinishProcessingErr);
    }
    let exec_finished = bc_event::new()?;
    exec_finished.record(ctx.get_exec_stream())?;
    ctx.get_d2h_stream().wait(exec_finished)?;

    Ok(())
}

pub fn batch_inversion(ctx: &GpuContext, h_this: &mut [u8]) -> Result<(), GpuError> {
    let len = h_this.len();
    let d_this = alloc_and_copy(ctx, h_this, ctx.get_h2d_stream())?;

    raw_batch_inversion(ctx, d_this, len)?;

    copy_and_free(h_this, d_this, ctx.get_d2h_stream())?;

    Ok(())
}

pub fn raw_batch_inversion(
    ctx: &GpuContext,
    d_this: *mut c_void,
    len: usize,
) -> Result<(), GpuError> {
    let cfg = ff_inverse_configuration {
        mem_pool: ctx.get_mem_pool(),
        stream: ctx.get_exec_stream(),
        inputs: d_this,
        outputs: d_this,
        count: (len / FIELD_ELEMENT_LEN) as u32,
    };

    if unsafe { ff_inverse(cfg) } != 0 {
        return Err(GpuError::SchedulingErr);
    }

    let exec_finished = bc_event::new()?;
    exec_finished.record(ctx.get_exec_stream())?;
    ctx.get_d2h_stream().wait(exec_finished)?;

    Ok(())
}

pub fn evaluate(
    ctx: &GpuContext,
    h_this: &[u8],
    h_point: &[u8],
) -> Result<[u8; FIELD_ELEMENT_LEN], GpuError> {
    assert_eq!(h_point.len(), FIELD_ELEMENT_LEN);
    let len = h_this.len();
    let d_this = alloc_and_copy(ctx, h_this, ctx.get_h2d_stream())?;
    let h_point = h_point.as_ptr() as *const c_void;
    let mut result = [0u8; 32];
    let d_result = alloc_and_copy(ctx, result.as_ref(), ctx.get_h2d_stream())?;
    raw_evaluate(ctx, d_this, len, h_point, d_result)?;

    copy_and_free(result.as_mut(), d_result, ctx.get_d2h_stream())?;
    free_async(d_this, ctx.get_d2h_stream())?;

    Ok(result)
}

pub fn raw_evaluate(
    ctx: &GpuContext,
    d_this: *mut c_void,
    len: usize,
    h_point: *const c_void,
    d_result: *mut c_void,
) -> Result<(), GpuError> {
    let count = (len / FIELD_ELEMENT_LEN) as u32;
    let cfg = ff_poly_evaluate_configuration {
        mem_pool: ctx.get_mem_pool(),
        stream: ctx.get_exec_stream(),
        values: d_this,
        point: h_point as *mut c_void,
        result: d_result,
        count,
    };

    if unsafe { ff_poly_evaluate(cfg) } != 0 {
        return Err(GpuError::SchedulingErr);
    }


    Ok(())
}

pub fn distribute_powers(ctx: &GpuContext, this: &mut [u8], base: &[u8]) -> Result<(), GpuError> {
    assert_eq!(base.len(), FIELD_ELEMENT_LEN);
    let len = this.len();

    let d_this = alloc_and_copy(ctx, this, ctx.get_h2d_stream())?;
    let h_base = base.as_ptr() as *const c_void;

    raw_distribute_powers(ctx, d_this, len, h_base)?;

    copy_and_free(this, d_this, ctx.get_d2h_stream())?;

    Ok(())
}

// This is temp function that will be replaced with gpu based one later
pub fn raw_distribute_powers(
    ctx: &GpuContext,
    d_this: *mut c_void,
    len: usize,
    h_base: *const c_void,
) -> Result<(), GpuError> {
    let mut d_other = std::ptr::null_mut();
    malloc_from_pool_async(
        addr_of_mut!(d_other),
        len,
        ctx.get_mem_pool(),
        ctx.get_h2d_stream(),
    )?;
    let count = (len / FIELD_ELEMENT_LEN) as u32;
    unsafe {
        if ff_set_value(d_other, h_base, count, ctx.get_exec_stream()) != 0 {
            return Err(GpuError::SchedulingErr);
        }
        if ff_set_value_one(d_other, 1, ctx.get_exec_stream()) != 0 {
            return Err(GpuError::SchedulingErr);
        }
    }

    let h2d_finished = bc_event::new()?;
    h2d_finished.record(ctx.get_h2d_stream())?;
    ctx.get_exec_stream().wait(h2d_finished)?;

    let cfg = ff_grand_product_configuration {
        mem_pool: ctx.get_mem_pool(),
        stream: ctx.get_exec_stream(),
        inputs: d_other,
        outputs: d_other,
        count,
    };

    unsafe {
        if ff_grand_product(cfg) != 0 {
            return Err(GpuError::FinishProcessingErr);
        }

        if ff_x_mul_y(d_this, d_other, d_this, count, ctx.get_exec_stream()) != 0 {
            return Err(GpuError::SchedulingErr);
        }
    }
    let exec_finished = bc_event::new()?;
    exec_finished.record(ctx.get_exec_stream())?;
    ctx.get_d2h_stream().wait(exec_finished)?;

    free_async(d_other, ctx.get_d2h_stream())?;
    Ok(())
}
