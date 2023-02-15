use super::*;
use bellman::Field;
use crate::cuda_bindings::GpuError;
use core::ops::Range;

pub enum Operation {
    AddConst,
    SubConst,
    MulConst,
    Add,
    Sub,
    Mul,
    AddScaled,
    SubScaled,
    BatchInv,
    GrandProd,
    SetValue
}

impl DeviceBuf<Fr> {
    pub fn async_exec_op(
        &mut self,
        ctx: &mut GpuContext,
        other: Option<&mut DeviceBuf<Fr>>,
        constant: Option<Fr>,
        range: Range<usize>,
        op: Operation,
    ) -> GpuResult<()> {
        assert!(ctx.ff, "ff is not set up on GpuContext with id {}", ctx.device_id());
        set_device(ctx.device_id())?;

        let length = range.len();

        ctx.exec_stream.wait(self.write_event())?;
        ctx.exec_stream.wait(self.read_event())?;

        if let Some(other) = &other {
            ctx.exec_stream.wait(other.write_event())?;
        }

        let result = unsafe {
            match op {
                Operation::AddConst => {
                    assert!(other.is_none(), "other DeviceBuf should be None in AddConst operation");
                    let constant = constant.expect("constant should be Some in AddConst operation");

                    ff_a_plus_x(
                        &constant as *const Fr as *const c_void,
                        self.as_ptr(range.clone()) as *const c_void,
                        self.as_mut_ptr(range) as *mut c_void,
                        length as u32,
                        ctx.exec_stream.inner,
                    )
                },
                Operation::SubConst => {
                    assert!(other.is_none(), "other DeviceBuf should be None in SubConst operation");
                    let constant = constant.expect("constant should be Some in SubConst operation");

                    let mut constant = constant;
                    constant.negate();

                    ff_a_plus_x(
                        &constant as *const Fr as *const c_void,
                        self.as_ptr(range.clone()) as *const c_void,
                        self.as_mut_ptr(range) as *mut c_void,
                        length as u32,
                        ctx.exec_stream.inner,
                    )
                },
                Operation::MulConst => {
                    assert!(other.is_none(), "other DeviceBuf should be None in MulConst operation");
                    let constant = constant.expect("constant should be Some in MulConst operation");

                    ff_ax(
                        &constant as *const Fr as *const c_void,
                        self.as_ptr(range.clone()) as *const c_void,
                        self.as_mut_ptr(range) as *mut c_void,
                        length as u32,
                        ctx.exec_stream.inner,
                    )
                },
                Operation::Add => {
                    assert!(constant.is_none(), "constant should be None in Add operation");
                    let other = other.as_ref().expect("other DeviceBuf should be Some in Add operation");

                    ff_x_plus_y(
                        self.as_ptr(range.clone()) as *const c_void, 
                        other.as_ptr(range.clone()) as *const c_void, 
                        self.as_mut_ptr(range) as *mut c_void,
                        length as u32, 
                        ctx.exec_stream.inner,
                    )
                },
                Operation::Sub => {
                    assert!(constant.is_none(), "constant should be None in Sub operation");
                    let other = other.as_ref().expect("other DeviceBuf should be Some in Sub operation");

                    ff_x_minus_y(
                        self.as_ptr(range.clone()) as *const c_void, 
                        other.as_ptr(range.clone()) as *const c_void, 
                        self.as_mut_ptr(range) as *mut c_void,
                        length as u32, 
                        ctx.exec_stream.inner,
                    )
                },
                Operation::Mul => {
                    assert!(constant.is_none(), "constant should be None in Mul operation");
                    let other = other.as_ref().expect("other DeviceBuf should be Some in Mul operation");

                    ff_x_mul_y(
                        self.as_ptr(range.clone()) as *const c_void, 
                        other.as_ptr(range.clone()) as *const c_void, 
                        self.as_mut_ptr(range) as *mut c_void,
                        length as u32, 
                        ctx.exec_stream.inner,
                    )
                },
                Operation::AddScaled => {
                    let constant = constant.expect("constant should be Some in AddScaled operation");
                    let other = other.as_ref().expect("other DeviceBuf should be Some in AddScaled operation");

                    ff_ax_plus_y(
                        &constant as *const Fr as *const c_void,
                        other.as_ptr(range.clone()) as *const c_void, 
                        self.as_ptr(range.clone()) as *const c_void, 
                        self.as_mut_ptr(range) as *mut c_void,
                        length as u32, 
                        ctx.exec_stream.inner,
                    )
                },
                Operation::SubScaled => {
                    let constant = constant.expect("constant should be Some in SubScaled operation");
                    let other = other.as_ref().expect("other DeviceBuf should be Some in SubScaled operation");

                    ff_x_minus_ay(
                        &constant as *const Fr as *const c_void,
                        self.as_ptr(range.clone()) as *const c_void, 
                        other.as_ptr(range.clone()) as *const c_void, 
                        self.as_mut_ptr(range) as *mut c_void,
                        length as u32, 
                        ctx.exec_stream.inner,
                    )
                },
                Operation::BatchInv => {
                    assert!(other.is_none(), "other DeviceBuf should be None in BatchInv operation");
                    assert!(constant.is_none(), "constant should be None in BatchInv operation");

                    let mem_pool = ctx.mem_pool.expect("mem pool should be allocated in BatchInv operation");

                    let cfg = ff_inverse_configuration {
                        mem_pool,
                        stream: ctx.exec_stream.inner,
                        inputs: self.as_mut_ptr(range.clone()) as *mut c_void,
                        outputs: self.as_mut_ptr(range) as *mut c_void,
                        count: length as u32,
                    };
                
                    ff_inverse(cfg)
                }
                Operation::GrandProd => {
                    assert!(other.is_none(), "other DeviceBuf should be None in GrandProd operation");
                    assert!(constant.is_none(), "constant should be None in GrandProd operation");

                    let mem_pool = ctx.mem_pool.expect("mem pool should be allocated in GrandProd operation");

                    let cfg = ff_grand_product_configuration {
                        mem_pool,
                        stream: ctx.exec_stream.inner,
                        inputs: self.as_mut_ptr(range.clone()) as *mut c_void,
                        outputs: self.as_mut_ptr(range) as *mut c_void,
                        count: length as u32,
                    };

                    ff_grand_product(cfg)
                }
                Operation::SetValue => {
                    assert!(other.is_none(), "other DeviceBuf should be None in SetValue operation");
                    let constant = constant.expect("constant should be Some in SetValue operation");

                    ff_set_value(
                        self.as_mut_ptr(range) as *mut c_void,
                        &constant as *const Fr as *const c_void,
                        length as u32, 
                        ctx.exec_stream.inner
                    )
                }
                _ => unreachable!()
            }
        };

        if result != 0 {
            return Err(GpuError::ArithmeticErr(result));
        }

        self.write_event.record(&ctx.exec_stream)?;
        if let Some(other) = other {
            other.read_event.record(&ctx.exec_stream)?;
        }

        Ok(())
    }

    // output[i] = input[i] * w^((i + offset)*shift)
    // w^(2^log_degree) = 1
    pub fn distribute_omega_powers(
        &mut self,
        ctx: &mut GpuContext,
        log_degree: usize,
        offset: usize,
        shift: usize,
        inverse: bool,
    ) -> GpuResult<()> {
        assert!(ctx.ff, "ff is not set up on GpuContext with id {}", ctx.device_id());
        set_device(ctx.device_id())?;

        let length = self.len();
        ctx.exec_stream.wait(self.write_event())?;
        ctx.exec_stream.wait(self.read_event())?;

        unsafe {
            let result = ff_omega_shift(
                self.as_ptr(0..length) as *const c_void,
                self.as_mut_ptr(0..length) as *mut c_void,
                log_degree as u32,
                shift as u32,
                offset as u32,
                length as u32,
                inverse,
                ctx.exec_stream.inner
            );
            if result != 0 {
                return Err(GpuError::DistributeOmegasErr(result));
            }
        }

        self.write_event.record(&ctx.exec_stream)?;

        Ok(())
    }
}