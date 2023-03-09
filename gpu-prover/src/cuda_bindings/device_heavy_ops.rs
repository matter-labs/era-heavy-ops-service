use super::*;
use crate::cuda_bindings::GpuError;

impl DeviceBuf<Fr> {
    pub fn msm(&mut self, ctx: &mut GpuContext) -> GpuResult<DeviceBuf<G1>> {
        let length = self.len();
        assert_eq!(
            length, ctx.bases_len,
            "length of polynomial and bases should be equal"
        );
        assert!(
            ctx.mem_pool.is_some(),
            "mem_pool should be set up on GpuContext"
        );
        set_device(ctx.device_id())?;

        let mut result = DeviceBuf::<G1>::async_alloc_in_exec(ctx, 254)?;
        ctx.exec_stream.wait(self.write_event())?;

        let null_ptr = std::ptr::null_mut() as *mut c_void;
        let null_event = bc_event { handle: null_ptr };

        let cfg = msm_configuration {
            mem_pool: ctx.mem_pool.unwrap(),
            stream: ctx.exec_stream.inner,
            bases: ctx.bases.unwrap() as *mut c_void,
            scalars: self.as_mut_ptr(0..length) as *mut c_void,
            results: result.as_mut_ptr(0..254) as *mut c_void,
            log_scalars_count: log_2(length),
            h2d_copy_finished: null_event,
            h2d_copy_finished_callback: None,
            h2d_copy_finished_callback_data: null_ptr,
            d2h_copy_finished: null_event,
            d2h_copy_finished_callback: None,
            d2h_copy_finished_callback_data: null_ptr,
        };

        unsafe {
            let result = msm_execute_async(cfg);
            if result != 0 {
                return Err(GpuError::MSMErr(result));
            };
        }

        self.read_event.record(ctx.exec_stream())?;
        result.write_event.record(ctx.exec_stream())?;

        Ok(result)
    }

    pub fn evaluate_at(&mut self, ctx: &mut GpuContext, mut point: Fr) -> GpuResult<DeviceBuf<Fr>> {
        assert!(
            ctx.mem_pool.is_some(),
            "mem_pool should be set up on GpuContext"
        );
        assert!(ctx.ff, "ff should be set up on GpuContext");
        set_device(ctx.device_id())?;

        let length = self.len();
        let mut result = DeviceBuf::<Fr>::async_alloc_in_exec(ctx, 1)?;
        ctx.exec_stream.wait(self.write_event())?;

        let cfg = ff_poly_evaluate_configuration {
            mem_pool: ctx.mem_pool.unwrap(),
            stream: ctx.exec_stream.inner,
            values: self.as_mut_ptr(0..length) as *mut c_void,
            point: &mut point as *mut Fr as *mut c_void,
            result: result.as_mut_ptr(0..1) as *mut c_void,
            count: length as u32,
        };

        unsafe {
            let result = ff_poly_evaluate(cfg);
            if result != 0 {
                return Err(GpuError::EvaluationErr(result));
            }
        }

        self.read_event.record(ctx.exec_stream())?;
        result.write_event.record(ctx.exec_stream())?;

        Ok(result)
    }

    pub fn ntt(
        buffers: &mut Vec<&mut Self>,
        ctx: &mut GpuContext,
        bits_reversed: bool,
        inverse: bool,
        coset_index: Option<usize>,
        lde_factor: Option<u32>,
        final_bitreverse: bool,
    ) -> GpuResult<()> {
        set_device(ctx.device_id())?;

        let mut length = buffers[0].len();
        for buffer in buffers.iter() {
            ctx.exec_stream.wait(buffer.write_event())?;
            ctx.exec_stream.wait(buffer.read_event())?;
        }

        for i in 0..(buffers.len() - 1) {
            assert_eq!(
                buffers[i].as_ptr(length..length),
                buffers[i + 1].as_ptr(0..0),
                "Buffers should be allocated one by one"
            );

            assert_eq!(
                length,
                buffers[i + 1].len(),
                "Buffers' lengths are not equal"
            );
        }

        let d_scalars = buffers[0].as_ptr(0..length);
        length = length * buffers.len();

        // use bellman::PrimeField;
        // let fr_size = Fr::zero().into_raw_repr().as_ref().len() * 8;
        let log_scalars_count = log_2(length);

        let mut log_extension_degree = 0;
        let mut coset_idx = 0;
        if let (Some(coset_idxt), Some(lde_factor)) = (coset_index, lde_factor) {
            log_extension_degree = log_2(lde_factor as usize);
            coset_idx = bitreverse(coset_idxt, log_extension_degree as usize);
        }

        // this.coset_index = coset_index as u32;
        // this.log_extension_degree = log_extension_degree;

        let cfg = ntt_configuration {
            mem_pool: ctx.mem_pool.expect("mem_pool should be allocated"),
            stream: ctx.exec_stream().inner,
            inputs: d_scalars as *mut c_void,
            outputs: d_scalars as *mut c_void,
            log_values_count: log_scalars_count,
            bit_reversed_inputs: bits_reversed,
            inverse,
            h2d_copy_finished: bc_event {
                handle: std::ptr::null_mut() as *mut c_void,
            },
            h2d_copy_finished_callback: None,
            h2d_copy_finished_callback_data: std::ptr::null_mut() as *mut c_void,
            d2h_copy_finished: bc_event {
                handle: std::ptr::null_mut() as *mut c_void,
            },
            d2h_copy_finished_callback: None,
            d2h_copy_finished_callback_data: std::ptr::null_mut() as *mut c_void,
            can_overwrite_inputs: false,
            coset_index: coset_idx as u32,
            log_extension_degree: log_extension_degree,
        };

        unsafe {
            let result = ntt_execute_async(cfg);
            if result != 0 {
                return Err(GpuError::NTTErr(result));
            }
        }

        // if final_bitreverse {
        //     if unsafe { ff_bit_reverse(
        //         d_scalars as *const c_void,
        //         d_scalars as *mut c_void,
        //         log_scalars_count as u32,
        //         ctx.exec_stream.inner,
        //     ) } != 0 {
        //         return Err(GpuError::SchedulingErr);
        //     }
        // }

        for buffer in buffers.iter_mut() {
            buffer.read_event.record(ctx.exec_stream())?;
            buffer.write_event.record(ctx.exec_stream())?;
        }

        if final_bitreverse {
            DeviceBuf::bitreverse(buffers, ctx)?;
        }

        Ok(())
    }

    pub fn multigpu_ntt(
        buffers: &mut Vec<Self>,
        ctx: &mut Vec<GpuContext>,
        bits_reversed: bool,
        inverse: bool,
        coset_index: Option<usize>,
        lde_factor: Option<u32>,
        final_bitreverse: bool,
    ) -> GpuResult<()> {
        assert_eq!(buffers.len(), ctx.len());
        let mut length = buffers[0].len();
        for i in 0..(buffers.len() - 1) {
            assert_eq!(
                length,
                buffers[i + 1].len(),
                "Buffers' lengths are not equal"
            );
        }

        length = length * buffers.len();
        let log_scalars_count = log_2(length);

        let mut log_extension_degree = 0;
        let mut coset_idx = 0;
        if let (Some(coset_idxt), Some(lde_factor)) = (coset_index, lde_factor) {
            log_extension_degree = log_2(lde_factor as usize);
            coset_idx = bitreverse(coset_idxt, log_extension_degree as usize);
        }

        let mut cfgs = vec![];
        let mut dev_ids = vec![];

        for (buffer_idx, buffer) in buffers.iter().enumerate() {
            let device_id = ctx[buffer_idx].device_id();
            set_device(device_id)?;
            ctx[buffer_idx].exec_stream.wait(buffer.write_event())?;
            ctx[buffer_idx].exec_stream.wait(buffer.read_event())?;

            let d_scalars = buffer.as_ptr(0..buffer.len());

            let cfg = ntt_configuration {
                mem_pool: ctx[buffer_idx]
                    .mem_pool
                    .expect("mem_pool should be allocated"),
                stream: ctx[buffer_idx].exec_stream().inner,
                inputs: d_scalars as *mut c_void,
                outputs: d_scalars as *mut c_void,
                log_values_count: log_scalars_count,
                bit_reversed_inputs: bits_reversed,
                inverse,
                h2d_copy_finished: bc_event {
                    handle: std::ptr::null_mut() as *mut c_void,
                },
                h2d_copy_finished_callback: None,
                h2d_copy_finished_callback_data: std::ptr::null_mut() as *mut c_void,
                d2h_copy_finished: bc_event {
                    handle: std::ptr::null_mut() as *mut c_void,
                },
                d2h_copy_finished_callback: None,
                d2h_copy_finished_callback_data: std::ptr::null_mut() as *mut c_void,
                can_overwrite_inputs: false,
                coset_index: coset_idx as u32,
                log_extension_degree: log_extension_degree,
            };

            cfgs.push(cfg);
            dev_ids.push(device_id as i32);
        }

        // let cfgs = &mut cfgs as *mut Vec<ntt_configuration> as *mut ntt_configuration;
        // let dev_ids = &mut dev_ids as *mut Vec<i32> as *mut ::std::os::raw::c_int;

        let cfgs = &mut cfgs[0] as *mut ntt_configuration;
        let dev_ids = &mut dev_ids[0] as *mut ::std::os::raw::c_int;

        unsafe {
            let result = ntt_execute_async_multigpu(cfgs, dev_ids, log_2(buffers.len()));
            if result != 0 {
                return Err(GpuError::MultiGpuNTTErr(result));
            }
        }

        for (device_id, buffer) in buffers.iter_mut().enumerate() {
            buffer.read_event.record(ctx[device_id].exec_stream())?;
            buffer.write_event.record(ctx[device_id].exec_stream())?;
        }

        if final_bitreverse {
            DeviceBuf::multigpu_bitreverse(buffers, ctx)?;
        }

        Ok(())
    }

    pub fn multigpu_4n_ntt(
        buffers: &mut Vec<&mut Self>,
        ctx: &mut Vec<GpuContext>,
        bits_reversed: bool,
        inverse: bool,
        coset_index: Option<usize>,
        lde_factor: Option<u32>,
        final_bitreverse: bool,
    ) -> GpuResult<()> {
        let mut length = buffers[0].len();
        let num_gpus = buffers.len() / crate::LDE_FACTOR;
        for i in 0..(buffers.len() - 1) {
            assert_eq!(
                length,
                buffers[i + 1].len(),
                "Buffers' lengths are not equal"
            );
        }

        length = length * buffers.len();
        let log_scalars_count = log_2(length);

        let mut log_extension_degree = 0;
        let mut coset_idx = 0;
        if let (Some(coset_idxt), Some(lde_factor)) = (coset_index, lde_factor) {
            log_extension_degree = log_2(lde_factor as usize);
            coset_idx = bitreverse(coset_idxt, log_extension_degree as usize);
        }

        let mut cfgs = vec![];
        let mut dev_ids = vec![];

        for (ctx_id, buffer) in buffers.iter().enumerate() {
            let ctx_id = ctx_id % num_gpus;
            // dbg!(device_id);
            let device_id = ctx[ctx_id].device_id();
            set_device(device_id)?;
            ctx[ctx_id].exec_stream.wait(buffer.write_event())?;
            ctx[ctx_id].exec_stream.wait(buffer.read_event())?;

            let d_scalars = buffer.as_ptr(0..buffer.len());

            let cfg = ntt_configuration {
                mem_pool: ctx[ctx_id].mem_pool.expect("mem_pool should be allocated"),
                stream: ctx[ctx_id].exec_stream().inner,
                inputs: d_scalars as *mut c_void,
                outputs: d_scalars as *mut c_void,
                log_values_count: log_scalars_count,
                bit_reversed_inputs: bits_reversed,
                inverse,
                h2d_copy_finished: bc_event {
                    handle: std::ptr::null_mut() as *mut c_void,
                },
                h2d_copy_finished_callback: None,
                h2d_copy_finished_callback_data: std::ptr::null_mut() as *mut c_void,
                d2h_copy_finished: bc_event {
                    handle: std::ptr::null_mut() as *mut c_void,
                },
                d2h_copy_finished_callback: None,
                d2h_copy_finished_callback_data: std::ptr::null_mut() as *mut c_void,
                can_overwrite_inputs: false,
                coset_index: coset_idx as u32,
                log_extension_degree: log_extension_degree,
            };

            cfgs.push(cfg);
            dev_ids.push(device_id as i32);
        }

        // let cfgs = &mut cfgs as *mut Vec<ntt_configuration> as *mut ntt_configuration;
        // let dev_ids = &mut dev_ids as *mut Vec<i32> as *mut ::std::os::raw::c_int;

        let cfgs = &mut cfgs[0] as *mut ntt_configuration;
        let dev_ids = &mut dev_ids[0] as *mut ::std::os::raw::c_int;
        let log_n_dev = log_2(buffers.len());

        // dbg!(log_n_dev);
        unsafe {
            let result = ntt_execute_async_multigpu(cfgs, dev_ids, log_n_dev);
            if result != 0 {
                return Err(GpuError::MultiGpuLargeNTTErr(result));
            };
        }

        for (ctx_id, buffer) in buffers.iter_mut().enumerate() {
            let ctx_id = ctx_id % num_gpus;
            buffer.read_event.record(ctx[ctx_id].exec_stream())?;
            buffer.write_event.record(ctx[ctx_id].exec_stream())?;
        }

        // if final_bitreverse {
        //     DeviceBuf::multigpu_bitreverse(buffers, ctx)?;
        // }

        Ok(())
    }

    pub fn bitreverse(buffers: &mut Vec<&mut Self>, ctx: &mut GpuContext) -> GpuResult<()> {
        set_device(ctx.device_id())?;
        let mut length = buffers[0].len();
        for buffer in buffers.iter() {
            ctx.exec_stream.wait(buffer.write_event())?;
            ctx.exec_stream.wait(buffer.read_event())?;
        }

        for i in 0..(buffers.len() - 1) {
            assert_eq!(
                buffers[i].as_ptr(length..length),
                buffers[i + 1].as_ptr(0..0),
                "Buffers should be allocated one by one"
            );

            assert_eq!(
                length,
                buffers[i + 1].len(),
                "Buffers' lengths are not equal"
            );
        }

        let d_scalars = buffers[0].as_ptr(0..length);
        length = length * buffers.len();

        let log_scalars_count = log_2(length);
        unsafe {
            let result = ff_bit_reverse(
                d_scalars as *const c_void,
                d_scalars as *mut c_void,
                log_scalars_count as u32,
                ctx.exec_stream.inner,
            );
            if result != 0 {
                return Err(GpuError::BitReverseErr(result));
            }
        }

        for buffer in buffers.into_iter() {
            buffer.read_event.record(ctx.exec_stream())?;
            buffer.write_event.record(ctx.exec_stream())?;
        }

        Ok(())
    }

    pub fn multigpu_bitreverse(
        buffers: &mut Vec<Self>,
        ctx: &mut Vec<GpuContext>,
    ) -> GpuResult<()> {
        let mut length = buffers[0].len();
        for i in 0..(buffers.len() - 1) {
            assert_eq!(
                length,
                buffers[i + 1].len(),
                "Buffers' lengths are not equal"
            );
        }

        length = length * buffers.len();
        let log_scalars_count = log_2(length);

        let mut values = vec![];
        let mut dev_ids = vec![];
        let mut streams = vec![];

        for (buffer_idx, buffer) in buffers.iter().enumerate() {
            let device_id = ctx[buffer_idx].device_id();
            set_device(device_id)?;
            ctx[buffer_idx].exec_stream.wait(buffer.write_event())?;
            ctx[buffer_idx].exec_stream.wait(buffer.read_event())?;

            let d_scalars = buffer.as_ptr(0..buffer.len());

            values.push(d_scalars);
            dev_ids.push(device_id as i32);
            streams.push(ctx[buffer_idx].exec_stream.inner);
        }

        let values = &mut values[0] as *mut *const Fr;
        let dev_ids = &mut dev_ids[0] as *mut ::std::os::raw::c_int;
        let streams = &mut streams[0] as *mut bc_stream;
        unsafe {
            let result = ff_bit_reverse_multigpu(
                values as *mut *const c_void,
                values as *mut *mut c_void,
                log_scalars_count as u32,
                streams,
                dev_ids,
                log_2(buffers.len()),
            );
            if result != 0 {
                return Err(GpuError::MultiGpuBitReverseErr(result));
            }
        }

        for (device_id, buffer) in buffers.iter_mut().enumerate() {
            buffer.read_event.record(ctx[device_id].exec_stream())?;
            buffer.write_event.record(ctx[device_id].exec_stream())?;
        }

        Ok(())
    }
}

pub fn log_2(num: usize) -> u32 {
    assert!(num > 0);
    let mut pow = 0;
    while (1 << (pow + 1)) <= num {
        pow += 1;
    }
    assert_eq!(1 << pow, num);
    pow
}

#[inline(always)]
pub fn bitreverse(n: usize, l: usize) -> usize {
    let mut r = n.reverse_bits();
    // now we need to only use the bits that originally were "last" l, so shift

    r >>= (std::mem::size_of::<usize>() * 8) - l;

    r
}
