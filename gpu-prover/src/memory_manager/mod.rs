use super::*;
use crate::cuda_bindings::{device_info, DeviceBuf, GpuContext, GpuResult};
use core::ops::Range;
use gpu_ffi::*;

mod copying_operations;
mod polynomial_identifiers;
mod polynomial_operations;
mod proving_operations;
#[cfg(test)]
mod tests;
pub use copying_operations::*;
use itertools::Itertools;
pub use polynomial_identifiers::*;
pub use polynomial_operations::*;
pub use proving_operations::*;

const NUM_MSM_RESULT_POINTS: usize = 254;
const NUM_POLY_EVAL_RESULT_ELEMS: usize = 1;

pub struct DeviceMemoryManager<F: PrimeField, MC: ManagerConfigs> {
    pub(crate) ctx: Vec<GpuContext>,
    pub(crate) slots: Vec<(Vec<DeviceBuf<F>>, SlotStatus)>,
    pub(crate) host_slots: Vec<(AsyncVec<F>, SlotStatus)>,
    pub(crate) host_buf_for_msm: AsyncVec<G1>,
    pub(crate) host_buf_for_poly_eval: AsyncVec<Fr>,
    _configs: std::marker::PhantomData<MC>,
}

impl<F: PrimeField, MC: ManagerConfigs> DeviceMemoryManager<F, MC> {
    pub fn init(device_ids: &[usize], bases: &[CompactG1Affine]) -> GpuResult<Self> {
        assert_eq!(
            bases.len(),
            MC::FULL_SLOT_SIZE,
            "number of bases should be equal to size of full slot"
        );
        dbg!(device_ids);
        let num_devices = device_ids.len();
        assert_eq!(num_devices, MC::NUM_GPUS);
        let mut ctx = vec![];
        for (device_id, bases_chunk) in device_ids.iter().zip(bases.chunks(MC::SLOT_SIZE)) {
            let mut context = GpuContext::new_with_affinity(*device_id, device_ids)?;

            context.set_up_ff()?;
            context.set_up_ntt()?;
            context.set_up_pn()?;
            context.set_up_msm(bases_chunk)?;

            ctx.push(context);
        }
        let mut manager = Self {
            ctx,
            slots: Vec::with_capacity(MC::NUM_SLOTS),
            host_slots: Vec::with_capacity(MC::NUM_HOST_SLOTS),
            host_buf_for_msm: AsyncVec::allocate_new(NUM_MSM_RESULT_POINTS),
            host_buf_for_poly_eval: AsyncVec::allocate_new(NUM_POLY_EVAL_RESULT_ELEMS),
            _configs: std::marker::PhantomData::<MC>,
        };

        manager.allocate_slots().unwrap();
        manager.allocate_host_slots().unwrap();
        manager.allocate_mem_pool().unwrap();

        Ok(manager)
    }

    pub fn allocate_new(bases: &[CompactG1Affine]) -> GpuResult<Self> {
        assert_eq!(
            bases.len(),
            MC::FULL_SLOT_SIZE,
            "number of bases should be equal to size of full slot"
        );

        let mut ctx = vec![];
        for (device_id, bases_chunk) in bases.chunks(MC::SLOT_SIZE).enumerate() {
            let mut context = GpuContext::new(device_id)?;

            context.set_up_ff()?;
            context.set_up_ntt()?;
            context.set_up_pn()?;
            context.set_up_msm(bases_chunk)?;

            ctx.push(context);
        }

        Ok(Self {
            ctx,
            slots: Vec::with_capacity(MC::NUM_SLOTS),
            host_slots: Vec::with_capacity(MC::NUM_HOST_SLOTS),
            host_buf_for_msm: AsyncVec::allocate_new(NUM_MSM_RESULT_POINTS),
            host_buf_for_poly_eval: AsyncVec::allocate_new(NUM_POLY_EVAL_RESULT_ELEMS),
            _configs: std::marker::PhantomData::<MC>,
        })
    }

    pub fn new_from_ctx(ctx: Vec<GpuContext>) -> Self {
        assert_eq!(ctx.len(), MC::NUM_GPUS, "number of GpuContexts is wrong");
        for id in 0..MC::NUM_GPUS {
            // assert_eq!(ctx[id].device_id(), id, "enumeration of GpuContex is wrong");
            assert!(ctx[id].ff, "ff should be set up for GpuContex");
            assert!(ctx[id].ntt, "ntt should be set up for GpuContex");
            assert!(
                ctx[id].bases.is_none(),
                "msm should be set up for GpuContex"
            );
        }

        Self {
            ctx,
            slots: Vec::with_capacity(MC::NUM_SLOTS),
            host_slots: Vec::with_capacity(MC::NUM_HOST_SLOTS),
            host_buf_for_msm: AsyncVec::allocate_new(NUM_MSM_RESULT_POINTS),
            host_buf_for_poly_eval: AsyncVec::allocate_new(NUM_POLY_EVAL_RESULT_ELEMS),
            _configs: std::marker::PhantomData::<MC>,
        }
    }

    pub fn allocate_mem_pool(&mut self) -> GpuResult<()> {
        for id in 0..MC::NUM_GPUS {
            self.ctx[id].set_up_mem_pool()?;
        }
        Ok(())
    }

    // pub fn allocate_dummy_static_memory(&mut self) -> GpuResult<()> {
    //     for device_id in 0..MC::NUM_GPUS {
    //         let info = device_info(device_id as i32)?;

    //         const LOG_SLACK: u32 = 25;
    //         let size = (((info.free >> LOG_SLACK) - 1) << LOG_SLACK) >> 5;

    //         self.dummy_buf.push(DeviceBuf::<F>::alloc_static(
    //             &self.ctx[device_id],
    //             size as usize,
    //         )?);
    //     }
    //     Ok(())
    // }

    // pub fn deallocate_dummy_static_memory(&mut self) -> GpuResult<()> {
    //     self.dummy_buf.clear();

    //     Ok(())
    // }

    pub fn allocate_slots(&mut self) -> GpuResult<()> {
        assert_eq!(self.slots.len(), 0, "slots are already allocated");

        let mut slots: Vec<Vec<DeviceBuf<F>>> = (0..MC::NUM_SLOTS).map(|_| vec![]).collect();

        for ctx_id in 0..MC::NUM_GPUS {
            let mut big_splited_buf =
                DeviceBuf::<F>::alloc_static(&self.ctx[ctx_id], MC::SLOT_SIZE * MC::NUM_SLOTS)?
                    .split(MC::NUM_SLOTS);

            for slot_id in (0..MC::NUM_SLOTS).rev() {
                slots[slot_id].push(big_splited_buf.pop().unwrap());
            }
        }

        for slot in slots.into_iter() {
            self.slots.push((slot, SlotStatus::Free));
        }
        Ok(())
    }

    pub fn allocate_host_slots(&mut self) -> GpuResult<()> {
        assert_eq!(self.host_slots.len(), 0, "host slots are already allocated");

        for idx in 0..MC::NUM_HOST_SLOTS {
            let host_slot = AsyncVec::allocate_new(MC::FULL_SLOT_SIZE);
            self.host_slots.push((host_slot, SlotStatus::Free));
        }
        Ok(())
    }

    pub fn new_empty_slot(&mut self, id: PolyId, form: PolyForm) {
        assert!(
            self.get_slot_idx(id, form).is_none(),
            "There already exists polynomial {:?} {:?}",
            id,
            form
        );
        let idx = self
            .free_slot_idx()
            .expect(&format!("No free slot for {:?} {:?}", id, form));
        self.slots[idx].1 = SlotStatus::Busy(id, form);
    }

    pub fn sync(&mut self) -> GpuResult<()> {
        for ctx in self.ctx.iter() {
            ctx.sync()?;
        }
        Ok(())
    }

    pub fn free_slot(&mut self, id: PolyId, form: PolyForm) {
        let idx = self
            .get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));
        self.slots[idx].1 = SlotStatus::Free;
    }
    pub fn free_all_slots(&mut self) {
        for slot in self.slots.iter_mut() {
            slot.1 = SlotStatus::Free
        }
    }

    pub fn free_host_slot(&mut self, id: PolyId, form: PolyForm) {
        let idx = self
            .get_host_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));
        self.host_slots[idx].1 = SlotStatus::Free;
    }

    pub fn get_host_slot_values(&self, id: PolyId, form: PolyForm) -> GpuResult<&[F]> {
        let idx = self
            .get_host_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));
        self.host_slots[idx].0.get_values()
    }

    pub fn get_host_slot_values_mut(&mut self, id: PolyId, form: PolyForm) -> GpuResult<&mut [F]> {
        let idx = self
            .get_host_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));
        self.host_slots[idx].0.get_values_mut()
    }

    pub fn get_free_host_slot_values_mut(
        &mut self,
        id: PolyId,
        form: PolyForm,
    ) -> GpuResult<&mut [F]> {
        assert!(
            self.get_host_slot_idx(id, form).is_none(),
            "There is already such polynomial: {:?} {:?}",
            id,
            form
        );
        let idx = self
            .free_host_slot_idx()
            .expect(&format!("No free host slot for {:?} {:?}", id, form));

        self.host_slots[idx].1 = SlotStatus::Busy(id, form);
        self.host_slots[idx].0.get_values_mut()
    }

    pub fn rename_slot(&mut self, id: PolyId, new_id: PolyId, form: PolyForm) {
        assert!(
            self.get_slot_idx(new_id, form).is_none(),
            "There is already such polynomial: {:?} {:?}",
            id,
            form
        );
        let idx = self
            .get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));

        self.slots[idx].1 = SlotStatus::Busy(new_id, form)
    }

    pub fn polynomials_on_device(&self) -> Vec<(PolyId, PolyForm)> {
        assert_eq!(
            self.slots.len(),
            MC::NUM_SLOTS,
            "slots are not allocated yet"
        );

        let mut res = vec![];
        for poly in self.slots.iter() {
            match poly.1 {
                SlotStatus::Busy(id, form) => res.push((id, form)),
                _ => {}
            }
        }
        res
    }

    pub fn number_of_free_slots(&self) -> usize {
        assert_eq!(
            self.slots.len(),
            MC::NUM_SLOTS,
            "slots are not allocated yet"
        );

        let mut res = 0;
        for poly in self.slots.iter() {
            if poly.1 == SlotStatus::Free {
                res += 1;
            }
        }
        res
    }

    pub fn free_slot_idx(&self) -> Option<usize> {
        assert_eq!(
            self.slots.len(),
            MC::NUM_SLOTS,
            "slots are not allocated yet"
        );

        for idx in 0..MC::NUM_SLOTS {
            if self.slots[idx].1 == SlotStatus::Free {
                return Some(idx);
            }
        }
        None
    }

    pub fn get_slot_idx(&self, id: PolyId, form: PolyForm) -> Option<usize> {
        assert_eq!(
            self.slots.len(),
            MC::NUM_SLOTS,
            "slots are not allocated yet"
        );

        for idx in 0..MC::NUM_SLOTS {
            if self.slots[idx].1 == SlotStatus::Busy(id, form) {
                return Some(idx);
            }
        }
        None
    }

    pub fn free_host_slot_idx(&self) -> Option<usize> {
        assert_eq!(
            self.host_slots.len(),
            MC::NUM_HOST_SLOTS,
            "host slots are not allocated yet"
        );

        for idx in 0..MC::NUM_HOST_SLOTS {
            if self.host_slots[idx].1 == SlotStatus::Free {
                return Some(idx);
            }
        }
        None
    }

    pub fn get_host_slot_idx(&self, id: PolyId, form: PolyForm) -> Option<usize> {
        assert_eq!(
            self.host_slots.len(),
            MC::NUM_HOST_SLOTS,
            "host slots are not allocated yet"
        );

        for idx in 0..MC::NUM_HOST_SLOTS {
            if self.host_slots[idx].1 == SlotStatus::Busy(id, form) {
                return Some(idx);
            }
        }
        None
    }

    pub fn get_free_big_slot_idx(&self, size: usize) -> Option<usize> {
        assert!(
            self.slots.len() == MC::NUM_SLOTS,
            "slots are not allocated yet"
        );
        assert!(
            size <= MC::NUM_SLOTS,
            "requested size on big slot is bigger than number of slots"
        );

        for slot_idx in (size..MC::NUM_SLOTS).rev() {
            let mut is_free = true;
            for j in 0..size {
                if self.slots[slot_idx - j].1 != SlotStatus::Free {
                    is_free = false;
                    break;
                }
            }

            if is_free {
                return Some(slot_idx - size + 1);
            }
        }

        None
    }

    fn get_ctx_id_by_device_id(&self, device_id: usize) -> usize {
        let available_devices = self.ctx.iter().map(|c| c.device_id()).join(", ");
        self.ctx
            .iter()
            .position(|c| c.device_id() == device_id)
            .expect(&format!(
                "unknown device id: {}, available: {}",
                device_id, available_devices
            ))
    }
}

impl<F: PrimeField, MC: ManagerConfigs> Drop for DeviceMemoryManager<F, MC> {
    fn drop(&mut self) {
        for ctx_id in 0..MC::NUM_GPUS {
            for slot in self.slots.iter_mut() {
                slot.0[ctx_id].read_event.sync().unwrap();
                slot.0[ctx_id].write_event.sync().unwrap();
            }

            if self.slots.len() > 0 {
                let ptr = self.slots[0].0[ctx_id].as_mut_ptr(0..0);
                if unsafe { bc_free(ptr as *mut c_void) } != 0 {
                    panic!("Can't free memory of DeviceBufs of DeviceMemoryManager");
                }
            }
        }
    }
}
