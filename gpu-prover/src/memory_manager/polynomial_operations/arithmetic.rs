use super::*;
use crate::cuda_bindings::{DeviceBuf, GpuError, Operation};

impl<MC: ManagerConfigs> DeviceMemoryManager<Fr, MC> {
    pub fn add_constant(&mut self, id: PolyId, form: PolyForm, constant: Fr) -> GpuResult<()> {
        let idx = self
            .get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));

        for device_id in 0..MC::NUM_GPUS {
            self.slots[idx].0[device_id].async_exec_op(
                &mut self.ctx[device_id],
                None,
                Some(constant),
                0..MC::SLOT_SIZE,
                Operation::AddConst,
            )?;
        }

        Ok(())
    }

    pub fn sub_constant(&mut self, id: PolyId, form: PolyForm, constant: Fr) -> GpuResult<()> {
        let idx = self
            .get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));

        for device_id in 0..MC::NUM_GPUS {
            self.slots[idx].0[device_id].async_exec_op(
                &mut self.ctx[device_id],
                None,
                Some(constant),
                0..MC::SLOT_SIZE,
                Operation::SubConst,
            )?;
        }

        Ok(())
    }

    pub fn mul_constant(&mut self, id: PolyId, form: PolyForm, constant: Fr) -> GpuResult<()> {
        let idx = self
            .get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));

        for device_id in 0..MC::NUM_GPUS {
            self.slots[idx].0[device_id].async_exec_op(
                &mut self.ctx[device_id],
                None,
                Some(constant),
                0..MC::SLOT_SIZE,
                Operation::MulConst,
            )?;
        }

        Ok(())
    }

    pub fn add_assign(&mut self, id_1: PolyId, id_2: PolyId, form: PolyForm) -> GpuResult<()> {
        let idx_1 = self
            .get_slot_idx(id_1, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id_1, form));
        let idx_2 = self
            .get_slot_idx(id_2, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id_2, form));

        for device_id in 0..MC::NUM_GPUS {
            let (slot_1, slot_2) = get_two_mut(&mut self.slots, idx_1, idx_2);

            slot_1.0[device_id].async_exec_op(
                &mut self.ctx[device_id],
                Some(&mut slot_2.0[device_id]),
                None,
                0..MC::SLOT_SIZE,
                Operation::Add,
            )?;
        }

        Ok(())
    }

    pub fn sub_assign(&mut self, id_1: PolyId, id_2: PolyId, form: PolyForm) -> GpuResult<()> {
        let idx_1 = self
            .get_slot_idx(id_1, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id_1, form));
        let idx_2 = self
            .get_slot_idx(id_2, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id_2, form));

        for device_id in 0..MC::NUM_GPUS {
            let (slot_1, slot_2) = get_two_mut(&mut self.slots, idx_1, idx_2);

            slot_1.0[device_id].async_exec_op(
                &mut self.ctx[device_id],
                Some(&mut slot_2.0[device_id]),
                None,
                0..MC::SLOT_SIZE,
                Operation::Sub,
            )?;
        }

        Ok(())
    }

    pub fn mul_assign(&mut self, id_1: PolyId, id_2: PolyId, form: PolyForm) -> GpuResult<()> {
        let idx_1 = self
            .get_slot_idx(id_1, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id_1, form));
        let idx_2 = self
            .get_slot_idx(id_2, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id_2, form));

        for device_id in 0..MC::NUM_GPUS {
            let (slot_1, slot_2) = get_two_mut(&mut self.slots, idx_1, idx_2);

            slot_1.0[device_id].async_exec_op(
                &mut self.ctx[device_id],
                Some(&mut slot_2.0[device_id]),
                None,
                0..MC::SLOT_SIZE,
                Operation::Mul,
            )?;
        }

        Ok(())
    }

    pub fn add_assign_scaled(
        &mut self,
        id_1: PolyId,
        id_2: PolyId,
        form: PolyForm,
        constant: Fr,
    ) -> GpuResult<()> {
        let idx_1 = self
            .get_slot_idx(id_1, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id_1, form));
        let idx_2 = self
            .get_slot_idx(id_2, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id_2, form));

        for device_id in 0..MC::NUM_GPUS {
            let (slot_1, slot_2) = get_two_mut(&mut self.slots, idx_1, idx_2);

            slot_1.0[device_id].async_exec_op(
                &mut self.ctx[device_id],
                Some(&mut slot_2.0[device_id]),
                Some(constant),
                0..MC::SLOT_SIZE,
                Operation::AddScaled,
            )?;
        }

        Ok(())
    }

    pub fn sub_assign_scaled(
        &mut self,
        id_1: PolyId,
        id_2: PolyId,
        form: PolyForm,
        constant: Fr,
    ) -> GpuResult<()> {
        let idx_1 = self
            .get_slot_idx(id_1, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id_1, form));
        let idx_2 = self
            .get_slot_idx(id_2, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id_2, form));

        for device_id in 0..MC::NUM_GPUS {
            let (slot_1, slot_2) = get_two_mut(&mut self.slots, idx_1, idx_2);

            slot_1.0[device_id].async_exec_op(
                &mut self.ctx[device_id],
                Some(&mut slot_2.0[device_id]),
                Some(constant),
                0..MC::SLOT_SIZE,
                Operation::SubScaled,
            )?;
        }

        Ok(())
    }

    pub fn shifted_grand_product_to_new_slot(
        &mut self,
        id1: PolyId,
        id2: PolyId,
        form: PolyForm,
    ) -> Result<(), GpuError> {
        self.copy_shifted_from_device_to_free_device(id1, id2, form, Fr::one())?;
        self.grand_product(id2, form)?;
        Ok(())
    }

    pub fn grand_product(&mut self, id: PolyId, form: PolyForm) -> GpuResult<()> {
        assert!(MC::NUM_GPUS <= 2, "other cases are lot needed for now");

        let idx = self
            .get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));

        for device_id in 0..MC::NUM_GPUS {
            self.slots[idx].0[device_id].async_exec_op(
                &mut self.ctx[device_id],
                None,
                None,
                0..MC::SLOT_SIZE,
                Operation::GrandProd,
            )?;
        }

        // We need to multiply the second chunk
        // with the last element of the first one
        if MC::NUM_GPUS == 2 {
            unsafe {
                self.grand_product_inner(idx)?;
            }
        }

        Ok(())
    }

    unsafe fn grand_product_inner(&mut self, idx: usize) -> GpuResult<()> {
        assert_eq!(
            MC::NUM_GPUS,
            2,
            "this function should be used only for 2 GPUs"
        );
        let device_id = self.ctx[1].device_id();
        crate::cuda_bindings::set_device(device_id)?;

        self.ctx[1]
            .exec_stream
            .wait(self.slots[idx].0[0].write_event())?;

        let poly = self.slots[idx].0[1].as_mut_ptr(0..MC::SLOT_SIZE);
        let constant = self.slots[idx].0[0].as_ptr((MC::SLOT_SIZE - 1)..MC::SLOT_SIZE);

        let result = ff_ax(
            constant as *const c_void,
            poly as *const c_void,
            poly as *mut c_void,
            MC::SLOT_SIZE as u32,
            self.ctx[1].exec_stream.inner,
        );
        if result != 0 {
            return Err(GpuError::SchedulingErr(result));
        }

        self.slots[idx].0[0]
            .read_event
            .record(self.ctx[1].exec_stream())?;
        self.slots[idx].0[1]
            .write_event
            .record(self.ctx[1].exec_stream())?;

        Ok(())
    }

    pub fn batch_inversion(&mut self, id: PolyId, form: PolyForm) -> GpuResult<()> {
        let idx = self
            .get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));

        for device_id in 0..MC::NUM_GPUS {
            self.slots[idx].0[device_id].async_exec_op(
                &mut self.ctx[device_id],
                None,
                None,
                0..MC::SLOT_SIZE,
                Operation::BatchInv,
            )?;
        }

        Ok(())
    }

    // output[i] = input[i] * w^((i + offset)*shift)
    // w^(2^log_degree) = 1
    pub fn distribute_omega_powers(
        &mut self,
        id: PolyId,
        form: PolyForm,
        log_degree: usize,
        offset: usize,
        shift: usize,
        inverse: bool,
    ) -> GpuResult<()> {
        let idx = self
            .get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));

        for device_id in 0..MC::NUM_GPUS {
            self.slots[idx].0[device_id].distribute_omega_powers(
                &mut self.ctx[device_id],
                log_degree,
                offset + device_id * MC::SLOT_SIZE,
                shift,
                inverse,
            )?;
        }

        Ok(())
    }

    pub fn distribute_powers(
        &mut self,
        id: PolyId,
        form: PolyForm,
        base: Fr,
    ) -> Result<(), GpuError> {
        let idx = self
            .get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));
        self.new_empty_slot(PolyId::Custom("powers of base"), form);
        let tmp_idx = self
            .get_slot_idx(PolyId::Custom("powers of base"), form)
            .unwrap();

        self.set_values(PolyId::Custom("powers of base"), form, base)?;
        self.slots[tmp_idx].0[0].async_exec_op(
            &mut self.ctx[0],
            None,
            Some(Fr::one()),
            0..1,
            Operation::SetValue,
        )?;

        self.grand_product(PolyId::Custom("powers of base"), form)?;
        self.mul_assign(id, PolyId::Custom("powers of base"), form)?;
        self.free_slot(PolyId::Custom("powers of base"), form);

        Ok(())
    }
}
