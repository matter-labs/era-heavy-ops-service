use super::*;
use crate::cuda_bindings::{DeviceBuf, GpuError, Operation};

impl<MC: ManagerConfigs> DeviceMemoryManager<Fr, MC> {
    pub fn set_values(&mut self, id: PolyId, form: PolyForm, value: Fr) -> Result<(), GpuError> {
        let idx = self
            .get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));

        for device_id in 0..MC::NUM_GPUS {
            self.slots[idx].0[device_id].async_exec_op(
                &mut self.ctx[device_id],
                None,
                Some(value),
                0..MC::SLOT_SIZE,
                Operation::SetValue,
            )?;
        }

        Ok(())
    }

    pub fn set_values_with_range(
        &mut self,
        id: PolyId,
        form: PolyForm,
        value: Fr,
        range: Range<usize>,
    ) -> Result<(), GpuError> {
        assert!(range.len() > 0);
        let idx = self
            .get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));

        for device_id in 0..MC::NUM_GPUS {
            let chunk_start = device_id * MC::SLOT_SIZE;
            let chunk_end = chunk_start + MC::SLOT_SIZE;

            if range.start >= chunk_end || range.end <= chunk_start {
                continue;
            }

            let start = range.start.max(chunk_start) - chunk_start;
            let end = range.end.min(chunk_end) - chunk_start;

            self.slots[idx].0[device_id].async_exec_op(
                &mut self.ctx[device_id],
                None,
                Some(value),
                start..end,
                Operation::SetValue,
            )?;
        }

        Ok(())
    }

    pub fn create_x_poly_in_free_slot(
        &mut self,
        id: PolyId,
        form: PolyForm,
    ) -> Result<(), GpuError> {
        self.new_empty_slot(id, form);
        let idx = self.get_slot_idx(id, form).unwrap();

        match form {
            PolyForm::Monomial => {
                self.set_values(id, form, Fr::one())?;

                self.slots[idx].0[0].async_exec_op(
                    &mut self.ctx[0],
                    None,
                    Some(Fr::one()),
                    1..2,
                    Operation::SetValue,
                )?;
            }
            PolyForm::Values => {
                self.set_values(id, form, Fr::one())?;
                self.distribute_omega_powers(id, form, MC::FULL_SLOT_SIZE_LOG, 0, 1, false)?;
            }
            PolyForm::LDE(i) => {
                let mut g = domain_generator::<Fr>(4 * MC::FULL_SLOT_SIZE);
                let bitrevessed_idx = [0, 2, 1, 3];
                g = g.pow([bitrevessed_idx[i] as u64]);
                g.mul_assign(&Fr::multiplicative_generator());
                self.set_values(id, form, g)?;
                self.distribute_omega_powers(id, form, MC::FULL_SLOT_SIZE_LOG, 0, 1, false)?;

                self.multigpu_bitreverse(id, form)?;
            }
        }

        Ok(())
    }

    pub fn create_lagrange_poly_in_free_slot(
        &mut self,
        id: PolyId,
        form: PolyForm,
        point: usize,
    ) -> Result<(), GpuError> {
        assert!(point < MC::FULL_SLOT_SIZE);

        match form {
            PolyForm::Values => {
                self.new_empty_slot(id, form);
                let idx = self
                    .get_slot_idx(id, form)
                    .expect(&format!("No such polynomial: {:?} {:?}", id, form));

                self.set_values(id, form, Fr::zero())?;

                let device_id = point / MC::SLOT_SIZE;
                let point = point % MC::SLOT_SIZE;

                self.slots[idx].0[device_id].async_exec_op(
                    &mut self.ctx[device_id],
                    None,
                    Some(Fr::one()),
                    point..(point + 1),
                    Operation::SetValue,
                )?;
            }
            PolyForm::Monomial => {
                self.new_empty_slot(id, form);

                let x = Fr::from_str(&MC::FULL_SLOT_SIZE.to_string())
                    .unwrap()
                    .inverse()
                    .unwrap();
                self.set_values(id, form, x)?;

                let omega_pow = domain_generator::<Fr>(MC::FULL_SLOT_SIZE)
                    .pow([point as u64])
                    .inverse()
                    .unwrap();

                self.distribute_omega_powers(id, form, MC::FULL_SLOT_SIZE_LOG, 0, point, true)?;
            }
            PolyForm::LDE(i) => {
                self.create_lagrange_poly_in_free_slot(id, PolyForm::Monomial, point)?;
                self.multigpu_coset_fft(id, i)?;
            }
        }

        Ok(())
    }

    pub fn devide_monomial_by_degree_one_monomial(
        &mut self,
        id: PolyId,
        coeff: Fr,
    ) -> Result<(), GpuError> {
        // f(x) - f(coeff) values
        let value: Fr = self.evaluate_at(id, coeff)?.get_result(self)?;
        self.multigpu_fft(id, false)?;
        self.sub_constant(id, PolyForm::Values, value)?;

        // 1 / (x - coeff) values
        self.create_x_poly_in_free_slot(PolyId::Custom("1 / (x - coeff)"), PolyForm::Values)?;
        self.sub_constant(PolyId::Custom("1 / (x - coeff)"), PolyForm::Values, coeff)?;
        self.batch_inversion(PolyId::Custom("1 / (x - coeff)"), PolyForm::Values)?;

        // (f(x) - f(coeff)) / (x - coeff) monomial
        self.mul_assign(id, PolyId::Custom("1 / (x - coeff)"), PolyForm::Values)?;
        self.free_slot(PolyId::Custom("1 / (x - coeff)"), PolyForm::Values);
        self.multigpu_ifft(id, false)?;

        Ok(())
    }

    pub fn bitreverse(&mut self, id: PolyId, form: PolyForm, device_id: usize) -> GpuResult<()> {
        let idx = self.get_slot_idx(id, form).expect(&format!(
            "No such polynomial in such form: {:?} {:?}",
            id, form
        ));

        let big_slot_idx = self
            .get_free_big_slot_idx(MC::NUM_GPUS)
            .expect("there is no N free slots in a raw");

        for i in 0..MC::NUM_GPUS {
            let (slot, big_slot) = get_two_mut(&mut self.slots, idx, big_slot_idx + i);

            slot.0[i].async_copy_to_device(
                &mut self.ctx[device_id],
                &mut big_slot.0[device_id],
                0..MC::SLOT_SIZE,
                0..MC::SLOT_SIZE,
            )?;
        }

        let mut buffers_slice = &mut self.slots[(big_slot_idx)..(big_slot_idx + MC::NUM_GPUS)];
        let mut big_buffer = vec![];
        for i in 0..MC::NUM_GPUS {
            let mut buffer: &mut [_] = &mut [];
            (buffer, buffers_slice) = buffers_slice.split_at_mut(1);
            big_buffer.push(&mut buffer[0].0[device_id]);
        }

        DeviceBuf::bitreverse(&mut big_buffer, &mut self.ctx[device_id])?;

        for i in 0..MC::NUM_GPUS {
            let (slot, big_slot) = get_two_mut(&mut self.slots, idx, big_slot_idx + i);

            big_slot.0[device_id].async_copy_to_device(
                &mut self.ctx[device_id],
                &mut slot.0[i],
                0..MC::SLOT_SIZE,
                0..MC::SLOT_SIZE,
            )?;
        }

        self.slots[idx].1 = SlotStatus::Busy(id, form);

        Ok(())
    }

    pub fn multigpu_bitreverse(&mut self, id: PolyId, form: PolyForm) -> GpuResult<()> {
        let idx = self.get_slot_idx(id, form).expect(&format!(
            "No such polynomial in such form: {:?} {:?}",
            id, form
        ));

        DeviceBuf::multigpu_bitreverse(&mut self.slots[idx].0, &mut self.ctx)?;

        self.slots[idx].1 = SlotStatus::Busy(id, form);

        Ok(())
    }

    pub fn evaluate_at(&mut self, id: PolyId, base: Fr) -> Result<EvaluationHandle, GpuError> {
        let idx = self
            .get_slot_idx(id, PolyForm::Monomial)
            .expect(&format!("No such polynomial in monomial form: {:?}", id));

        let mut res_buffers = vec![];
        for device_id in 0..MC::NUM_GPUS {
            res_buffers
                .push(self.slots[idx].0[device_id].evaluate_at(&mut self.ctx[device_id], base)?);
        }
        let base_pow = base.pow([MC::SLOT_SIZE as u64]);

        Ok(EvaluationHandle::from_buffers_and_base_pow(
            res_buffers,
            base_pow,
        ))
    }
}

pub(crate) fn domain_generator<F: PrimeField>(degree: usize) -> F {
    let n = degree.next_power_of_two();

    let mut k = n;
    let mut power_of_two = 0;
    while k != 1 {
        k >>= 1;
        power_of_two += 1;
    }

    let max_power_of_two = F::S;

    assert!(power_of_two <= max_power_of_two);

    let mut generator = F::root_of_unity();

    for _ in power_of_two..max_power_of_two {
        generator.square()
    }

    generator
}
