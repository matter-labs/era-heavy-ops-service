use super::*;

impl<MC: ManagerConfigs> DeviceMemoryManager<Fr, MC> {
    pub fn msm(&mut self, id: PolyId) -> GpuResult<MSMHandle> {
        let idx = self.get_slot_idx(id, PolyForm::Monomial)
            .expect(&format!("No such polynomial in monomial form: {:?}", id));

        let mut res_buffers = vec![];
        for device_id in 0..MC::NUM_GPUS {
            res_buffers.push(
                self.slots[idx].0[device_id].msm(
                    &mut self.ctx[device_id]
                )?
            );
        }

        Ok(MSMHandle::from_buffers(res_buffers))
    }

    pub fn fft(&mut self, id: PolyId, bits_reversed: bool, device_id: usize) -> GpuResult<()> {
        self.ntt(id, bits_reversed, device_id, false, None, None, true)
    }

    pub fn ifft(&mut self, id: PolyId, bits_reversed: bool, device_id: usize) -> GpuResult<()> {
        self.ntt(id, bits_reversed, device_id, true, None, None, true)
    }

    pub fn fft_to_free_slot(&mut self, id: PolyId, bits_reversed: bool, device_id: usize) -> GpuResult<()> {
        // SAFETY: one of polynomials with the same status is immediately replaced
        unsafe {
            self.clone_slot_on_device(id, PolyForm::Monomial)?;
        }
        self.fft(id, bits_reversed, device_id)
    }

    pub fn ifft_to_free_slot(&mut self, id: PolyId, bits_reversed: bool, device_id: usize) -> GpuResult<()> {
        // SAFETY: one of polynomials with the same status is immediately replaced
        unsafe {
            self.clone_slot_on_device(id, PolyForm::Values)?;
        }
        self.ifft(id, bits_reversed, device_id)
    }

    pub fn coset_fft(&mut self, id: PolyId, coset_idx: usize, device_id: usize) -> GpuResult<()> {
        self.ntt(id, false, device_id, false, Some(coset_idx), Some(LDE_FACTOR as u32), false)
    }

    pub fn coset_fft_to_free_slot(&mut self, id: PolyId, coset_idx: usize, device_id: usize) -> GpuResult<()> {
        // SAFETY: one of polynomials with the same status is immediately replaced
        unsafe {
            self.clone_slot_on_device(id, PolyForm::Monomial)?;
        }
        self.coset_fft(id, coset_idx, device_id)
    }

    pub fn ntt(
        &mut self, 
        id: PolyId, 
        bits_reversed: bool, 
        device_id: usize, 
        inverse: bool, 
        coset_index: Option<usize>, 
        lde_factor: Option<u32>,
        final_bitreverse: bool,
    ) -> GpuResult<()> {
        let mut src_form = PolyForm::Monomial;
        let mut dst_form = PolyForm::Values;
        if inverse {
            src_form = PolyForm::Values;
            dst_form = PolyForm::Monomial;
        }
        if let Some(coset_idx) = coset_index {
            if inverse {
                src_form = PolyForm::LDE(coset_idx);
            } else {
                dst_form = PolyForm::LDE(coset_idx);
            }
        }

        assert!(device_id < MC::NUM_GPUS,
            "Device id is {}, while number of gpus is {}", device_id, MC::NUM_GPUS);
        assert!(self.get_slot_idx(id, dst_form).is_none(), "There is already such polynomial in such form: {:?} {:?}", id, dst_form);
        let idx = self.get_slot_idx(id, src_form)
            .expect(&format!("No such polynomial in such form {:?} {:?}", id, src_form));

        let big_slot_idx = self.get_free_big_slot_idx(MC::NUM_GPUS).expect("there is no N free slots in a raw");

        for i in 0..MC::NUM_GPUS {
            let (slot, big_slot) = get_two_mut(&mut self.slots, idx, big_slot_idx + i);

            slot.0[i].async_copy_to_device(
                &mut self.ctx[device_id], 
                &mut big_slot.0[device_id], 
                0..MC::SLOT_SIZE, 
                0..MC::SLOT_SIZE
            )?;
        }

        let mut buffers_slice = &mut self.slots[(big_slot_idx)..(big_slot_idx + MC::NUM_GPUS)];
        let mut big_buffer = vec![];
        for i in 0..MC::NUM_GPUS {
            let mut buffer: &mut [_] = &mut [];
            (buffer, buffers_slice) = buffers_slice.split_at_mut(1);
            big_buffer.push(&mut buffer[0].0[device_id]);
        }

        DeviceBuf::ntt(&mut big_buffer, &mut self.ctx[device_id], bits_reversed, inverse, coset_index, lde_factor, final_bitreverse)?;

        for i in 0..MC::NUM_GPUS {
            let (slot, big_slot) = get_two_mut(&mut self.slots, idx, big_slot_idx + i);

            big_slot.0[device_id].async_copy_to_device(
                &mut self.ctx[device_id], 
                &mut slot.0[i], 
                0..MC::SLOT_SIZE, 
                0..MC::SLOT_SIZE
            )?;
        }

        self.slots[idx].1 = SlotStatus::Busy(id, dst_form);

        Ok(())
    }




    pub fn multigpu_fft(&mut self, id: PolyId, bits_reversed: bool) -> GpuResult<()> {
        self.multigpu_ntt(id, bits_reversed, false, None, None, true)
    }

    pub fn multigpu_ifft(&mut self, id: PolyId, bits_reversed: bool) -> GpuResult<()> {
        self.multigpu_ntt(id, bits_reversed, true, None, None, true)
    }

    pub fn multigpu_fft_to_free_slot(&mut self, id: PolyId, bits_reversed: bool) -> GpuResult<()> {
        // SAFETY: one of polynomials with the same status is immediately replaced
        unsafe {
            self.clone_slot_on_device(id, PolyForm::Monomial)?;
        }
        self.multigpu_fft(id, bits_reversed)
    }

    pub fn multigpu_ifft_to_free_slot(&mut self, id: PolyId, bits_reversed: bool) -> GpuResult<()> {
        // SAFETY: one of polynomials with the same status is immediately replaced
        unsafe {
            self.clone_slot_on_device(id, PolyForm::Values)?;
        }
        self.multigpu_ifft(id, bits_reversed)
    }

    pub fn multigpu_coset_fft(&mut self, id: PolyId, coset_idx: usize) -> GpuResult<()> {
        self.multigpu_ntt(id, false, false, Some(coset_idx), Some(LDE_FACTOR as u32), false)
    }

    pub fn multigpu_coset_fft_to_free_slot(&mut self, id: PolyId, coset_idx: usize) -> GpuResult<()> {
        // SAFETY: one of polynomials with the same status is immediately replaced
        unsafe {
            self.clone_slot_on_device(id, PolyForm::Monomial)?;
        }
        self.multigpu_coset_fft(id, coset_idx)
    }

    pub fn multigpu_coset_ifft(&mut self, id: PolyId, coset_idx: usize) -> GpuResult<()> {
        self.multigpu_ntt(id, true, true, Some(coset_idx), Some(LDE_FACTOR as u32), false)
    }

    pub fn multigpu_ntt(
        &mut self, 
        id: PolyId, 
        bits_reversed: bool, 
        inverse: bool, 
        coset_index: Option<usize>, 
        lde_factor: Option<u32>,
        final_bitreverse: bool,
    ) -> GpuResult<()> {
        let mut src_form = PolyForm::Monomial;
        let mut dst_form = PolyForm::Values;
        if inverse {
            src_form = PolyForm::Values;
            dst_form = PolyForm::Monomial;
        }
        if let Some(coset_idx) = coset_index {
            if inverse {
                src_form = PolyForm::LDE(coset_idx);
            } else {
                dst_form = PolyForm::LDE(coset_idx);
            }
        }

        assert!(self.get_slot_idx(id, dst_form).is_none(), "There is already such polynomial in such form: {:?} {:?}", id, dst_form);
        let idx = self.get_slot_idx(id, src_form)
            .expect(&format!("No such polynomial in such form {:?} {:?}", id, src_form));

        DeviceBuf::multigpu_ntt(&mut self.slots[idx].0, &mut self.ctx, bits_reversed, inverse, coset_index, lde_factor, final_bitreverse)?;

        self.slots[idx].1 = SlotStatus::Busy(id, dst_form);

        Ok(())
    }

    pub fn parallel_ffts(&mut self, ids: Vec<PolyId>, bits_reversed: bool) -> GpuResult<()> {
        self.parallel_ntts(ids, vec![bits_reversed, bits_reversed], vec![false, false], None, None, true)
    }

    pub fn parallel_iffts(&mut self, ids: Vec<PolyId>, bits_reversed: bool) -> GpuResult<()> {
        self.parallel_ntts(ids, vec![bits_reversed, bits_reversed], vec![true, true], None, None, true)
    }

    pub fn parallel_ifft_fft(&mut self, ids: Vec<PolyId>, bits_reversed: bool) -> GpuResult<()> {
        self.parallel_ntts(ids, vec![bits_reversed, bits_reversed], vec![true, false], None, None, true)
    }

    pub fn parallel_coset_ffts(&mut self, ids: Vec<PolyId>, coset_idx: usize) -> GpuResult<()> {
        self.parallel_ntts(ids, vec![false, false], vec![false, false], Some(coset_idx), Some(LDE_FACTOR as u32), false)
    }

    pub fn parallel_iffts_to_free_slot(&mut self, ids: Vec<PolyId>, bits_reversed: bool) -> GpuResult<()> {
        // SAFETY: one of polynomials with the same status is immediately replaced
        unsafe {
            for id in ids.iter() {
                self.clone_slot_on_device(*id, PolyForm::Values)?;
            }
        }
        self.parallel_iffts(ids, bits_reversed)
    }

    pub fn parallel_ffts_to_free_slot(&mut self, ids: Vec<PolyId>, bits_reversed: bool) -> GpuResult<()> {
        // SAFETY: one of polynomials with the same status is immediately replaced
        unsafe {
            for id in ids.iter() {
                self.clone_slot_on_device(*id, PolyForm::Monomial)?;
            }
        }
        self.parallel_ffts(ids, bits_reversed)
    }

    pub fn parallel_ifft_fft_to_free_slot(&mut self, ids: Vec<PolyId>, bits_reversed: bool) -> GpuResult<()> {
        // SAFETY: one of polynomials with the same status is immediately replaced
        unsafe {
            for (i, id) in ids.iter().enumerate() {
                let mut src_form = PolyForm::Values;
                if i == 1 {
                    src_form = PolyForm::Monomial;
                }
                self.clone_slot_on_device(*id, src_form)?;
            }
        }
        self.parallel_ifft_fft(ids, bits_reversed)
    }

    pub fn parallel_coset_ffts_to_free_slot(&mut self, ids: Vec<PolyId>, coset_idx: usize) -> GpuResult<()> {
        // SAFETY: one of polynomials with the same status is immediately replaced
        unsafe {
            for id in ids.iter() {
                self.clone_slot_on_device(*id, PolyForm::Monomial)?;
            }
        }
        self.parallel_coset_ffts(ids, coset_idx)
    }

    pub fn parallel_ntts(
        &mut self, 
        ids: Vec<PolyId>, 
        bits_reversed: Vec<bool>, 
        inverse: Vec<bool>,
        coset_index: Option<usize>, 
        lde_factor: Option<u32>,
        final_bitreverse: bool,
    ) -> GpuResult<()> {
        let num_ffts = ids.len();

        assert!(ids.len() <= MC::NUM_GPUS,
            "Device id is {}, while number of gpus is {}", ids.len(), MC::NUM_GPUS);
        assert!(num_ffts <= MC::NUM_GPUS, "number of ffts is biger than number of gpus");
        for i in 0..(num_ffts - 1) {
            for j in (i+1)..num_ffts {
                assert_ne!(ids[i], ids[j], "ids in parallel ffts be different");
            }
        }

        let big_slot_idx = self.get_free_big_slot_idx(MC::NUM_GPUS).expect("there is no N free slots in a raw");

        for device_id in 0..num_ffts {
            let mut src_form = PolyForm::Monomial;
            let mut dst_form = PolyForm::Values;
            if inverse[device_id] {
                src_form = PolyForm::Values;
                dst_form = PolyForm::Monomial;
            }
            if let Some(coset_idx) = coset_index {
                if inverse[device_id] {
                    src_form = PolyForm::LDE(coset_idx);
                } else {
                    dst_form = PolyForm::LDE(coset_idx);
                }
            }

            assert!(self.get_slot_idx(ids[device_id], dst_form).is_none(), "There is already such polynomial in such form {:?} {:?}", ids[device_id], dst_form);
            let idx = self.get_slot_idx(ids[device_id], src_form)
                .expect(&format!("No such polynomial in such form {:?} {:?}", ids[device_id], src_form));
           
            for i in 0..MC::NUM_GPUS {
                let (slot, big_slot) = get_two_mut(&mut self.slots, idx, big_slot_idx + i);
    
                slot.0[i].async_copy_to_device(
                    &mut self.ctx[device_id], 
                    &mut big_slot.0[device_id], 
                    0..MC::SLOT_SIZE, 
                    0..MC::SLOT_SIZE
                )?;
            }

            let mut buffers_slice = &mut self.slots[(big_slot_idx)..(big_slot_idx + MC::NUM_GPUS)];
            let mut big_buffer = vec![];
            for i in 0..MC::NUM_GPUS {
                let mut buffer: &mut [_] = &mut [];
                (buffer, buffers_slice) = buffers_slice.split_at_mut(1);
                big_buffer.push(&mut buffer[0].0[device_id]);
            }
    
            DeviceBuf::ntt(&mut big_buffer, &mut self.ctx[device_id], bits_reversed[device_id], inverse[device_id], coset_index, lde_factor, final_bitreverse)?;
    
            for i in 0..MC::NUM_GPUS {
                let (slot, big_slot) = get_two_mut(&mut self.slots, idx, big_slot_idx + i);
    
                big_slot.0[device_id].async_copy_to_device(
                    &mut self.ctx[device_id], 
                    &mut slot.0[i], 
                    0..MC::SLOT_SIZE, 
                    0..MC::SLOT_SIZE
                )?;
            }

            self.slots[idx].1 = SlotStatus::Busy(ids[device_id], dst_form);
        }

        Ok(())
    }

    pub fn coset_4n_ifft(&mut self, ids: [PolyId; 4], device_id: usize) -> GpuResult<()> {
        for i in 0..3 {
            for j in (i+1)..4 {
                assert_ne!(ids[i], ids[j], "ids in 4n-ifft should be different");
            }
        }

        let big_slot_idx = self.get_free_big_slot_idx(4 * MC::NUM_GPUS).expect("there is no N free slots in a raw");

        for coset_idx in 0..4 {
            assert!(self.get_slot_idx(ids[coset_idx], PolyForm::Monomial).is_none(),
                "There is already such polynomial in monomial form: {:?}", ids[coset_idx]);
            let idx = self.get_slot_idx(ids[coset_idx], PolyForm::LDE(coset_idx))
                .expect(&format!("No such polynomial in such lde form: {:?}", ids[coset_idx]));

            for i in 0..MC::NUM_GPUS {
                let numb = big_slot_idx + i + coset_idx * MC::NUM_GPUS;
                let (slot, big_slot) = get_two_mut(&mut self.slots, idx, numb);

                slot.0[i].async_copy_to_device(
                    &mut self.ctx[device_id], 
                    &mut big_slot.0[device_id], 
                    0..MC::SLOT_SIZE, 
                    0..MC::SLOT_SIZE
                )?;
            }
        }

        let mut buffers_slice = &mut self.slots[(big_slot_idx)..(big_slot_idx + 4 * MC::NUM_GPUS)];
        let mut big_buffer = vec![];
        for _ in 0..(4 * MC::NUM_GPUS) {
            let mut buffer: &mut [_] = &mut [];
            (buffer, buffers_slice) = buffers_slice.split_at_mut(1);
            big_buffer.push(&mut buffer[0].0[device_id]);
        }

        DeviceBuf::ntt(&mut big_buffer, &mut self.ctx[device_id], true, true, None, None, false)?;


        for coset_idx in 0..4 {
            let idx = self.get_slot_idx(ids[coset_idx], PolyForm::LDE(coset_idx))
                .expect(&format!("No such polynomial in such lde form: {:?}", ids[coset_idx]));

            for i in 0..MC::NUM_GPUS {
                let numb = big_slot_idx + i + coset_idx * MC::NUM_GPUS;
                let (slot, big_slot) = get_two_mut(&mut self.slots, idx, numb);

                slot.0[i].async_copy_from_device(
                    &mut self.ctx[device_id], 
                    &mut big_slot.0[device_id], 
                    0..MC::SLOT_SIZE, 
                    0..MC::SLOT_SIZE
                )?;
            }
    
            self.slots[idx].1 = SlotStatus::Busy(ids[coset_idx], PolyForm::Monomial);
        }
        
        Ok(())
    }

    pub fn multigpu_coset_4n_ifft(&mut self, ids: [PolyId; 4]) -> GpuResult<()> {
        for i in 0..3 {
            for j in (i+1)..4 {
                assert_ne!(ids[i], ids[j], "ids in 4n-ifft should be different");
            }
        }

        let mut idxs = [0; 4];
        for coset_idx in 0..4 {
            assert!(self.get_slot_idx(ids[coset_idx], PolyForm::Monomial).is_none(),
                "There is already such polynomial in monomial form: {:?}", ids[coset_idx]);
            let idx = self.get_slot_idx(ids[coset_idx], PolyForm::LDE(coset_idx))
                .expect(&format!("No such polynomial in such lde form: {:?}", ids[coset_idx]));

            idxs[coset_idx] = idx;
        }

        let mut slots = get_multi_mut(&mut self.slots, idxs.to_vec());

        let mut big_buffer = vec![];
        for slot in slots.iter_mut() {
            for buf in slot.0.iter_mut() {
                big_buffer.push(buf);
            }
        }          


        DeviceBuf::multigpu_4n_ntt(&mut big_buffer, &mut self.ctx, true, true, None, None, false)?;

        for coset_idx in 0..4 {
            let idx = self.get_slot_idx(ids[coset_idx], PolyForm::LDE(coset_idx))
                .expect(&format!("No such polynomial in such lde form: {:?}", ids[coset_idx]));

            self.slots[idx].1 = SlotStatus::Busy(ids[coset_idx], PolyForm::Monomial);
        }
        
        Ok(())
    }
}
