use std::convert::TryInto;
use core::fmt::{Debug, Formatter};
use super::*;
use crate::cuda_bindings::{DeviceBuf, Operation, GpuError};

impl<MC: ManagerConfigs> DeviceMemoryManager<Fr, MC> {

    pub fn async_copy_to_device(
        &mut self, 
        poly: &mut AsyncVec<Fr>, 
        id: PolyId, 
        form: PolyForm,
        range: Range<usize>,
    ) -> GpuResult<()> {
        assert_eq!(range.len(), MC::FULL_SLOT_SIZE, "Wrong polynomial size");
        assert!(self.get_slot_idx(id, form).is_none(),
            "There is already such polynomial: {:?} {:?}", id, form);
        
        let idx = self.free_slot_idx().expect(&format!("No free slots: {:?} {:?}", id, form));

        for ctx_id in 0..MC::NUM_GPUS {
            let start = range.start + ctx_id * MC::SLOT_SIZE;
            let this_range = start..(start + MC::SLOT_SIZE);

            poly.async_copy_to_device(
                &mut self.ctx[ctx_id],
                &mut self.slots[idx].0[ctx_id],
                this_range,
                0..MC::SLOT_SIZE
            )?;
        }
        self.slots[idx].1 = SlotStatus::Busy(id, form);

        Ok(())
    }

    pub fn copy_to_device_with_host_slot(
        &mut self,
        worker: &Worker,
        poly: &[Fr],
        id: PolyId,
        form: PolyForm
    ) -> GpuResult<()> {
        assert_eq!(poly.len(), MC::FULL_SLOT_SIZE, "Wrong polynomial size");
        assert!(self.get_slot_idx(id, form).is_none(),
            "There is already such polynomial: {:?} {:?}", id, form);

        let host_idx = self.free_host_slot_idx().expect("No free host slots"); //sync??

        async_copy(
            worker, 
            self.host_slots[host_idx].0.get_values_mut().unwrap(),
            poly
        );

        let idx = self.free_slot_idx().expect("No free slots");

        for ctx_id in 0..MC::NUM_GPUS {
            let start = ctx_id * MC::SLOT_SIZE;
            let this_range = start..(start + MC::SLOT_SIZE);

            self.host_slots[host_idx].0.async_copy_to_device(
                &mut self.ctx[ctx_id],
                &mut self.slots[idx].0[ctx_id],
                this_range,
                0..MC::SLOT_SIZE
            )?;
        }

        self.host_slots[host_idx].1 = SlotStatus::Busy(id, form);
        self.slots[idx].1 = SlotStatus::Busy(id, form);

        Ok(())
    }

    pub fn copy_from_device_with_host_slot(
        &mut self,
        worker: &Worker,
        poly: &mut [Fr],
        id: PolyId,
        form: PolyForm
    ) -> GpuResult<()> {
        assert_eq!(poly.len(), MC::FULL_SLOT_SIZE, "Wrong polynomial size");
        assert!(self.get_host_slot_idx(id, form).is_none(),
        "There is already such host slot: {:?} {:?}", id, form);

        let host_idx = self.free_host_slot_idx().expect("No free host slots");
        let idx = self.get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));

        for ctx_id in 0..MC::NUM_GPUS {
            let start = ctx_id * MC::SLOT_SIZE;
            let this_range = start..(start + MC::SLOT_SIZE);

            self.host_slots[host_idx].0.async_copy_from_device(
                &mut self.ctx[ctx_id],
                &mut self.slots[idx].0[ctx_id],
                this_range,
                0..MC::SLOT_SIZE
            )?;
        }

        self.host_slots[host_idx].1 = SlotStatus::Busy(id, form);
        self.slots[idx].1 = SlotStatus::Busy(id, form);

        async_copy(
            worker, 
            poly,
            self.host_slots[host_idx].0.get_values_mut().unwrap(),
        );

        Ok(())
    }

    pub fn copy_from_device_to_host_pinned(
        &mut self, 
        id: PolyId,
        form: PolyForm,
    ) -> GpuResult<()> {
        assert!(self.get_host_slot_idx(id, form).is_none(),
            "There is already such host slot: {:?} {:?}", id, form);

        let host_idx = self.free_host_slot_idx().expect("No free host slots");
        let idx = self.get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));

        for ctx_id in 0..MC::NUM_GPUS {
            let start = ctx_id * MC::SLOT_SIZE;
            let this_range = start..(start + MC::SLOT_SIZE);

            self.host_slots[host_idx].0.async_copy_from_device(
                &mut self.ctx[ctx_id],
                &mut self.slots[idx].0[ctx_id],
                this_range,
                0..MC::SLOT_SIZE
            )?;
        }

        self.host_slots[host_idx].1 = SlotStatus::Busy(id, form);
        self.slots[idx].1 = SlotStatus::Busy(id, form);

        Ok(())
    }

    pub fn copy_from_host_pinned_to_device(
        &mut self, 
        id: PolyId,
        form: PolyForm,
    ) -> GpuResult<()> {
        assert!(self.get_slot_idx(id, form).is_none(),
            "There is already such polynomial: {:?} {:?}", id, form);
        
        let idx = self.free_slot_idx().expect("No free slots");
        let host_idx = self.get_host_slot_idx(id, form)
            .expect(&format!("No such polynomial on host: {:?} {:?}", id, form));

        for ctx_id in 0..MC::NUM_GPUS {
            let start = ctx_id * MC::SLOT_SIZE;
            let this_range = start..(start + MC::SLOT_SIZE);

            self.host_slots[host_idx].0.async_copy_to_device(
                &mut self.ctx[ctx_id],
                &mut self.slots[idx].0[ctx_id],
                this_range,
                0..MC::SLOT_SIZE
            )?;
        }

        self.slots[idx].1 = SlotStatus::Busy(id, form);
        self.host_slots[host_idx].1 = SlotStatus::Busy(id, form);

        Ok(())
    }

    pub fn async_copy_from_device(
        &mut self,
        poly: &mut AsyncVec<Fr>,
        id: PolyId,
        form: PolyForm,
        range: Range<usize>,
    ) -> GpuResult<()> {
        assert_eq!(range.len(), MC::FULL_SLOT_SIZE, "Wrong polynomial size");

        let idx = self.get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));

        for ctx_id in 0..MC::NUM_GPUS {
            let start = range.start + ctx_id * MC::SLOT_SIZE;
            let this_range = start..(start + MC::SLOT_SIZE);

            poly.async_copy_from_device(
                &mut self.ctx[ctx_id],
                &mut self.slots[idx].0[ctx_id],
                this_range,
                0..MC::SLOT_SIZE
            )?;
        }
        self.slots[idx].1 = SlotStatus::Busy(id, form);

        Ok(())
    }

    pub fn copy_from_device_to_free_device(
        &mut self, 
        id: PolyId, 
        new_id: PolyId, 
        form: PolyForm,
    ) -> GpuResult<()> {
        assert!(self.get_slot_idx(new_id, form).is_none(),
            "There is already such polynomial: {:?} {:?}", new_id, form);

        let idx = self.get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));
        let new_idx = self.free_slot_idx().expect("No free slots");

        for ctx_id in 0..MC::NUM_GPUS {
            let (slot_1, slot_2) = get_two_mut(&mut self.slots, idx, new_idx);

            slot_1.0[ctx_id].async_copy_to_device(
                &mut self.ctx[ctx_id],
                &mut slot_2.0[ctx_id],
                0..MC::SLOT_SIZE,
                0..MC::SLOT_SIZE
            )?;
        }
        self.slots[new_idx].1 = SlotStatus::Busy(new_id, form);

        Ok(())
    }

    pub fn copy_shifted_from_device_to_free_device(
        &mut self,
        id: PolyId, 
        new_id: PolyId, 
        form: PolyForm,
        new_first_value: Fr,
    ) -> GpuResult<()> {
        assert!(self.get_slot_idx(new_id, form).is_none(),
            "There is already such polynomial: {:?} {:?}", new_id, form);

        let idx1 = self.get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));
        let idx2 = self.free_slot_idx().expect("No free slots");

        for ctx_id in 0..MC::NUM_GPUS {
            let (slot_1, slot_2) = get_two_mut(&mut self.slots, idx1, idx2);

            slot_1.0[ctx_id].async_copy_to_device(
                &mut self.ctx[ctx_id],
                &mut slot_2.0[ctx_id],
                0..(MC::SLOT_SIZE-1),
                1..MC::SLOT_SIZE
            )?;

            if ctx_id == 0 {
                slot_2.0[ctx_id].async_exec_op(
                    &mut self.ctx[ctx_id],
                    None,
                    Some(new_first_value),
                    0..1,
                    Operation::SetValue
                )?;
            } else {
                slot_1.0[ctx_id - 1].async_copy_to_device(
                    &mut self.ctx[ctx_id],
                    &mut slot_2.0[ctx_id],
                    (MC::SLOT_SIZE-1)..MC::SLOT_SIZE,
                    0..1
                )?;
            }
        }
        self.slots[idx2].1 = SlotStatus::Busy(new_id, form);

        Ok(())
    }

    pub fn copy_leftshifted_from_device_to_free_device(
        &mut self,
        id: PolyId, 
        new_id: PolyId, 
        form: PolyForm,
        new_last_value: Fr,
    ) -> GpuResult<()> {
        assert!(self.get_slot_idx(new_id, form).is_none(),
            "There is already such polynomial: {:?} {:?}", new_id, form);

        let idx1 = self.get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));
        let idx2 = self.free_slot_idx().expect("No free slots");

        for ctx_id in 0..MC::NUM_GPUS {
            let (slot_1, slot_2) = get_two_mut(&mut self.slots, idx1, idx2);

            slot_1.0[ctx_id].async_copy_to_device(
                &mut self.ctx[ctx_id],
                &mut slot_2.0[ctx_id],
                1..MC::SLOT_SIZE,
                0..(MC::SLOT_SIZE-1)
            )?;

            if ctx_id == MC::NUM_GPUS - 1 {
                slot_2.0[ctx_id].async_exec_op(
                    &mut self.ctx[ctx_id],
                    None,
                    Some(new_last_value),
                    (MC::SLOT_SIZE-1)..MC::SLOT_SIZE,
                    Operation::SetValue
                )?;
            } else {
                slot_1.0[ctx_id + 1].async_copy_to_device(
                    &mut self.ctx[ctx_id],
                    &mut slot_2.0[ctx_id],
                    0..1,
                    (MC::SLOT_SIZE-1)..MC::SLOT_SIZE,
                )?;
            }
        }
        self.slots[idx2].1 = SlotStatus::Busy(new_id, form);

        Ok(())
    }

    pub fn copy_from_device_to_device(
        &mut self, 
        id1: PolyId, 
        id2: PolyId, 
        form: PolyForm,
    ) -> GpuResult<()> {
        let idx1 = self.get_slot_idx(id1, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id1, form));
        let idx2 = self.get_slot_idx(id2, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id2, form));

        for ctx_id in 0..MC::NUM_GPUS {
            let (slot_1, slot_2) = get_two_mut(&mut self.slots, idx1, idx2);

            slot_1.0[ctx_id].async_copy_to_device(
                &mut self.ctx[ctx_id],
                &mut slot_2.0[ctx_id],
                0..MC::SLOT_SIZE,
                0..MC::SLOT_SIZE
            )?;
        }

        self.slots[idx1].1 = SlotStatus::Busy(id1, form);
        self.slots[idx2].1 = SlotStatus::Busy(id2, form);

        Ok(())
    }

    // UNSAFE: creates second slot with the same status
    pub unsafe fn clone_slot_on_device(&mut self, id: PolyId, form: PolyForm,) -> GpuResult<()> {
        let idx = self.get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));
        let new_idx = self.free_slot_idx().expect("No free slots");

        for ctx_id in 0..MC::NUM_GPUS {
            let (slot_1, slot_2) = get_two_mut(&mut self.slots, idx, new_idx);

            slot_1.0[ctx_id].async_copy_to_device(
                &mut self.ctx[ctx_id],
                &mut slot_2.0[ctx_id],
                0..MC::SLOT_SIZE,
                0..MC::SLOT_SIZE
            )?;
        }
        self.slots[new_idx].1 = SlotStatus::Busy(id, form);

        Ok(())
    }

    // UNSAFE: operation with pointer
    pub unsafe fn async_copy_from_pointer_and_range(
        &mut self,
        id: PolyId,
        form: PolyForm,
        ptr: *const Fr,
        range: Range<usize>,
    ) -> GpuResult<()> {
        let idx = self.get_slot_idx(id, form)
            .expect(&format!("No such polynomial: {:?} {:?}", id, form));

        for ctx_id in 0..MC::NUM_GPUS {
            let chunk_start = ctx_id * MC::SLOT_SIZE;
            let chunk_end = chunk_start + MC::SLOT_SIZE;

            if range.start >= chunk_end || range.end <= chunk_start {
                continue;
            }

            let ptr = ptr.add(range.start.max(chunk_start) - range.start);
            let start = range.start.max(chunk_start) - chunk_start;
            let end = range.end.min(chunk_end) - chunk_start;

            self.slots[idx].0[ctx_id].async_copy_from_pointer_and_len(
                &mut self.ctx[ctx_id],
                ptr,
                start..end,
                end - start,
            )?;
        }

        Ok(())
    }
}

pub(crate) fn get_two_mut<T>(
    vector: &mut Vec<T>, 
    idx_1: usize, 
    idx_2: usize
) -> (&mut T, &mut T) {
    assert_ne!(idx_1, idx_2);

    if idx_1 < idx_2{
        let (part_1, part_2) = vector.split_at_mut(idx_2);

        (&mut part_1[idx_1], &mut part_2[0])
    } else {
        let (part_1, part_2) = vector.split_at_mut(idx_1);

        (&mut part_2[0], &mut part_1[idx_2])
    }
}

pub(crate) fn get_multi_mut<T>(
    vector: &mut [T], 
    ids: Vec<usize>, 
) -> Vec<&mut T> {
    let length = vector.len();
    let number = ids.len();

    for i in 0..number {
        for j in i+1..number {
            assert_ne!(ids[i], ids[j], "idxs should be different");
        }
        assert!(ids[i] < length, "idx is out of range");
    }

    let mut result: Vec<_> = (0..number).map(|_| None).collect();
    for (i, el) in vector.iter_mut().enumerate() {
        let position = ids.iter().position(|&x| x == i);

        if let Some(j) = position {
            result[j] = Some(el);
        }
    }

    let mut res = vec![];
    for el in result.into_iter() {
        res.push(el.unwrap());
    }

    res
}
