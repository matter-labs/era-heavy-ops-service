use super::*;

pub fn crate_selector_on_manager<
    S: SynthesisMode,
    MC: ManagerConfigs
> (
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    poly_id: PolyId
) -> GpuResult<()> {
    assert!(S::PRODUCE_SETUP);
    assert!(assembly.is_finalized, "assembly should be finalized");
    assert_eq!(manager.slots.len(), MC::NUM_SLOTS, "slots should be allocated");

    let mut device_buffers = crate_buffers_for_bitmasks_with_shift(manager)?;

    match poly_id {
        PolyId::QMainSelector |
        PolyId::QCustomSelector => {
            copy_gate_selector_to_buffers(
                manager,
                assembly,
                &mut device_buffers,
                poly_id,
            )?;
        },
        PolyId::QLookupSelector => {
            create_lookup_selector_in_buffers(
                manager,
                assembly,
                &mut device_buffers,
            )?;
        },
        _ => panic!("Poly Id is not a Selector"),
    }

    let num_input_gates = assembly.num_input_gates;
    let num_all_gates = num_input_gates + assembly.num_aux_gates;
    compute_selectors_from_buffers_to_slots(
        manager,
        &mut device_buffers,
        poly_id,
        num_input_gates,
        num_all_gates
    )?;

    Ok(())
}

fn crate_buffers_for_bitmasks_with_shift<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
) -> GpuResult<Vec<DeviceBuf<u32>>> {
    let mut result = vec![];

    let mut lens_and_devices = vec![(MC::SLOT_SIZE / 32, 0)];
    for ctx_id in 1..MC::NUM_GPUS {
        lens_and_devices.push((1, ctx_id));
        lens_and_devices.push((MC::SLOT_SIZE / 32, ctx_id));
    }

    for (length, ctx_id) in lens_and_devices.into_iter() {
        let buffer = DeviceBuf::async_alloc_in_h2d(
            &manager.ctx[ctx_id],
            length
        )?;
        result.push(buffer);
    }

    Ok(result)
}

fn copy_gate_selector_to_buffers<
    S: SynthesisMode,
    MC: ManagerConfigs,
> (
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    device_buffers: &mut Vec<DeviceBuf<u32>>,
    poly_id: PolyId,
) -> GpuResult<()> {
    let num_inputs = assembly.num_input_gates;
    let offset = num_inputs % 32;
    let mut chunk_0_len = (MC::SLOT_SIZE - num_inputs) / 32;
    if offset != 0 {
        chunk_0_len += 1;
    }

    let host_idx = manager.free_host_slot_idx();

    let bitvec = match poly_id {
        PolyId::QMainSelector => {
            let id = &assembly.sorted_gates[0];
            assembly.aux_gate_density.0.get(id).unwrap()
        },
        PolyId::QCustomSelector => {
            let id = &assembly.sorted_gates[1];
            assembly.aux_gate_density.0.get(id).unwrap()
        },
        _ => panic!("wrong poly id")
    };

    match host_idx {
        None => {
            panic!("There should be a free host slot");
        },
        Some(idx) => unsafe {
            manager.host_slots[idx].0.get_values_mut()?;

            for ctx_id in 0..MC::NUM_GPUS {
                let start = MC::SLOT_SIZE * ctx_id;
                let big_buff = std::slice::from_raw_parts_mut(
                    manager.host_slots[idx].0.as_ptr(start+1..start+1) as *mut u32,
                    MC::SLOT_SIZE / 32
                );
                let extra_buff = std::slice::from_raw_parts_mut(
                    manager.host_slots[idx].0.as_ptr(start..start) as *mut u32,
                    1
                );

                if offset != 0 && ctx_id > 0 {
                    let element = bitvec.storage()[(ctx_id - 1) * MC::SLOT_SIZE / 32 + chunk_0_len - 1];
                    extra_buff[0] = element >> (32 - offset);
                    device_buffers[2 * ctx_id - 1].async_copy_from_pointer_and_len(
                        &mut manager.ctx[ctx_id],
                        &extra_buff[0] as *const u32,
                        0..1,
                        1
                    )?;
                }

                if ctx_id == 0 {
                    let elements = &bitvec.storage()[0..chunk_0_len];
                    big_buff[0..chunk_0_len].copy_from_slice(elements);
                    device_buffers[0].async_copy_from_pointer_and_len(
                        &mut manager.ctx[ctx_id],
                        &big_buff[0] as *const u32,
                        0..chunk_0_len,
                        chunk_0_len
                    )?;
                } else {
                    let start = (ctx_id - 1) * MC::SLOT_SIZE / 32 + chunk_0_len;
                    let elements = &bitvec.storage()[start..(start + MC::SLOT_SIZE / 32)];
                    big_buff.copy_from_slice(elements);
                    device_buffers[2 * ctx_id].async_copy_from_pointer_and_len(
                        &mut manager.ctx[ctx_id],
                        &big_buff[0] as *const u32,
                        0..MC::SLOT_SIZE/32,
                        MC::SLOT_SIZE/32
                    )?;
                }

                manager.host_slots[idx].0.write_event.record(
                    manager.ctx[ctx_id].h2d_stream()
                )?;
            }   
        },
    }

    Ok(())
}

fn create_lookup_selector_in_buffers<
    S: SynthesisMode,
    MC: ManagerConfigs,
> (
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    device_buffers: &mut Vec<DeviceBuf<u32>>,
) -> GpuResult<()> {
    let poly_id = PolyId::QLookupSelector;
    let num_inputs = assembly.num_input_gates;
    let offset = num_inputs % 32;
    let mut chunk_0_len = (MC::SLOT_SIZE - num_inputs) / 32;
    if offset != 0 {
        chunk_0_len += 1;
    }

    let host_idx = manager.free_host_slot_idx();

    match host_idx {
        None => {
            panic!("There should be a free host slot");
        },
        Some(idx) => unsafe {
            manager.host_slots[idx].0.get_values_mut()?;

            for ctx_id in 0..MC::NUM_GPUS {
                // Get host buffers from host slot
                let start = MC::SLOT_SIZE * ctx_id;
                let big_buff = std::slice::from_raw_parts_mut(
                    manager.host_slots[idx].0.as_ptr(start+1..start+1) as *mut u32,
                    MC::SLOT_SIZE / 32
                );
                let extra_buff = std::slice::from_raw_parts_mut(
                    manager.host_slots[idx].0.as_ptr(start..start) as *mut u32,
                    1
                );
                for el in big_buff.iter_mut() {
                    *el = 0;
                }
                extra_buff[0] = 0;

                // Create bitmasks
                for single_application in assembly.tables.iter() {
                    let table_name = single_application.functional_name();
                    let bitvec = assembly.table_selectors.get(&table_name).unwrap();

                    if offset != 0 && ctx_id > 0 {
                        let element = bitvec.storage()[(ctx_id - 1) * MC::SLOT_SIZE / 32 + chunk_0_len - 1];
                        extra_buff[0] = extra_buff[0] | element;
                    }

                    if ctx_id == 0 {
                        let elements = &bitvec.storage()[0..chunk_0_len];
                        for (el1, el2) in big_buff[0..chunk_0_len].iter_mut().zip(elements.iter()) {
                            *el1 = *el1 | *el2;
                        }
                    } else {
                        let start = (ctx_id - 1) * MC::SLOT_SIZE / 32 + chunk_0_len;
                        let elements = &bitvec.storage()[start..(start + MC::SLOT_SIZE / 32)];
                        for (el1, el2) in big_buff.iter_mut().zip(elements.iter()) {
                            *el1 = *el1 | *el2;
                        }
                    }
                }

                // Copy to device
                if offset != 0 && ctx_id > 0 {
                    extra_buff[0] = extra_buff[0] >> (32 - offset);
                    device_buffers[2 * ctx_id - 1].async_copy_from_pointer_and_len(
                        &mut manager.ctx[ctx_id],
                        &extra_buff[0] as *const u32,
                        0..1,
                        1
                    )?;
                }

                if ctx_id == 0 {
                    device_buffers[0].async_copy_from_pointer_and_len(
                        &mut manager.ctx[ctx_id],
                        &big_buff[0] as *const u32,
                        0..chunk_0_len,
                        chunk_0_len
                    )?;
                } else {
                    device_buffers[2 * ctx_id].async_copy_from_pointer_and_len(
                        &mut manager.ctx[ctx_id],
                        &big_buff[0] as *const u32,
                        0..MC::SLOT_SIZE/32,
                        MC::SLOT_SIZE/32
                    )?;
                }

                manager.host_slots[idx].0.write_event.record(
                    manager.ctx[ctx_id].h2d_stream()
                )?;
            }   
        },
    }

    Ok(())
}

fn compute_selectors_from_buffers_to_slots<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    device_buffers: &mut Vec<DeviceBuf<u32>>,
    poly_id: PolyId,
    num_inputs: usize,
    num_all_gates: usize
) -> GpuResult<()> {
    let offset = (num_inputs % 32);

    manager.new_empty_slot(poly_id, PolyForm::Values);
    let slot_idx = manager.get_slot_idx(poly_id, PolyForm::Values).unwrap();

    let result_slot = &mut manager.slots[slot_idx].0[0];
    create_selectors_inner(
        &mut manager.ctx[0],
        result_slot,
        &mut device_buffers[0],
        num_inputs..MC::SLOT_SIZE
    )?;
    device_buffers[0].async_free(&mut manager.ctx[0].exec_stream)?;

    if num_inputs > 0 {
        let value = match poly_id {
            PolyId::QMainSelector => Fr::one(),
            PolyId::QCustomSelector |
            PolyId::QLookupSelector => Fr::zero(),
            _ => panic!("poly id is not a gate selector")
        };

        manager.slots[slot_idx].0[0].async_exec_op(
            &mut manager.ctx[0],
            None,
            Some(value),
            0..num_inputs,
            cuda_bindings::Operation::SetValue
        )?;
    }

    for ctx_id in 1..MC::NUM_GPUS {
        let result_slot = &mut manager.slots[slot_idx].0[ctx_id];

        if num_inputs > 0 {
            create_selectors_inner(
                &mut manager.ctx[ctx_id],
                result_slot,
                &mut device_buffers[2*ctx_id - 1],
                0..offset
            )?;
        }

        create_selectors_inner(
            &mut manager.ctx[ctx_id],
            result_slot,
            &mut device_buffers[2*ctx_id],
            offset..MC::SLOT_SIZE
        )?;

        device_buffers[2*ctx_id].async_free(&mut manager.ctx[ctx_id].exec_stream)?;
        device_buffers[2*ctx_id - 1].async_free(&mut manager.ctx[ctx_id].exec_stream)?;
    }

    match poly_id {
        PolyId::QLookupSelector => {
            manager.set_values_with_range(
                PolyId::QLookupSelector,
                PolyForm::Values,
                Fr::zero(),
                num_all_gates..MC::FULL_SLOT_SIZE
            )?;
        },
        _ => {}
    };

    Ok(())
}

pub fn create_selectors_inner(
    ctx: &mut GpuContext,
    result: &mut DeviceBuf<Fr>,
    bitvec: &mut DeviceBuf<u32>,
    result_range: Range<usize>,
) -> GpuResult<()> {
    assert!(result_range.len() <= bitvec.len() * 256);

    ctx.exec_stream.wait(result.read_event())?;
    ctx.exec_stream.wait(result.write_event())?;
    ctx.exec_stream.wait(bitvec.write_event())?;

    let length: u32 = result_range.len().try_into().unwrap();

    set_device(ctx.device_id())?;
    let res = unsafe { pn_set_values_from_packed_bits(
        result.as_mut_ptr(result_range) as *mut c_void,
        bitvec.as_ptr(0..0) as *const c_void,
        length,
        ctx.exec_stream().inner,
    )};

    if res != 0 {
        return Err(GpuError::SchedulingErr(res));
    }

    result.write_event.record(ctx.exec_stream())?;
    bitvec.read_event.record(ctx.exec_stream())?;

    Ok(())
}

pub fn compute_values_from_bitvec<MC: ManagerConfigs> (
    manager: &mut DeviceMemoryManager<Fr, MC>,
    bitvec: &bit_vec::BitVec,
    poly_id: PolyId
) -> GpuResult<()> {
    let mut device_buffers = crate_buffers_for_bitmasks(manager)?;

    copy_bitvec_to_buffers(manager, &mut device_buffers, bitvec)?;

    compute_values_from_buffers_to_slots(
        manager,
        &mut device_buffers,
        poly_id,
    )?;

    Ok(())
}

fn crate_buffers_for_bitmasks<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
) -> GpuResult<Vec<DeviceBuf<u32>>> {
    let mut result = vec![];

    for ctx_id in 0..MC::NUM_GPUS {
        let buffer = DeviceBuf::async_alloc_in_h2d(
            &manager.ctx[ctx_id],
            MC::SLOT_SIZE / 32,
        )?;
        result.push(buffer);
    }

    Ok(result)
}

fn copy_bitvec_to_buffers<MC: ManagerConfigs> (
    manager: &mut DeviceMemoryManager<Fr, MC>,
    device_buffers: &mut Vec<DeviceBuf<u32>>,
    bitvec: &bit_vec::BitVec
) -> GpuResult<()> {
    assert_eq!(bitvec.len(), MC::FULL_SLOT_SIZE, "bitvec length should be domain size");

    let host_idx = manager.free_host_slot_idx();

    match host_idx {
        None => {
            panic!("There should be a free host slot");
        },
        Some(idx) => unsafe {
            manager.host_slots[idx].0.get_values_mut()?;

            for ctx_id in 0..MC::NUM_GPUS {
                let start = MC::SLOT_SIZE * ctx_id;
                let host_buff = std::slice::from_raw_parts_mut(
                    manager.host_slots[idx].0.as_ptr(start..start) as *mut u32,
                    MC::SLOT_SIZE / 32
                );

                let start = ctx_id * MC::SLOT_SIZE / 32;
                let elements = &bitvec.storage()[start..(start + MC::SLOT_SIZE / 32)];
                host_buff.copy_from_slice(elements);
                device_buffers[ctx_id].async_copy_from_pointer_and_len(
                    &mut manager.ctx[ctx_id],
                    &host_buff[0] as *const u32,
                    0..MC::SLOT_SIZE/32,
                    MC::SLOT_SIZE/32
                )?;

                manager.host_slots[idx].0.write_event.record(
                    manager.ctx[ctx_id].h2d_stream()
                )?;
            }   
        },
    }

    Ok(())
}

fn compute_values_from_buffers_to_slots<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    device_buffers: &mut Vec<DeviceBuf<u32>>,
    poly_id: PolyId,
) -> GpuResult<()> {
    manager.new_empty_slot(poly_id, PolyForm::Values);
    let slot_idx = manager.get_slot_idx(poly_id, PolyForm::Values).unwrap();

    for ctx_id in 0..MC::NUM_GPUS {
        let result_slot = &mut manager.slots[slot_idx].0[ctx_id];

        create_selectors_inner(
            &mut manager.ctx[ctx_id],
            result_slot,
            &mut device_buffers[ctx_id],
            0..MC::SLOT_SIZE
        )?;

        device_buffers[ctx_id].async_free(&mut manager.ctx[ctx_id].exec_stream)?;
    }

    Ok(())
}
