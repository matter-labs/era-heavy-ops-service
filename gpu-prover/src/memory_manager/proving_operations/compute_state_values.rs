use super::*;

pub fn assign_cs_variables<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
) -> GpuResult<()> {
    assert!(S::PRODUCE_WITNESS);
    // assert_eq!(<DefaultAssembly<S> as PlonkConstraintSystemParams>::STATE_WIDTH, 4);
    assert!(assembly.is_finalized, "assembly should be finalized");
    assert_eq!(
        manager.slots.len(),
        MC::NUM_SLOTS,
        "slots should be allocated"
    );
    assert_eq!(
        manager.polynomials_on_device().len(),
        0,
        "manager should be empty"
    );

    wait_events_before_computing_assigments(manager)?;

    let num_all_assignments = assembly.input_assingments.len() + assembly.aux_assingments.len();
    let (mut state_polys, mut variables, mut all_assignments) =
        create_buffers_for_computing_assigments(manager, num_all_assignments)?;

    let device_id = manager.ctx[0].device_id();
    copy_variables(manager, assembly, worker, &mut variables, device_id)?;

    copy_input_assigments_to_state_polys(manager, assembly, &mut state_polys)?;

    copy_assigments_to_assigments_poly(manager, assembly, &mut all_assignments)?;

    split_variables_and_schedule_computation(
        manager,
        &mut state_polys,
        variables,
        &mut all_assignments,
        assembly.num_input_gates,
    )?;

    final_copying_to_slots(manager, &mut state_polys)?;

    write_events_after_computing_assigments(manager)?;

    Ok(())
}

fn create_buffers_for_computing_assigments<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assignments_len: usize,
) -> GpuResult<(Vec<DeviceBuf<Fr>>, DeviceBuf<u32>, DeviceBuf<Fr>)> {
    // Check if enought space
    let offset = 4;
    let device_id = manager.ctx[0].device_id();

    assert!(MC::NUM_SLOTS >= offset + 4 + 4 * MC::NUM_GPUS);
    let free_slots_for_assigments = MC::NUM_SLOTS - offset - 4 * MC::NUM_GPUS - 4;
    assert!(free_slots_for_assigments * MC::SLOT_SIZE >= assignments_len);

    // Create slots for result
    for id in [PolyId::A, PolyId::B, PolyId::C, PolyId::D].into_iter() {
        manager.new_empty_slot(id, PolyForm::Values);
    }

    // Create buffers for result
    let state_polys: Vec<_> = (0..4)
        .map(|i| DeviceBuf {
            ptr: manager.slots[offset + i * MC::NUM_GPUS].0[0].as_mut_ptr(0..0),
            len: MC::FULL_SLOT_SIZE,
            device_id,

            is_static_mem: true,
            is_freed: true,

            read_event: Event::new(),
            write_event: Event::new(),
        })
        .collect();

    // Create buffer for variables
    let variables = DeviceBuf {
        ptr: manager.slots[offset + 4 * MC::NUM_GPUS].0[0].as_mut_ptr(0..0) as *mut u32,
        len: 4 * MC::FULL_SLOT_SIZE,
        device_id,

        is_static_mem: true,
        is_freed: true,

        read_event: Event::new(),
        write_event: Event::new(),
    };

    // Create buffer for assigments with an offset
    // and set the first value to zero
    let mut all_assignments = DeviceBuf {
        ptr: manager.slots[offset + 4 * MC::NUM_GPUS + 4].0[0].as_mut_ptr(0..0),
        len: assignments_len + 1,
        device_id,

        is_static_mem: true,
        is_freed: true,

        read_event: Event::new(),
        write_event: Event::new(),
    };

    all_assignments.async_exec_op(
        &mut manager.ctx[0],
        None,
        Some(Fr::zero()),
        0..1,
        crate::cuda_bindings::Operation::SetValue,
    )?;

    Ok((state_polys, variables, all_assignments))
}

pub(crate) fn copy_input_assigments_to_state_polys<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    state_polynomials: &mut Vec<DeviceBuf<Fr>>,
) -> GpuResult<()> {
    let num_input_assignments = assembly.input_assingments.len();
    if num_input_assignments < 1 {
        return Ok(());
    }
    let input_assignments = assembly.input_assingments.as_ptr();

    unsafe {
        state_polynomials[0].async_copy_from_pointer_and_len(
            &mut manager.ctx[0],
            input_assignments,
            0..num_input_assignments,
            num_input_assignments,
        )?;
    }

    for poly in state_polynomials.iter_mut().skip(1) {
        poly.async_exec_op(
            &mut manager.ctx[0],
            None,
            Some(Fr::zero()),
            0..num_input_assignments,
            crate::cuda_bindings::Operation::SetValue,
        )?;
    }

    Ok(())
}

pub(crate) fn copy_assigments_to_assigments_poly<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    all_assignments: &mut DeviceBuf<Fr>,
) -> GpuResult<()> {
    let num_input_assignments = assembly.input_assingments.len();
    let input_assignments = assembly.input_assingments.as_ptr();

    if num_input_assignments > 0 {
        unsafe {
            all_assignments.async_copy_from_pointer_and_len(
                &mut manager.ctx[0],
                input_assignments,
                1..(num_input_assignments + 1),
                num_input_assignments,
            )?
        }
    }

    let num_aux_assignments = assembly.aux_assingments.len();
    let aux_assignments = assembly.aux_assingments.as_ptr();
    let start = num_input_assignments + 1;

    unsafe {
        all_assignments.async_copy_from_pointer_and_len(
            &mut manager.ctx[0],
            aux_assignments,
            start..(num_aux_assignments + start),
            num_aux_assignments,
        )?
    }

    Ok(())
}

pub(crate) fn split_variables_and_schedule_computation<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    state_polys: &mut Vec<DeviceBuf<Fr>>,
    variables: DeviceBuf<u32>,
    all_assignments: &mut DeviceBuf<Fr>,
    num_input_gates: usize,
) -> GpuResult<()> {
    let mut variables = variables.split_any_buf(4);

    for poly_idx in 0..4 {
        variables[poly_idx].ptr = unsafe { variables[poly_idx].ptr.add(num_input_gates) };
        variables[poly_idx].len -= num_input_gates;

        assign_variables(
            manager,
            &mut state_polys[poly_idx],
            &mut variables[poly_idx],
            all_assignments,
            num_input_gates,
        )?;
    }

    Ok(())
}

fn copy_single_variables<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    worker: &Worker,
    host_variables: &Vec<Variable>,
    variables: &mut DeviceBuf<u32>,
    host_buffer: &mut AsyncVec<u32>,
    num_input_assignments: usize,
) -> GpuResult<()> {
    let buffer = host_buffer.get_values_mut().expect("get buffer values");
    worker.scope(host_variables.len(), |scope, chunk_size| {
        for (src, dst) in host_variables
            .chunks(chunk_size)
            .zip(buffer.chunks_mut(chunk_size))
        {
            scope.spawn(|_| {
                for (el, out) in src.iter().zip(dst.iter_mut()) {
                    *out = transform_variable(el, num_input_assignments);
                }
            });
        }
    });

    variables.async_copy_from_host(
        &mut manager.ctx[0],
        host_buffer,
        0..MC::FULL_SLOT_SIZE,
        0..MC::FULL_SLOT_SIZE,
    )?;

    Ok(())
}

pub fn assign_variables<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    state_polys: &mut DeviceBuf<Fr>,
    variables: &mut DeviceBuf<u32>,
    assigments: &mut DeviceBuf<Fr>,
    num_input_gates: usize,
) -> GpuResult<()> {
    let device_id = manager.ctx[0].device_id();
    let stream = &mut manager.ctx[0].exec_stream;

    stream.wait(state_polys.write_event())?;
    stream.wait(variables.write_event())?;
    stream.wait(assigments.write_event())?;

    let length = variables.len();
    let result = state_polys.as_mut_ptr(num_input_gates..length) as *mut c_void;
    let variables = variables.as_ptr(0..length);
    let assigments = assigments.as_ptr(0..0) as *const c_void;

    set_device(device_id)?;
    unsafe {
        let result = ff_select(assigments, result, variables, length as u32, stream.inner);
        if result != 0 {
            return Err(GpuError::FFAssignErr(result));
        };
    }
    state_polys.write_event.record(stream)?;

    Ok(())
}

fn final_copying_to_slots<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    state_polys: &mut Vec<DeviceBuf<Fr>>,
) -> GpuResult<()> {
    for poly_idx in 0..4 {
        for ctx_id in 0..MC::NUM_GPUS {
            let slot = &mut manager.slots[poly_idx].0[ctx_id];
            let start = ctx_id * MC::SLOT_SIZE;
            let end = start + MC::SLOT_SIZE;

            slot.async_copy_from_device(
                &mut manager.ctx[0],
                &mut state_polys[poly_idx],
                0..MC::SLOT_SIZE,
                start..end,
            )?;
        }
    }

    Ok(())
}

fn wait_events_before_computing_assigments<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
) -> GpuResult<()> {
    for slot in manager.slots.iter_mut().skip(4) {
        manager.ctx[0].h2d_stream.wait(slot.0[0].read_event())?;
        manager.ctx[0].h2d_stream.wait(slot.0[0].write_event())?;
    }

    Ok(())
}

fn write_events_after_computing_assigments<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
) -> GpuResult<()> {
    for slot in manager.slots.iter_mut().skip(4) {
        slot.0[0].write_event.record(&manager.ctx[0].exec_stream());
    }

    Ok(())
}
