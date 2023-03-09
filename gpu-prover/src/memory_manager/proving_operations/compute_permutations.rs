use super::*;

pub fn compute_permutation_polynomials<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
) -> GpuResult<()> {
    assert!(S::PRODUCE_SETUP);
    // assert_eq!(<DefaultAssembly as PlonkConstraintSystemParams>::STATE_WIDTH, 4);
    assert!(assembly.is_finalized, "assembly should be finalized");
    assert_eq!(
        manager.slots.len(),
        MC::NUM_SLOTS,
        "slots should be allocated"
    );
    assert_eq!(
        manager.polynomials_on_device().len(),
        0,
        "manager should not contain any polynomials"
    );

    let ctx_id = 0;
    let device_id = manager.ctx[ctx_id].device_id();
    wait_events_before_computing_pernutations(manager, device_id)?;

    let (mut permutations, mut variables, mut non_residues) =
        create_buffers_for_computing_assigments(manager, device_id)?;

    copy_variables(manager, assembly, worker, &mut variables, device_id)?;

    compute_permutation_polynomials_on_device(
        &mut manager.ctx[ctx_id],
        &mut variables,
        &mut non_residues,
        &mut permutations,
        MC::FULL_SLOT_SIZE_LOG,
    )?;

    final_copying_to_slots(manager, &mut permutations, device_id)?;

    write_events_after_computing_pernutations(manager, device_id)?;

    Ok(())
}

fn create_buffers_for_computing_assigments<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    device_id: usize,
) -> GpuResult<(DeviceBuf<Fr>, DeviceBuf<u32>, DeviceBuf<Fr>)> {
    // Create slots for result
    for i in 0..4 {
        manager.new_empty_slot(PolyId::Sigma(i), PolyForm::Values);
    }

    let ctx_id = manager.get_ctx_id_by_device_id(device_id);
    let offset = manager.polynomials_on_device().len();

    // Create buffer for result
    let permutations = DeviceBuf {
        ptr: manager.slots[offset].0[ctx_id].as_mut_ptr(0..0),
        len: 4 * MC::FULL_SLOT_SIZE,
        device_id,

        is_static_mem: true,
        is_freed: true,

        read_event: Event::new(),
        write_event: Event::new(),
    };

    // Create buffer for variables
    let variables = DeviceBuf {
        ptr: manager.slots[offset + 4 * MC::NUM_GPUS].0[ctx_id].as_mut_ptr(0..0) as *mut u32,
        len: 4 * MC::FULL_SLOT_SIZE,
        device_id,

        is_static_mem: true,
        is_freed: true,

        read_event: Event::new(),
        write_event: Event::new(),
    };

    // Create buffer non_residues
    let mut non_residues = DeviceBuf {
        ptr: manager.slots[offset + 4 * MC::NUM_GPUS + 4].0[ctx_id].as_mut_ptr(0..0),
        len: 4,
        device_id,

        is_static_mem: true,
        is_freed: true,

        read_event: Event::new(),
        write_event: Event::new(),
    };

    // Compute non_residues
    use bellman::plonk::better_cs::generator::make_non_residues;
    let num_non_residues = 4;
    let mut host_non_residues = AsyncVec::allocate_new(num_non_residues);

    let mut host_buff = host_non_residues.get_values_mut()?;
    host_buff[0] = Fr::one();
    host_buff[1..].copy_from_slice(&make_non_residues::<Fr>(num_non_residues - 1));

    non_residues.async_copy_from_host(
        &mut manager.ctx[ctx_id],
        &mut host_non_residues,
        0..num_non_residues,
        0..num_non_residues,
    )?;

    Ok((permutations, variables, non_residues))
}

pub(crate) fn copy_variables<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
    variables: &mut DeviceBuf<u32>,
    device_id: usize,
) -> GpuResult<()> {
    let ctx_id = manager.get_ctx_id_by_device_id(device_id);
    match manager.free_host_slot_idx() {
        Some(idx) => {
            let host_ptr =
                &mut manager.host_slots[idx].0.get_values_mut()?[0] as *mut Fr as *mut u32;

            // SAFETY: host buf lives long enought
            // and its size is 2 GB, while we need 1 GB
            let mut host_buff =
                unsafe { std::slice::from_raw_parts_mut(host_ptr, 4 * MC::FULL_SLOT_SIZE) };

            copy_variables_to_buffer(assembly, worker, host_buff);

            unsafe {
                variables.async_copy_from_pointer_and_len(
                    &mut manager.ctx[ctx_id],
                    host_ptr,
                    0..4 * MC::FULL_SLOT_SIZE,
                    4 * MC::FULL_SLOT_SIZE,
                )?;
            }

            manager.host_slots[idx]
                .0
                .write_event
                .record(manager.ctx[ctx_id].h2d_stream())?;
        }
        None => {
            dbg!("allocating additional host slots");
            let mut host_variables = AsyncVec::allocate_new(4 * MC::FULL_SLOT_SIZE);
            let mut host_buff = host_variables.get_values_mut()?;
            copy_variables_to_buffer(assembly, worker, host_buff);

            variables.async_copy_from_host(
                &mut manager.ctx[ctx_id],
                &mut host_variables,
                0..4 * MC::FULL_SLOT_SIZE,
                0..4 * MC::FULL_SLOT_SIZE,
            )?;
        }
    }

    Ok(())
}

fn copy_variables_to_buffer<S: SynthesisMode>(
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
    buffer: &mut [u32],
) {
    let num_input_gates = assembly.num_input_gates;
    let domain_size = assembly.n() + 1;
    assert_eq!(4 * domain_size, buffer.len());

    for poly_idx in 0..4 {
        let poly_id = PolyIdentifier::VariablesPolynomial(poly_idx);

        if num_input_gates > 0 {
            let variables = assembly
                .inputs_storage
                .state_map
                .get(&poly_id)
                .expect("get poly from input storage");
            assert_eq!(variables.len(), num_input_gates);

            // find corresponding starting cell
            let range_start = poly_idx * domain_size;
            for (src, dst) in variables.iter().zip(buffer[range_start..].iter_mut()) {
                *dst = transform_variable(src, num_input_gates)
            }
        }

        let aux_vars = assembly
            .aux_storage
            .state_map
            .get(&poly_id)
            .expect("get poly from aux storage");

        worker.scope(aux_vars.len(), |scope, chunk_size| {
            // don't forget that first few gates are input gates
            let range_start = poly_idx * domain_size + num_input_gates;
            for (src_chunk, dst_chunk) in aux_vars
                .chunks(chunk_size)
                .zip(buffer[range_start..].chunks_mut(chunk_size))
            {
                scope.spawn(|_| {
                    for (src, dst) in src_chunk.iter().zip(dst_chunk.iter_mut()) {
                        *dst = transform_variable(src, num_input_gates)
                    }
                });
            }
        });

        let start = poly_idx * domain_size + num_input_gates + assembly.num_aux_gates;
        let end = poly_idx * domain_size + domain_size;
        fill_with(worker, &mut buffer[start..end], 0);
    }
}

pub fn compute_permutation_polynomials_on_device(
    ctx: &mut GpuContext,
    variables: &mut DeviceBuf<u32>,
    non_residues: &mut DeviceBuf<Fr>,
    permutations: &mut DeviceBuf<Fr>,
    count_log: usize,
) -> Result<(), GpuError> {
    let stream = &mut ctx.exec_stream;

    stream.wait(variables.write_event())?;
    stream.wait(non_residues.write_event())?;

    let indexes = variables.as_mut_ptr(0..0);
    let scalars = non_residues.as_mut_ptr(0..0) as *mut c_void;
    let target = permutations.as_mut_ptr(0..0) as *mut c_void;

    set_device(stream.device_id())?;
    let cfg = generate_permutation_polynomials_configuration {
        mem_pool: ctx.mem_pool.expect("mem pool should be allocated"),
        stream: stream.inner,
        indexes,
        scalars,
        target,
        columns_count: 4,
        log_rows_count: count_log as u32,
    };

    unsafe {
        let result = pn_generate_permutation_polynomials(cfg);
        if result != 0 {
            return Err(GpuError::PermutationPolysErr(result));
        }
    }

    permutations.write_event.record(&stream)?;

    Ok(())
}

fn final_copying_to_slots<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    permutations: &mut DeviceBuf<Fr>,
    exec_device_id: usize,
) -> GpuResult<()> {
    let exec_ctx_id = manager.get_ctx_id_by_device_id(exec_device_id);

    for poly_idx in 0..4 {
        for ctx_id in 0..MC::NUM_GPUS {
            let slot_idx = manager
                .get_slot_idx(PolyId::Sigma(poly_idx), PolyForm::Values)
                .unwrap();
            let slot = &mut manager.slots[slot_idx].0[ctx_id];

            let start = poly_idx * MC::FULL_SLOT_SIZE + ctx_id * MC::SLOT_SIZE;
            let end = start + MC::SLOT_SIZE;

            slot.async_copy_from_device(
                &mut manager.ctx[exec_ctx_id],
                permutations,
                0..MC::SLOT_SIZE,
                start..end,
            )?;
        }
    }

    Ok(())
}

fn wait_events_before_computing_pernutations<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    device_id: usize,
) -> GpuResult<()> {
    let ctx_id = manager.get_ctx_id_by_device_id(device_id);

    for slot in manager.slots.iter_mut() {
        manager.ctx[ctx_id]
            .h2d_stream
            .wait(slot.0[ctx_id].read_event())?;
        manager.ctx[ctx_id]
            .h2d_stream
            .wait(slot.0[ctx_id].write_event())?;
    }

    Ok(())
}

fn write_events_after_computing_pernutations<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    device_id: usize,
) -> GpuResult<()> {
    let ctx_id = manager.get_ctx_id_by_device_id(device_id);

    for slot in manager.slots.iter_mut() {
        slot.0[ctx_id]
            .write_event
            .record(&manager.ctx[ctx_id].exec_stream());
    }

    Ok(())
}
