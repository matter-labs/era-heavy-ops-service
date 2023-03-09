use super::*;

pub fn compute_assigments_and_permutations<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
) -> GpuResult<()> {
    // assert!(S::PRODUCE_SETUP);
    assert!(S::PRODUCE_WITNESS);
    // assert_eq!(<DefaultAssembly as PlonkConstraintSystemParams>::STATE_WIDTH, 4);
    assert!(assembly.is_finalized, "assembly should be finalized");
    assert!(MC::NUM_GPUS <= 2, "only this configuration are supported");
    assert_eq!(
        manager.slots.len(),
        MC::NUM_SLOTS,
        "slots should be allocated"
    );
    assert_eq!(
        manager.polynomials_on_device().len(),
        0,
        "manager should not contain any polynomial"
    );

    let device_id_0 = manager.ctx[0].device_id();
    let ctx_id_1 = 1 % MC::NUM_GPUS;
    let device_id_1 = manager.ctx[ctx_id_1].device_id();

    wait_events_before_computations(manager)?;

    let num_all_assignments = assembly.input_assingments.len() + assembly.aux_assingments.len();
    let (mut state_polys, mut permutations, mut variables, mut all_assignments, mut non_residues) =
        create_buffers_for_computing_assigments_and_permutations(manager, num_all_assignments)?;

    set_initial_values(manager, &mut all_assignments, &mut non_residues)?;
    copy_variables(manager, assembly, worker, &mut variables, device_id_1)?;

    compute_permutation_polynomials_on_device(
        &mut manager.ctx[ctx_id_1],
        &mut variables,
        &mut non_residues,
        &mut permutations,
        MC::FULL_SLOT_SIZE_LOG,
    )?;

    copy_input_assigments_to_state_polys(manager, assembly, &mut state_polys)?;
    copy_assigments_to_assigments_poly(manager, assembly, &mut all_assignments)?;

    split_variables_and_schedule_computation(
        manager,
        &mut state_polys,
        variables,
        &mut all_assignments,
        assembly.num_input_gates,
    )?;

    final_copying_to_slots(manager, &mut state_polys, &mut permutations)?;
    write_events_after_computations(manager)?;

    Ok(())
}

fn create_buffers_for_computing_assigments_and_permutations<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assignments_len: usize,
) -> GpuResult<(
    Vec<DeviceBuf<Fr>>,
    DeviceBuf<Fr>,
    DeviceBuf<u32>,
    DeviceBuf<Fr>,
    DeviceBuf<Fr>,
)> {
    // Create slots for result
    for (i, poly_id) in [PolyId::A, PolyId::B, PolyId::C, PolyId::D]
        .into_iter()
        .enumerate()
    {
        manager.new_empty_slot(poly_id, PolyForm::Values);
        manager.new_empty_slot(PolyId::Sigma(i), PolyForm::Values);
    }

    let device_id_0 = manager.ctx[0].device_id();
    let device_id_1 = manager.ctx[1 % MC::NUM_GPUS].device_id();
    let mut offset = manager.polynomials_on_device().len();

    // Create buffers for result of state values
    let state_polys: Vec<_> = (0..4)
        .map(|i| DeviceBuf {
            ptr: manager.slots[offset + i * MC::NUM_GPUS].0[0].as_mut_ptr(0..0),
            len: MC::FULL_SLOT_SIZE,
            device_id: device_id_0,

            is_static_mem: true,
            is_freed: true,

            read_event: Event::new(),
            write_event: Event::new(),
        })
        .collect();

    if MC::NUM_GPUS == 1 {
        offset += 4 * MC::NUM_GPUS;
    }

    // Create buffer for result of permutations
    let permutations = DeviceBuf {
        ptr: manager.slots[offset].0[1 % MC::NUM_GPUS].as_mut_ptr(0..0),
        len: 4 * MC::FULL_SLOT_SIZE,
        device_id: device_id_1,

        is_static_mem: true,
        is_freed: true,

        read_event: Event::new(),
        write_event: Event::new(),
    };

    // Create buffer for variables
    let variables = DeviceBuf {
        ptr: manager.slots[offset + 4 * MC::NUM_GPUS].0[1 % MC::NUM_GPUS].as_mut_ptr(0..0)
            as *mut u32,
        len: 4 * MC::FULL_SLOT_SIZE,
        device_id: device_id_1,

        is_static_mem: true,
        is_freed: true,

        read_event: Event::new(),
        write_event: Event::new(),
    };

    // Create buffer non_residues
    let mut non_residues = DeviceBuf {
        ptr: manager.slots[offset + 6 * MC::NUM_GPUS].0[1 % MC::NUM_GPUS].as_mut_ptr(0..0),
        len: 4,
        device_id: device_id_1,

        is_static_mem: true,
        is_freed: true,

        read_event: Event::new(),
        write_event: Event::new(),
    };

    if MC::NUM_GPUS == 1 {
        offset += 7 * MC::NUM_GPUS;
    } else {
        offset += 4 * MC::NUM_GPUS;
    }

    // Create buffer for assigments with an offset
    let mut all_assignments = DeviceBuf {
        ptr: manager.slots[offset].0[0].as_mut_ptr(0..0),
        len: assignments_len + 1,
        device_id: device_id_0,

        is_static_mem: true,
        is_freed: true,

        read_event: Event::new(),
        write_event: Event::new(),
    };

    Ok((
        state_polys,
        permutations,
        variables,
        all_assignments,
        non_residues,
    ))
}

fn set_initial_values<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    all_assignments: &mut DeviceBuf<Fr>,
    non_residues: &mut DeviceBuf<Fr>,
) -> GpuResult<()> {
    let ctx_id_0 = 0;
    let ctx_id_1 = 1 % MC::NUM_GPUS;

    // Set the first value to zero
    all_assignments.async_exec_op(
        &mut manager.ctx[ctx_id_0],
        None,
        Some(Fr::zero()),
        0..1,
        crate::cuda_bindings::Operation::SetValue,
    )?;

    // Compute non_residues
    use bellman::plonk::better_cs::generator::make_non_residues;
    let num_non_residues = 4;
    let mut host_non_residues = AsyncVec::allocate_new(num_non_residues);

    let mut host_buff = host_non_residues.get_values_mut()?;
    host_buff[0] = Fr::one();
    host_buff[1..].copy_from_slice(&make_non_residues::<Fr>(num_non_residues - 1));

    non_residues.async_copy_from_host(
        &mut manager.ctx[ctx_id_1],
        &mut host_non_residues,
        0..num_non_residues,
        0..num_non_residues,
    )?;

    Ok(())
}

fn final_copying_to_slots<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    state_polys: &mut Vec<DeviceBuf<Fr>>,
    permutations: &mut DeviceBuf<Fr>,
) -> GpuResult<()> {
    let ctx_id_0 = 0;
    let ctx_id_1 = 1 % MC::NUM_GPUS;

    for poly_idx in 0..4 {
        for ctx_id in 0..MC::NUM_GPUS {
            let slot_idx = manager
                .get_slot_idx(PolyId::Sigma(poly_idx), PolyForm::Values)
                .unwrap();
            let slot = &mut manager.slots[slot_idx].0[ctx_id];

            let start = poly_idx * MC::FULL_SLOT_SIZE + ctx_id * MC::SLOT_SIZE;
            let end = start + MC::SLOT_SIZE;

            slot.async_copy_from_device(
                &mut manager.ctx[ctx_id_1],
                permutations,
                0..MC::SLOT_SIZE,
                start..end,
            )?;
        }
    }

    let state_ids = [PolyId::A, PolyId::B, PolyId::C, PolyId::D];
    for poly_idx in 0..4 {
        for ctx_id in 0..MC::NUM_GPUS {
            let slot_idx = manager
                .get_slot_idx(state_ids[poly_idx], PolyForm::Values)
                .unwrap();
            let slot = &mut manager.slots[slot_idx].0[ctx_id];

            let start = ctx_id * MC::SLOT_SIZE;
            let end = start + MC::SLOT_SIZE;

            slot.async_copy_from_device(
                &mut manager.ctx[ctx_id_0],
                &mut state_polys[poly_idx],
                0..MC::SLOT_SIZE,
                start..end,
            )?;
        }
    }

    Ok(())
}

fn wait_events_before_computations<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
) -> GpuResult<()> {
    for slot in manager.slots.iter_mut().skip(8) {
        for ctx_id in 0..MC::NUM_GPUS {
            manager.ctx[ctx_id]
                .h2d_stream
                .wait(slot.0[ctx_id].write_event())?;
            manager.ctx[ctx_id]
                .h2d_stream
                .wait(slot.0[ctx_id].read_event())?;
        }
    }

    Ok(())
}

fn write_events_after_computations<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
) -> GpuResult<()> {
    for slot in manager.slots.iter_mut().skip(8) {
        for ctx_id in 0..MC::NUM_GPUS {
            slot.0[ctx_id]
                .write_event
                .record(&manager.ctx[0].exec_stream());

            if MC::NUM_GPUS > 1 {
                slot.0[ctx_id]
                    .write_event
                    .record(&manager.ctx[1].exec_stream());
            }
        }
    }

    Ok(())
}
