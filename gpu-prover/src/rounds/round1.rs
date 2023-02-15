use super::*;

pub fn round1<    
    S: SynthesisMode,
    C: Circuit<Bn256>, 
    T: Transcript<Fr>,
    MC: ManagerConfigs,
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
    proof: &mut Proof<Bn256, C>,
    transcript: &mut T,
    setup: &mut AsyncSetup,
    msm_handles_round1: &mut Vec<MSMHandle>,
) -> Result<(), ProvingError> {

    schedule_state_commitments::<S, _>(manager, worker, setup, msm_handles_round1)?;
    
    // SCHEDULE COPY OPS FOR NEXT ROUND

    upload_lookup_selector_and_table_type(manager, assembly, worker, setup)?;
    upload_t_poly_parts(manager, assembly, worker, setup)?;

    schedule_auxiliary_operations::<S, _>(manager, setup)?;

    Ok(())
}

fn schedule_state_commitments<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    worker: &Worker,
    setup: &mut AsyncSetup,
    msm_handles_round1: &mut Vec<MSMHandle>,
) -> Result<(), ProvingError> {
    let state_polys_ids = [PolyId::A, PolyId::B, PolyId::C, PolyId::D,];

    for id in state_polys_ids.into_iter() {
        manager.multigpu_ifft_to_free_slot(id, false)?;
        let handle = manager.msm(id)?;
        msm_handles_round1.push(handle);    
    }

    Ok(())
}

fn upload_t_poly_parts<    
    S: SynthesisMode,
    MC: ManagerConfigs,
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
    setup: &mut AsyncSetup,
) -> Result<(), ProvingError> {

    if !S::PRODUCE_SETUP {
        let copy_start = MC::FULL_SLOT_SIZE - setup.lookup_tables_values[0].len() - 1;
        let copy_end = copy_start + setup.lookup_tables_values[0].len();

        for i in 0..3 {
            manager.new_empty_slot(PolyId::Col(i), PolyForm::Values);

            unsafe {
                manager.async_copy_from_pointer_and_range(
                    PolyId::Col(i),
                    PolyForm::Values,
                    &setup.lookup_tables_values[i].get_values().unwrap()[0] as *const Fr,
                    copy_start..copy_end
                )?;
            }

            if copy_start != 0 {
                manager.set_values_with_range(
                    PolyId::Col(i),
                    PolyForm::Values,
                    Fr::zero(),
                    0..copy_start
                )?;
            }
            
            manager.set_values_with_range(
                PolyId::Col(i),
                PolyForm::Values,
                Fr::zero(),
                copy_end..MC::FULL_SLOT_SIZE
            )?;
        }

        manager.new_empty_slot(PolyId::TableType, PolyForm::Values);

        unsafe {
            manager.async_copy_from_pointer_and_range(
                PolyId::TableType,
                PolyForm::Values,
                &setup.lookup_tables_values[3].get_values().unwrap()[0] as *const Fr,
                copy_start..copy_end
            )?;
        }

        if copy_start != 0 {
            manager.set_values_with_range(
                PolyId::TableType,
                PolyForm::Values,
                Fr::zero(),
                0..copy_start
            )?;
        }
        
        manager.set_values_with_range(
            PolyId::TableType,
            PolyForm::Values,
            Fr::zero(),
            copy_end..MC::FULL_SLOT_SIZE
        )?;
    } else {
        let poly_id = [PolyId::Col(0), PolyId::Col(1), PolyId::Col(2), PolyId::TableType];

        #[cfg(feature = "allocator")]
        let t_poly_ends = assembly.calculate_t_polynomial_values_for_single_application_tables().unwrap();
        #[cfg(not(feature = "allocator"))]
        let t_poly_ends = assembly.calculate_t_polynomial_values_for_single_application_tables().unwrap();
        
        for (i, t_poly) in t_poly_ends.into_iter().enumerate() {
            let copy_start = MC::FULL_SLOT_SIZE - t_poly.len() - 1;
            let mut t_col = manager.get_free_host_slot_values_mut(poly_id[i], PolyForm::Values)?;
            fill_with_zeros(worker, &mut t_col[..copy_start]);
            async_copy(worker, &mut t_col[copy_start..(MC::FULL_SLOT_SIZE-1)], &t_poly);

            t_col[MC::FULL_SLOT_SIZE-1] = Fr::zero();
            manager.copy_from_host_pinned_to_device(poly_id[i], PolyForm::Values)?;

            if i > 0 {
                manager.free_host_slot(poly_id[i-1], PolyForm::Values);
            }
        }
        manager.free_host_slot(PolyId::TableType, PolyForm::Values);
    }

    Ok(())
}

fn upload_lookup_selector_and_table_type<    
    S: SynthesisMode,
    MC: ManagerConfigs
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
    setup: &mut AsyncSetup,
) -> Result<(), ProvingError> {

    if !S::PRODUCE_SETUP {
        compute_values_from_bitvec(manager, &setup.lookup_selector_bitvec, PolyId::QLookupSelector);

        manager.async_copy_to_device(
            &mut setup.lookup_table_type_monomial, 
            PolyId::QTableType, 
            PolyForm::Monomial, 
            0..MC::FULL_SLOT_SIZE,
        )?;
    } else {
        get_lookup_selector_from_assembly(manager, assembly, worker)?;
        get_table_type_from_assembly(manager, assembly)?;
    }

    Ok(())
}

fn schedule_auxiliary_operations<
    S: SynthesisMode,
    MC: ManagerConfigs,
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    setup: &mut AsyncSetup,
) -> Result<(), ProvingError> {
    if !S::PRODUCE_SETUP {
        manager.multigpu_fft(PolyId::QTableType, false)?;
    }

    Ok(())
}

pub fn get_lookup_selector_from_assembly<    
    S: SynthesisMode,
    MC: ManagerConfigs
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
) -> Result<(), ProvingError> {
    crate_selector_on_manager(manager, assembly, PolyId::QLookupSelector)?;

    Ok(())
}

pub fn get_table_type_from_assembly<    
    S: SynthesisMode,
    MC: ManagerConfigs
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
) -> Result<(), ProvingError> {
    let num_input_gates = assembly.num_input_gates;
    let num_all_gates = num_input_gates + assembly.num_aux_gates;

    manager.new_empty_slot(PolyId::QTableType, PolyForm::Values);

    unsafe {
        manager.async_copy_from_pointer_and_range(
            PolyId::QTableType,
            PolyForm::Values,
            &assembly.table_ids_poly[0] as *const Fr,
            num_input_gates..num_all_gates
        )?;
    }

    if num_input_gates > 0 {
        manager.set_values_with_range(
            PolyId::QTableType,
            PolyForm::Values,
            Fr::zero(),
            0..num_input_gates
        )?;
    }

    manager.set_values_with_range(
        PolyId::QTableType,
        PolyForm::Values,
        Fr::zero(),
        num_all_gates..MC::FULL_SLOT_SIZE
    )?;

    Ok(())
}
