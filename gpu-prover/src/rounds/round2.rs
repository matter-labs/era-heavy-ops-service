use super::*;

pub fn round2<S: SynthesisMode, C: Circuit<Bn256>, T: Transcript<Fr>, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
    proof: &mut Proof<Bn256, C>,
    constants: &mut ProverConstants<Fr>,
    transcript: &mut T,
    setup: &mut AsyncSetup,
    input_values: &Vec<Fr>,
) -> Result<(), ProvingError> {
    get_round_2_permutation_challenges(constants, transcript);

    compute_z_perm(manager, constants)?;

    let z_perm_handle = manager.msm(PolyId::ZPerm)?;

    let z_perm_commitment = z_perm_handle.get_result::<MC>(manager)?;
    commit_point_as_xy::<Bn256, T>(transcript, &z_perm_commitment);
    proof.copy_permutation_grand_product_commitment = z_perm_commitment;

    get_round_2_lookup_challenges::<T, MC>(constants, transcript);

    compute_z_lookup(manager, constants)?;

    let z_lookup_handle = manager.msm(PolyId::ZLookup)?;

    schedule_ops_for_round_3(manager, assembly, worker, setup, input_values)?;

    let z_lookup_commitment = z_lookup_handle.get_result::<MC>(manager)?;
    commit_point_as_xy::<Bn256, T>(transcript, &z_lookup_commitment);
    proof.lookup_grand_product_commitment = Some(z_lookup_commitment);

    Ok(())
}

pub fn get_round_2_permutation_challenges<T: Transcript<Fr>>(
    constants: &mut ProverConstants<Fr>,
    transcript: &mut T,
) {
    constants.beta = transcript.get_challenge();
    constants.gamma = transcript.get_challenge();
}

pub fn get_round_2_lookup_challenges<T: Transcript<Fr>, MC: ManagerConfigs>(
    constants: &mut ProverConstants<Fr>,
    transcript: &mut T,
) {
    constants.beta_for_lookup = transcript.get_challenge();
    constants.gamma_for_lookup = transcript.get_challenge();

    constants.beta_plus_one_lookup = constants.beta_for_lookup;
    constants.beta_plus_one_lookup.add_assign(&Fr::one());
    constants.gamma_beta_lookup = constants.beta_plus_one_lookup;
    constants
        .gamma_beta_lookup
        .mul_assign(&constants.gamma_for_lookup);

    constants.expected = constants
        .gamma_beta_lookup
        .pow([(MC::FULL_SLOT_SIZE - 1) as u64]);
}

pub fn compute_z_perm<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
) -> Result<(), ProvingError> {
    manager.new_empty_slot(PolyId::Tmp, PolyForm::Values);
    manager.new_empty_slot(PolyId::Tmp2, PolyForm::Values);

    compute_z_perm_num(manager, constants)?;
    compute_z_perm_den(manager, constants)?;

    manager.batch_inversion(PolyId::ZPermDen, PolyForm::Values)?;
    manager.mul_assign(PolyId::ZPermNum, PolyId::ZPermDen, PolyForm::Values)?;

    free_useless_round_2_slots_perm(manager)?;

    manager.shifted_grand_product_to_new_slot(PolyId::ZPermNum, PolyId::ZPerm, PolyForm::Values)?;

    manager.free_slot(PolyId::ZPermNum, PolyForm::Values);

    manager.multigpu_ifft(PolyId::ZPerm, false)?;
    Ok(())
}

pub fn compute_z_perm_num<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
) -> Result<(), ProvingError> {
    manager.create_x_poly_in_free_slot(PolyId::X, PolyForm::Values)?;
    manager.mul_constant(PolyId::X, PolyForm::Values, constants.beta)?;

    manager.copy_from_device_to_free_device(PolyId::A, PolyId::ZPermNum, PolyForm::Values)?;

    manager.add_constant(PolyId::ZPermNum, PolyForm::Values, constants.gamma)?;

    manager.add_assign(PolyId::ZPermNum, PolyId::X, PolyForm::Values)?;

    for (i, id) in [PolyId::B, PolyId::C, PolyId::D].iter().enumerate() {
        manager.copy_from_device_to_device(*id, PolyId::Tmp, PolyForm::Values)?;
        manager.add_assign_scaled(
            PolyId::Tmp,
            PolyId::X,
            PolyForm::Values,
            constants.non_residues[i],
        )?;
        manager.add_constant(PolyId::Tmp, PolyForm::Values, constants.gamma)?;
        manager.mul_assign(PolyId::ZPermNum, PolyId::Tmp, PolyForm::Values)?;
    }

    Ok(())
}

pub fn compute_z_perm_den<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
) -> Result<(), ProvingError> {
    manager.copy_from_device_to_free_device(PolyId::A, PolyId::ZPermDen, PolyForm::Values)?;
    manager.add_constant(PolyId::ZPermDen, PolyForm::Values, constants.gamma)?;
    manager.add_assign_scaled(
        PolyId::ZPermDen,
        PolyId::Sigma(0),
        PolyForm::Values,
        constants.beta,
    )?;

    for (i, id) in [PolyId::B, PolyId::C, PolyId::D].iter().enumerate() {
        manager.copy_from_device_to_device(*id, PolyId::Tmp2, PolyForm::Values)?;
        manager.add_assign_scaled(
            PolyId::Tmp2,
            PolyId::Sigma(i + 1),
            PolyForm::Values,
            constants.beta,
        )?;
        manager.add_constant(PolyId::Tmp2, PolyForm::Values, constants.gamma)?;
        manager.mul_assign(PolyId::ZPermDen, PolyId::Tmp2, PolyForm::Values)?;
    }

    Ok(())
}

pub fn compute_z_lookup<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
) -> Result<(), ProvingError> {
    compute_z_lookup_num(manager, constants)?;
    compute_z_lookup_den(manager, constants)?;

    manager.batch_inversion(PolyId::ZLookupDen, PolyForm::Values)?;
    manager.mul_assign(PolyId::ZLookupNum, PolyId::ZLookupDen, PolyForm::Values)?;

    free_useless_round_2_slots_lookup(manager)?;

    manager.shifted_grand_product_to_new_slot(
        PolyId::ZLookupNum,
        PolyId::ZLookup,
        PolyForm::Values,
    )?;

    manager.free_slot(PolyId::ZLookupNum, PolyForm::Values);

    manager.multigpu_ifft(PolyId::ZLookup, false)?;
    Ok(())
}

pub fn compute_z_lookup_num<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
) -> Result<(), ProvingError> {
    let mut beta_plus_one = constants.beta_for_lookup;
    beta_plus_one.add_assign(&Fr::one());
    let mut gamma_beta = beta_plus_one;
    gamma_beta.mul_assign(&constants.gamma_for_lookup);

    manager.copy_leftshifted_from_device_to_free_device(
        PolyId::T,
        PolyId::ZLookupNum,
        PolyForm::Values,
        Fr::one(),
    )?;

    manager.mul_constant(
        PolyId::ZLookupNum,
        PolyForm::Values,
        constants.beta_for_lookup,
    )?;
    manager.add_assign(PolyId::ZLookupNum, PolyId::T, PolyForm::Values)?;
    manager.add_constant(PolyId::ZLookupNum, PolyForm::Values, gamma_beta)?;

    manager.copy_from_device_to_device(PolyId::F, PolyId::Tmp, PolyForm::Values)?;
    manager.add_constant(PolyId::Tmp, PolyForm::Values, constants.gamma_for_lookup)?;
    manager.mul_assign(PolyId::ZLookupNum, PolyId::Tmp, PolyForm::Values)?;
    manager.mul_constant(PolyId::ZLookupNum, PolyForm::Values, beta_plus_one)?;

    Ok(())
}

pub fn compute_z_lookup_den<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
) -> Result<(), ProvingError> {
    let mut beta_plus_one = constants.beta_for_lookup;
    beta_plus_one.add_assign(&Fr::one());
    let mut gamma_beta = beta_plus_one;
    gamma_beta.mul_assign(&constants.gamma_for_lookup);

    manager.copy_leftshifted_from_device_to_free_device(
        PolyId::S,
        PolyId::ZLookupDen,
        PolyForm::Values,
        Fr::one(),
    )?;
    manager.mul_constant(
        PolyId::ZLookupDen,
        PolyForm::Values,
        constants.beta_for_lookup,
    )?;
    manager.add_assign(PolyId::ZLookupDen, PolyId::S, PolyForm::Values)?;
    manager.add_constant(PolyId::ZLookupDen, PolyForm::Values, gamma_beta)?;

    Ok(())
}

pub fn free_useless_round_2_slots_perm<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
) -> Result<(), ProvingError> {
    let poly_ids = [
        PolyId::A,
        PolyId::B,
        PolyId::C,
        PolyId::D,
        PolyId::X,
        PolyId::Tmp2,
        PolyId::ZPermDen,
    ];

    for id in poly_ids.iter() {
        manager.free_slot(*id, PolyForm::Values);
    }

    for idx in 0..4 {
        manager.multigpu_ifft(PolyId::Sigma(idx), false);
    }

    Ok(())
}

pub fn free_useless_round_2_slots_lookup<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
) -> Result<(), ProvingError> {
    let poly_ids = [
        PolyId::Tmp,
        PolyId::ZLookupDen,
        PolyId::F,
        PolyId::S,
        PolyId::T,
    ];

    for id in poly_ids.iter() {
        manager.free_slot(*id, PolyForm::Values);
    }
    Ok(())
}

pub fn schedule_ops_for_round_3<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
    setup: &mut AsyncSetup,
    input_values: &Vec<Fr>,
) -> Result<(), ProvingError> {
    let len = input_values.len();

    let mut poly = manager.get_free_host_slot_values_mut(PolyId::PI, PolyForm::Values)?;

    fill_with_zeros(worker, &mut poly[len..]);
    async_copy(worker, &mut poly[0..len], input_values);

    manager.copy_from_host_pinned_to_device(PolyId::PI, PolyForm::Values)?;

    schedule_copying_gate_coeffs(manager, assembly, worker, setup)?;

    Ok(())
}

pub fn schedule_copying_gate_coeffs<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
    setup: &mut AsyncSetup,
) -> Result<(), ProvingError> {
    if !S::PRODUCE_SETUP {
        copying_and_computing_q_const_plus_pi_with_setup(manager, setup)?;
    } else {
        copying_and_computing_ifft_of_gate_coeffs(manager, assembly, setup)?;
        copying_and_computing_q_const_plus_pi(manager, assembly, worker, setup)?;
        copying_and_computing_ifft_of_q_d_next(manager, assembly, setup)?;
    }

    Ok(())
}

pub fn copying_and_computing_q_const_plus_pi_with_setup<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    setup: &mut AsyncSetup,
) -> Result<(), ProvingError> {
    manager.async_copy_to_device(
        &mut setup.gate_setup_monomials[6],
        PolyId::QConst,
        PolyForm::Monomial,
        0..MC::FULL_SLOT_SIZE,
    )?;

    manager.multigpu_ifft(PolyId::PI, false)?;
    manager.add_assign(PolyId::PI, PolyId::QConst, PolyForm::Monomial)?;
    manager.free_slot(PolyId::QConst, PolyForm::Monomial);

    manager.free_host_slot(PolyId::PI, PolyForm::Values);

    Ok(())
}

pub fn copying_and_computing_ifft_of_gate_coeffs<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    setup: &mut AsyncSetup,
) -> Result<(), ProvingError> {
    manager.free_host_slot(PolyId::PI, PolyForm::Values);

    for i in 0..6 {
        copying_setup_poly(manager, &assembly, i);
        manager.multigpu_ifft(GATE_SETUP_LIST[i], false)?;
    }

    Ok(())
}

pub fn copying_and_computing_q_const_plus_pi<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
    setup: &mut AsyncSetup,
) -> Result<(), ProvingError> {
    copying_setup_poly(manager, &assembly, 6);

    manager.add_assign(PolyId::PI, PolyId::QConst, PolyForm::Values)?;
    manager.multigpu_ifft(PolyId::PI, false)?;

    manager.free_slot(PolyId::QConst, PolyForm::Values);

    Ok(())
}

pub fn copying_and_computing_ifft_of_q_d_next<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    setup: &mut AsyncSetup,
) -> Result<(), ProvingError> {
    copying_setup_poly(manager, assembly, 7);
    manager.multigpu_ifft(GATE_SETUP_LIST[7], false)?;

    Ok(())
}

pub fn copying_setup_poly<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    poly_id: usize,
) -> Result<(), ProvingError> {
    let id = PolyIdentifier::GateSetupPolynomial(
        "main gate of width 4 with D_next and selector optimization",
        poly_id,
    );
    let poly_id = GATE_SETUP_LIST[poly_id];
    let num_input_gates = assembly.num_input_gates;

    let src = &assembly.aux_storage.setup_map.get(&id).unwrap()[..];
    let end = num_input_gates + src.len();

    manager.new_empty_slot(poly_id, PolyForm::Values);

    if num_input_gates != 0 {
        unsafe {
            manager.async_copy_from_pointer_and_range(
                poly_id,
                PolyForm::Values,
                &assembly.inputs_storage.setup_map.get(&id).unwrap()[0] as *const Fr,
                0..num_input_gates,
            )?;
        }
    }

    unsafe {
        manager.async_copy_from_pointer_and_range(
            poly_id,
            PolyForm::Values,
            &src[0] as *const Fr,
            num_input_gates..end,
        )?;
    }

    manager.set_values_with_range(
        poly_id,
        PolyForm::Values,
        Fr::zero(),
        end..MC::FULL_SLOT_SIZE,
    )?;

    Ok(())
}
