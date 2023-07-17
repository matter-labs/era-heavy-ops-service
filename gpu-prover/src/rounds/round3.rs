use super::*;

pub fn round3<S: SynthesisMode, C: Circuit<Bn256>, T: Transcript<Fr>, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
    proof: &mut Proof<Bn256, C>,
    constants: &mut ProverConstants<Fr>,
    transcript: &mut T,
    setup: &mut AsyncSetup,
) -> Result<(), ProvingError> {
    get_round_3_challenges(constants, transcript);

    compute_main_custom_and_permutation_gates(manager, assembly, worker, constants, setup)?;

    compute_lookup_gate(manager, assembly, worker, constants, setup)?;

    let poly_ids = [PolyId::SShifted, PolyId::TShifted, PolyId::ZLookupShifted];

    for id in poly_ids.iter() {
        manager.free_slot(*id, PolyForm::Monomial);
    }

    let msm_handles = compute_quotient_monomial_and_schedule_commitments(manager)?;

    schedule_monomial_copyings_for_last_rounds(manager, assembly, setup, worker)?;

    for (i, commitment) in msm_handles.into_iter().enumerate() {
        let tpart_commitment = commitment.get_result::<MC>(manager)?;
        commit_point_as_xy::<Bn256, T>(transcript, &tpart_commitment);
        proof.quotient_poly_parts_commitments.push(tpart_commitment);
    }

    Ok(())
}

pub fn get_round_3_challenges<T: Transcript<Fr>>(
    constants: &mut ProverConstants<Fr>,
    transcript: &mut T,
) {
    constants.alpha = Vec::with_capacity(9);
    let alpha = transcript.get_challenge();
    let mut current_alpha = Fr::one();
    constants.alpha.push(current_alpha);

    for _ in 1..9 {
        current_alpha.mul_assign(&alpha);
        constants.alpha.push(current_alpha);
    }
}

pub fn compute_main_custom_and_permutation_gates<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
    constants: &mut ProverConstants<Fr>,
    setup: &mut AsyncSetup,
) -> Result<(), ProvingError> {
    if S::PRODUCE_SETUP {
        compute_gate_selector_monomials(manager, assembly, worker)?;
    }

    for coset_idx in 0..LDE_FACTOR {
        compute_shifted_polys_for_main_gate(manager, coset_idx)?;

        copy_polynomials_for_main_custom_and_permutation_gates::<S, _>(manager, setup, coset_idx)?;

        compute_main_gate(manager, coset_idx)?;
        compute_custom_gate(manager, constants, coset_idx)?;

        for id in [PolyId::A, PolyId::B, PolyId::C, PolyId::D].iter() {
            manager.multigpu_coset_ifft(*id, coset_idx)?;
        }
    }

    for coset_idx in 0..LDE_FACTOR {
        for i in 0..4 {
            manager.multigpu_coset_fft(PolyId::Sigma(i), coset_idx)?;
        }

        for id in [PolyId::A, PolyId::B, PolyId::C, PolyId::D].iter() {
            manager.multigpu_coset_fft_to_free_slot(*id, coset_idx)?;
            manager.add_constant(*id, PolyForm::LDE(coset_idx), constants.gamma)?;
        }

        manager.copy_from_device_to_free_device(
            PolyId::ZPerm,
            PolyId::ZPermShifted,
            PolyForm::Monomial,
        )?;
        manager.distribute_omega_powers(
            PolyId::ZPermShifted,
            PolyForm::Monomial,
            MC::FULL_SLOT_SIZE_LOG,
            0,
            1,
            false,
        )?;

        manager.multigpu_coset_fft(PolyId::ZPermShifted, coset_idx)?; // TODO save Monomial if there's enough space
        manager.multigpu_coset_fft_to_free_slot(PolyId::ZPerm, coset_idx)?;

        compute_permutation_gate_0(manager, constants, coset_idx)?;
        compute_permutation_gate_1(manager, constants, coset_idx)?;
        compute_permutation_gate_2(manager, constants, coset_idx)?;
    }
    manager.free_slot(PolyId::PI, PolyForm::Monomial);

    Ok(())
}

pub fn compute_gate_selector_monomials<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
) -> Result<(), ProvingError> {
    for (i, poly_id) in [PolyId::QMainSelector, PolyId::QCustomSelector]
        .into_iter()
        .enumerate()
    {
        get_gate_selector_values_from_assembly(manager, assembly, worker, i)?;
        manager.multigpu_ifft(poly_id, false)?;
    }
    // manager.free_host_slot(PolyId::QMainSelector, PolyForm::Values);
    // manager.free_host_slot(PolyId::QCustomSelector, PolyForm::Values);

    Ok(())
}

pub fn compute_shifted_polys_for_main_gate<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    coset_idx: usize,
) -> Result<(), ProvingError> {
    manager.copy_from_device_to_free_device(PolyId::D, PolyId::DNext, PolyForm::Monomial)?;
    manager.distribute_omega_powers(
        PolyId::DNext,
        PolyForm::Monomial,
        MC::FULL_SLOT_SIZE_LOG,
        0,
        1,
        false,
    )?;

    manager.multigpu_coset_fft(PolyId::DNext, coset_idx)?; //TODO save Monomial if there's enough space

    let poly_ids = [PolyId::A, PolyId::B, PolyId::C, PolyId::D];

    manager.multigpu_coset_fft_to_free_slot(PolyId::PI, coset_idx)?;

    for (i, id) in poly_ids.iter().enumerate() {
        manager.multigpu_coset_fft(*id, coset_idx)?;
    }

    Ok(())
}

pub fn copy_polynomials_for_main_custom_and_permutation_gates<
    S: SynthesisMode,
    MC: ManagerConfigs,
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    setup: &mut AsyncSetup,
    coset_idx: usize,
) -> Result<(), ProvingError> {
    let poly_ids = [
        PolyId::QA,
        PolyId::QB,
        PolyId::QC,
        PolyId::QD,
        PolyId::QMab,
        PolyId::QMac,
        PolyId::QDNext,
    ];

    if S::PRODUCE_SETUP && coset_idx == 0 {
        copy_monomials_for_permutation_gates_and_compute_ldes(manager, poly_ids, coset_idx)?;
    } else if coset_idx == 0 {
        copy_monomials_from_setup_for_main_custom_and_permutation_gates(
            manager, setup, poly_ids, coset_idx,
        )?;
    } else {
        copy_monomials_for_permutation_gates_and_compute_ldes(manager, poly_ids, coset_idx)?;
    }

    Ok(())
}

pub fn copy_monomials_from_setup_for_main_custom_and_permutation_gates<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    setup: &mut AsyncSetup,
    poly_ids: [PolyId; 7],
    coset_idx: usize,
) -> Result<(), ProvingError> {
    for (i, id) in poly_ids[..6].iter().enumerate() {
        manager.async_copy_to_device(
            &mut setup.gate_setup_monomials[i],
            *id,
            PolyForm::Monomial,
            0..MC::FULL_SLOT_SIZE,
        )?;
        manager.multigpu_coset_fft(*id, coset_idx)?;
    }

    manager.async_copy_to_device(
        &mut setup.gate_setup_monomials[7],
        PolyId::QDNext,
        PolyForm::Monomial,
        0..MC::FULL_SLOT_SIZE,
    )?;
    manager.multigpu_coset_fft(PolyId::QDNext, coset_idx)?;

    // manager.async_copy_to_device(
    //     &mut setup.gate_selectors_monomials[0],
    //     PolyId::QMainSelector,
    //     PolyForm::Monomial,
    //     0..MC::FULL_SLOT_SIZE,
    // )?;
    // manager.async_copy_to_device(
    //     &mut setup.gate_selectors_monomials[1],
    //     PolyId::QCustomSelector,
    //     PolyForm::Monomial,
    //     0..MC::FULL_SLOT_SIZE,
    // )?;
    compute_values_from_bitvec(
        manager,
        &setup.gate_selectors_bitvecs[0],
        PolyId::QMainSelector,
    );
    compute_values_from_bitvec(
        manager,
        &setup.gate_selectors_bitvecs[1],
        PolyId::QCustomSelector,
    );
    manager.multigpu_ifft(PolyId::QMainSelector, false);
    manager.multigpu_ifft(PolyId::QCustomSelector, false);

    for (i, id) in [PolyId::QMainSelector, PolyId::QCustomSelector]
        .iter()
        .enumerate()
    {
        manager.multigpu_coset_fft(*id, coset_idx)?;
    }

    Ok(())
}

pub fn copy_monomials_for_permutation_gates_and_compute_ldes<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    poly_ids: [PolyId; 7],
    coset_idx: usize,
) -> Result<(), ProvingError> {
    for i in 0..7 {
        manager.multigpu_coset_fft(poly_ids[i], coset_idx)?;
    }

    for (i, id) in [PolyId::QMainSelector, PolyId::QCustomSelector]
        .iter()
        .enumerate()
    {
        manager.multigpu_coset_fft(*id, coset_idx)?;
    }

    Ok(())
}

pub fn compute_main_gate<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    coset_idx: usize,
) -> Result<(), ProvingError> {
    manager.rename_slot(
        PolyId::PI,
        PolyId::Custom("main_gate"),
        PolyForm::LDE(coset_idx),
    );

    manager.new_empty_slot(PolyId::Tmp, PolyForm::LDE(coset_idx));

    manager.copy_from_device_to_device(PolyId::QMab, PolyId::Tmp, PolyForm::LDE(coset_idx));
    manager.mul_assign(PolyId::Tmp, PolyId::A, PolyForm::LDE(coset_idx))?;
    manager.mul_assign(PolyId::Tmp, PolyId::B, PolyForm::LDE(coset_idx))?;
    manager.add_assign(
        PolyId::Custom("main_gate"),
        PolyId::Tmp,
        PolyForm::LDE(coset_idx),
    )?;

    manager.copy_from_device_to_device(PolyId::QMac, PolyId::Tmp, PolyForm::LDE(coset_idx));
    manager.mul_assign(PolyId::Tmp, PolyId::A, PolyForm::LDE(coset_idx))?;
    manager.mul_assign(PolyId::Tmp, PolyId::C, PolyForm::LDE(coset_idx))?;
    manager.add_assign(
        PolyId::Custom("main_gate"),
        PolyId::Tmp,
        PolyForm::LDE(coset_idx),
    )?;

    for (id, q_id) in [PolyId::A, PolyId::B, PolyId::C, PolyId::D, PolyId::DNext]
        .iter()
        .zip(
            [
                PolyId::QA,
                PolyId::QB,
                PolyId::QC,
                PolyId::QD,
                PolyId::QDNext,
            ]
            .iter(),
        )
    {
        manager.copy_from_device_to_device(*q_id, PolyId::Tmp, PolyForm::LDE(coset_idx));
        manager.mul_assign(PolyId::Tmp, *id, PolyForm::LDE(coset_idx))?;
        manager.add_assign(
            PolyId::Custom("main_gate"),
            PolyId::Tmp,
            PolyForm::LDE(coset_idx),
        )?;
    }

    manager.free_slot(PolyId::Tmp, PolyForm::LDE(coset_idx));

    manager.mul_assign(
        PolyId::Custom("main_gate"),
        PolyId::QMainSelector,
        PolyForm::LDE(coset_idx),
    )?;

    manager.rename_slot(
        PolyId::Custom("main_gate"),
        PolyId::TPart(coset_idx),
        PolyForm::LDE(coset_idx),
    );

    let poly_ids = [
        PolyId::QMab,
        PolyId::QMac,
        PolyId::QA,
        PolyId::QB,
        PolyId::QC,
        PolyId::QD,
        PolyId::QDNext,
        PolyId::QMainSelector,
    ];

    for id in poly_ids.iter() {
        if coset_idx < 3 {
            manager.multigpu_coset_ifft(*id, coset_idx);
        } else {
            manager.free_slot(*id, PolyForm::LDE(coset_idx));
        }
    }
    manager.free_slot(PolyId::DNext, PolyForm::LDE(coset_idx));

    Ok(())
}

pub fn compute_custom_gate<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
    coset_idx: usize,
) -> Result<(), ProvingError> {
    manager.copy_from_device_to_free_device(
        PolyId::A,
        PolyId::Custom("custom_gate"),
        PolyForm::LDE(coset_idx),
    )?;
    manager.mul_assign(
        PolyId::Custom("custom_gate"),
        PolyId::A,
        PolyForm::LDE(coset_idx),
    )?;
    manager.sub_assign(
        PolyId::Custom("custom_gate"),
        PolyId::B,
        PolyForm::LDE(coset_idx),
    )?;
    manager.mul_constant(
        PolyId::Custom("custom_gate"),
        PolyForm::LDE(coset_idx),
        constants.alpha[1],
    )?;

    manager.copy_from_device_to_free_device(
        PolyId::B,
        PolyId::Custom("tmp"),
        PolyForm::LDE(coset_idx),
    )?;
    manager.mul_assign(PolyId::Custom("tmp"), PolyId::B, PolyForm::LDE(coset_idx))?;
    manager.sub_assign(PolyId::Custom("tmp"), PolyId::C, PolyForm::LDE(coset_idx))?;
    manager.add_assign_scaled(
        PolyId::Custom("custom_gate"),
        PolyId::Custom("tmp"),
        PolyForm::LDE(coset_idx),
        constants.alpha[2],
    )?;
    manager.free_slot(PolyId::Custom("tmp"), PolyForm::LDE(coset_idx));

    manager.copy_from_device_to_free_device(
        PolyId::A,
        PolyId::Custom("tmp"),
        PolyForm::LDE(coset_idx),
    )?;
    manager.mul_assign(PolyId::Custom("tmp"), PolyId::C, PolyForm::LDE(coset_idx))?;
    manager.sub_assign(PolyId::Custom("tmp"), PolyId::D, PolyForm::LDE(coset_idx))?;
    manager.add_assign_scaled(
        PolyId::Custom("custom_gate"),
        PolyId::Custom("tmp"),
        PolyForm::LDE(coset_idx),
        constants.alpha[3],
    )?;
    manager.free_slot(PolyId::Custom("tmp"), PolyForm::LDE(coset_idx));

    manager.mul_assign(
        PolyId::Custom("custom_gate"),
        PolyId::QCustomSelector,
        PolyForm::LDE(coset_idx),
    )?;
    manager.add_assign(
        PolyId::TPart(coset_idx),
        PolyId::Custom("custom_gate"),
        PolyForm::LDE(coset_idx),
    )?;

    manager.free_slot(PolyId::Custom("custom_gate"), PolyForm::LDE(coset_idx));

    if coset_idx < 3 {
        manager.multigpu_coset_ifft(PolyId::QCustomSelector, coset_idx);
    } else {
        manager.free_slot(PolyId::QCustomSelector, PolyForm::LDE(coset_idx));
    }

    Ok(())
}

pub fn compute_permutation_gate_0<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
    coset_idx: usize,
) -> Result<(), ProvingError> {
    manager.create_x_poly_in_free_slot(PolyId::X, PolyForm::LDE(coset_idx))?;
    manager.mul_constant(PolyId::X, PolyForm::LDE(coset_idx), constants.beta)?;

    manager.copy_from_device_to_free_device(
        PolyId::A,
        PolyId::Custom("permutation_gate"),
        PolyForm::LDE(coset_idx),
    )?;
    manager.add_assign(
        PolyId::Custom("permutation_gate"),
        PolyId::X,
        PolyForm::LDE(coset_idx),
    )?;

    for (i, id) in [PolyId::B, PolyId::C, PolyId::D].iter().enumerate() {
        manager.copy_from_device_to_free_device(
            *id,
            PolyId::Custom("tmp"),
            PolyForm::LDE(coset_idx),
        )?;
        manager.add_assign_scaled(
            PolyId::Custom("tmp"),
            PolyId::X,
            PolyForm::LDE(coset_idx),
            constants.non_residues[i],
        )?;
        manager.mul_assign(
            PolyId::Custom("permutation_gate"),
            PolyId::Custom("tmp"),
            PolyForm::LDE(coset_idx),
        )?;
        manager.free_slot(PolyId::Custom("tmp"), PolyForm::LDE(coset_idx));
    }

    manager.mul_assign(
        PolyId::Custom("permutation_gate"),
        PolyId::ZPerm,
        PolyForm::LDE(coset_idx),
    )?;

    manager.add_assign_scaled(
        PolyId::TPart(coset_idx),
        PolyId::Custom("permutation_gate"),
        PolyForm::LDE(coset_idx),
        constants.alpha[4],
    )?;
    manager.free_slot(PolyId::Custom("permutation_gate"), PolyForm::LDE(coset_idx));

    manager.free_slot(PolyId::X, PolyForm::LDE(coset_idx));

    Ok(())
}

pub fn compute_permutation_gate_1<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
    coset_idx: usize,
) -> Result<(), ProvingError> {
    manager.copy_from_device_to_free_device(
        PolyId::A,
        PolyId::Custom("permutation_gate"),
        PolyForm::LDE(coset_idx),
    )?;
    manager.add_assign_scaled(
        PolyId::Custom("permutation_gate"),
        PolyId::Sigma(0),
        PolyForm::LDE(coset_idx),
        constants.beta,
    )?;

    for (i, id) in [PolyId::B, PolyId::C, PolyId::D].iter().enumerate() {
        manager.copy_from_device_to_free_device(
            *id,
            PolyId::Custom("tmp"),
            PolyForm::LDE(coset_idx),
        )?;
        manager.add_assign_scaled(
            PolyId::Custom("tmp"),
            PolyId::Sigma(i + 1),
            PolyForm::LDE(coset_idx),
            constants.beta,
        )?;
        manager.mul_assign(
            PolyId::Custom("permutation_gate"),
            PolyId::Custom("tmp"),
            PolyForm::LDE(coset_idx),
        )?;
        manager.free_slot(PolyId::Custom("tmp"), PolyForm::LDE(coset_idx));
    }

    manager.mul_assign(
        PolyId::Custom("permutation_gate"),
        PolyId::ZPermShifted,
        PolyForm::LDE(coset_idx),
    )?;

    manager.sub_assign_scaled(
        PolyId::TPart(coset_idx),
        PolyId::Custom("permutation_gate"),
        PolyForm::LDE(coset_idx),
        constants.alpha[4],
    )?;
    manager.free_slot(PolyId::Custom("permutation_gate"), PolyForm::LDE(coset_idx));

    for i in 0..4 {
        manager.multigpu_coset_ifft(PolyId::Sigma(i), coset_idx);
    }

    for id in [PolyId::A, PolyId::B, PolyId::C, PolyId::D].iter() {
        manager.free_slot(*id, PolyForm::LDE(coset_idx));
    }

    manager.free_slot(PolyId::ZPermShifted, PolyForm::LDE(coset_idx));

    Ok(())
}

pub fn compute_permutation_gate_2<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
    coset_idx: usize,
) -> Result<(), ProvingError> {
    manager.create_lagrange_poly_in_free_slot(PolyId::L0, PolyForm::LDE(coset_idx), 0)?; //TODO create Monomial if there's enough space
    manager.sub_constant(PolyId::ZPerm, PolyForm::LDE(coset_idx), Fr::one())?;
    manager.mul_assign(PolyId::ZPerm, PolyId::L0, PolyForm::LDE(coset_idx))?;

    manager.add_assign_scaled(
        PolyId::TPart(coset_idx),
        PolyId::ZPerm,
        PolyForm::LDE(coset_idx),
        constants.alpha[5],
    )?;

    manager.free_slot(PolyId::L0, PolyForm::LDE(coset_idx));
    manager.free_slot(PolyId::ZPerm, PolyForm::LDE(coset_idx));

    Ok(())
}

pub fn compute_lookup_gate<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
    constants: &mut ProverConstants<Fr>,
    setup: &mut AsyncSetup,
) -> Result<(), ProvingError> {
    compute_shifted_lookup_polynomials(manager)?;

    for coset_idx in 0..LDE_FACTOR {
        let poly_ids = [
            PolyId::A,
            PolyId::B,
            PolyId::C,
            PolyId::T,
            PolyId::ZLookup,
            PolyId::TShifted,
        ];

        for (i, id) in poly_ids.iter().enumerate() {
            manager.multigpu_coset_fft_to_free_slot(*id, coset_idx)?;
        }

        manager.create_x_poly_in_free_slot(PolyId::X, PolyForm::LDE(coset_idx))?;
        let omega_pow =
            domain_generator::<Fr>(MC::FULL_SLOT_SIZE).pow([(MC::FULL_SLOT_SIZE - 1) as u64]);
        manager.sub_constant(PolyId::X, PolyForm::LDE(coset_idx), omega_pow)?;

        if S::PRODUCE_SETUP {
            if coset_idx == 0 {
                get_lookup_selector_from_assembly(manager, assembly, worker)?;
                manager.multigpu_ifft(PolyId::QLookupSelector, false)?;
                // manager.free_host_slot(PolyId::QLookupSelector, PolyForm::Values);
                manager
                    .copy_from_device_to_host_pinned(PolyId::QLookupSelector, PolyForm::Monomial)?;
            } else {
                manager
                    .copy_from_host_pinned_to_device(PolyId::QLookupSelector, PolyForm::Monomial)?;
            }

            manager.multigpu_coset_fft(PolyId::QLookupSelector, coset_idx)?;

            if coset_idx == 0 {
                get_table_type_from_assembly(manager, assembly)?;
                manager.multigpu_ifft(PolyId::QTableType, false)?;
                manager.copy_from_device_to_host_pinned(PolyId::QTableType, PolyForm::Monomial)?;
            } else {
                manager.copy_from_host_pinned_to_device(PolyId::QTableType, PolyForm::Monomial)?;
            }

            manager.multigpu_coset_fft(PolyId::QTableType, coset_idx)?;
        } else {
            // manager.async_copy_to_device(
            //     &mut setup.lookup_selector_monomial,
            //     PolyId::QLookupSelector,
            //     PolyForm::Monomial,
            //     0..MC::FULL_SLOT_SIZE,
            // )?;
            compute_values_from_bitvec(
                manager,
                &setup.lookup_selector_bitvec,
                PolyId::QLookupSelector,
            );
            manager.multigpu_ifft(PolyId::QLookupSelector, false)?;
            manager.multigpu_coset_fft(PolyId::QLookupSelector, coset_idx)?;
            manager.async_copy_to_device(
                &mut setup.lookup_table_type_monomial,
                PolyId::QTableType,
                PolyForm::Monomial,
                0..MC::FULL_SLOT_SIZE,
            )?;
            manager.multigpu_coset_fft(PolyId::QTableType, coset_idx)?;
        }

        compute_lookup_gate_0(manager, constants, coset_idx)?;

        let poly_ids = [PolyId::S, PolyId::SShifted, PolyId::ZLookupShifted];

        for (i, id) in poly_ids.iter().enumerate() {
            manager.multigpu_coset_fft_to_free_slot(*id, coset_idx)?;
        }

        compute_lookup_gate_1(manager, constants, coset_idx)?;

        compute_lookup_gate_2(manager, constants, coset_idx)?;

        compute_lookup_gate_3(manager, constants, coset_idx)?;

        //Compute Z_H
        let bitrevessed_idx = [0, 2, 1, 3];
        let mut omega = domain_generator::<Fr>(4 * MC::FULL_SLOT_SIZE);
        omega = omega.pow([bitrevessed_idx[coset_idx] as u64]);
        omega.mul_assign(&Fr::multiplicative_generator());
        omega = omega.pow([MC::FULL_SLOT_SIZE as u64]);
        omega.sub_assign(&Fr::one());
        omega = omega.inverse().unwrap();

        let z_h = omega;

        manager.mul_constant(PolyId::TPart(coset_idx), PolyForm::LDE(coset_idx), z_h)?;
    }

    Ok(())
}

pub fn compute_shifted_lookup_polynomials<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
) -> Result<(), ProvingError> {
    manager.copy_from_device_to_free_device(
        PolyId::ZLookup,
        PolyId::ZLookupShifted,
        PolyForm::Monomial,
    )?;
    manager.distribute_omega_powers(
        PolyId::ZLookupShifted,
        PolyForm::Monomial,
        MC::FULL_SLOT_SIZE_LOG,
        0,
        1,
        false,
    )?;

    manager.copy_from_device_to_free_device(PolyId::S, PolyId::SShifted, PolyForm::Monomial)?;
    manager.distribute_omega_powers(
        PolyId::SShifted,
        PolyForm::Monomial,
        MC::FULL_SLOT_SIZE_LOG,
        0,
        1,
        false,
    )?;

    manager.copy_from_device_to_free_device(PolyId::T, PolyId::TShifted, PolyForm::Monomial)?;
    manager.distribute_omega_powers(
        PolyId::TShifted,
        PolyForm::Monomial,
        MC::FULL_SLOT_SIZE_LOG,
        0,
        1,
        false,
    )?;

    Ok(())
}

pub fn compute_lookup_gate_0<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
    coset_idx: usize,
) -> Result<(), ProvingError> {
    let mut tmp = constants.eta;
    manager.rename_slot(PolyId::A, PolyId::F, PolyForm::LDE(coset_idx));
    manager.add_assign_scaled(PolyId::F, PolyId::B, PolyForm::LDE(coset_idx), tmp)?;
    tmp.mul_assign(&constants.eta);
    manager.add_assign_scaled(PolyId::F, PolyId::C, PolyForm::LDE(coset_idx), tmp)?;
    tmp.mul_assign(&constants.eta);
    manager.add_assign_scaled(PolyId::F, PolyId::QTableType, PolyForm::LDE(coset_idx), tmp)?;
    manager.mul_assign(PolyId::F, PolyId::QLookupSelector, PolyForm::LDE(coset_idx))?;

    manager.add_constant(
        PolyId::F,
        PolyForm::LDE(coset_idx),
        constants.gamma_for_lookup,
    )?;

    manager.rename_slot(
        PolyId::T,
        PolyId::Custom("lookup_gate"),
        PolyForm::LDE(coset_idx),
    );
    manager.add_assign_scaled(
        PolyId::Custom("lookup_gate"),
        PolyId::TShifted,
        PolyForm::LDE(coset_idx),
        constants.beta_for_lookup,
    )?;
    manager.add_constant(
        PolyId::Custom("lookup_gate"),
        PolyForm::LDE(coset_idx),
        constants.gamma_beta_lookup,
    )?;
    manager.mul_assign(
        PolyId::Custom("lookup_gate"),
        PolyId::F,
        PolyForm::LDE(coset_idx),
    )?;
    manager.mul_constant(
        PolyId::Custom("lookup_gate"),
        PolyForm::LDE(coset_idx),
        constants.beta_plus_one_lookup,
    )?;
    manager.mul_assign(
        PolyId::Custom("lookup_gate"),
        PolyId::ZLookup,
        PolyForm::LDE(coset_idx),
    )?;

    let poly_ids = [
        PolyId::F,
        PolyId::B,
        PolyId::C,
        PolyId::QTableType,
        PolyId::QLookupSelector,
        PolyId::TShifted,
    ];

    for id in poly_ids.iter() {
        manager.free_slot(*id, PolyForm::LDE(coset_idx));
    }

    Ok(())
}

pub fn compute_lookup_gate_1<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
    coset_idx: usize,
) -> Result<(), ProvingError> {
    manager.rename_slot(
        PolyId::S,
        PolyId::Custom("lookup_gate2"),
        PolyForm::LDE(coset_idx),
    );
    manager.add_assign_scaled(
        PolyId::Custom("lookup_gate2"),
        PolyId::SShifted,
        PolyForm::LDE(coset_idx),
        constants.beta_for_lookup,
    )?;
    manager.add_constant(
        PolyId::Custom("lookup_gate2"),
        PolyForm::LDE(coset_idx),
        constants.gamma_beta_lookup,
    )?;
    manager.mul_assign(
        PolyId::Custom("lookup_gate2"),
        PolyId::ZLookupShifted,
        PolyForm::LDE(coset_idx),
    )?;

    manager.sub_assign(
        PolyId::Custom("lookup_gate2"),
        PolyId::Custom("lookup_gate"),
        PolyForm::LDE(coset_idx),
    )?;

    manager.mul_assign(
        PolyId::Custom("lookup_gate2"),
        PolyId::X,
        PolyForm::LDE(coset_idx),
    )?;

    manager.add_assign_scaled(
        PolyId::TPart(coset_idx),
        PolyId::Custom("lookup_gate2"),
        PolyForm::LDE(coset_idx),
        constants.alpha[6],
    )?;

    let poly_ids = [
        PolyId::Custom("lookup_gate"),
        PolyId::Custom("lookup_gate2"),
        PolyId::SShifted,
        PolyId::X,
        PolyId::ZLookupShifted,
    ];

    for id in poly_ids.iter() {
        manager.free_slot(*id, PolyForm::LDE(coset_idx));
    }

    Ok(())
}

pub fn compute_lookup_gate_2<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
    coset_idx: usize,
) -> Result<(), ProvingError> {
    manager.create_lagrange_poly_in_free_slot(PolyId::L0, PolyForm::LDE(coset_idx), 0)?; //TODO create Monomial if there's enough space

    manager.copy_from_device_to_free_device(
        PolyId::ZLookup,
        PolyId::Custom("lookup_gate"),
        PolyForm::LDE(coset_idx),
    )?;
    manager.sub_constant(
        PolyId::Custom("lookup_gate"),
        PolyForm::LDE(coset_idx),
        Fr::one(),
    )?;
    manager.mul_assign(
        PolyId::Custom("lookup_gate"),
        PolyId::L0,
        PolyForm::LDE(coset_idx),
    )?;

    manager.add_assign_scaled(
        PolyId::TPart(coset_idx),
        PolyId::Custom("lookup_gate"),
        PolyForm::LDE(coset_idx),
        constants.alpha[7],
    )?;

    for id in [PolyId::Custom("lookup_gate"), PolyId::L0].iter() {
        manager.free_slot(*id, PolyForm::LDE(coset_idx));
    }

    Ok(())
}

pub fn compute_lookup_gate_3<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
    coset_idx: usize,
) -> Result<(), ProvingError> {
    manager.create_lagrange_poly_in_free_slot(
        PolyId::Ln1,
        PolyForm::LDE(coset_idx),
        MC::FULL_SLOT_SIZE - 1,
    )?; //TODO create Monomial if there's enough space

    manager.rename_slot(
        PolyId::ZLookup,
        PolyId::Custom("lookup_gate"),
        PolyForm::LDE(coset_idx),
    );
    manager.sub_constant(
        PolyId::Custom("lookup_gate"),
        PolyForm::LDE(coset_idx),
        constants.expected,
    )?;
    manager.mul_assign(
        PolyId::Custom("lookup_gate"),
        PolyId::Ln1,
        PolyForm::LDE(coset_idx),
    )?;

    manager.add_assign_scaled(
        PolyId::TPart(coset_idx),
        PolyId::Custom("lookup_gate"),
        PolyForm::LDE(coset_idx),
        constants.alpha[8],
    )?;

    for id in [PolyId::Custom("lookup_gate"), PolyId::Ln1].iter() {
        manager.free_slot(*id, PolyForm::LDE(coset_idx));
    }

    Ok(())
}

pub fn compute_quotient_monomial_and_schedule_commitments<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
) -> Result<Vec<MSMHandle>, ProvingError> {
    let polys = [
        PolyId::TPart(0),
        PolyId::TPart(1),
        PolyId::TPart(2),
        PolyId::TPart(3),
    ];

    manager.multigpu_coset_4n_ifft(polys)?;

    let geninv = Fr::multiplicative_generator().inverse().unwrap();
    for coset_idx in 0..LDE_FACTOR {
        manager.distribute_powers(PolyId::TPart(coset_idx), PolyForm::Monomial, geninv)?;
        let coset_mult = geninv.pow([(coset_idx * MC::FULL_SLOT_SIZE) as u64]);
        manager.mul_constant(PolyId::TPart(coset_idx), PolyForm::Monomial, coset_mult)?;
    }

    let mut msm_handles = vec![];

    for coset_idx in 0..LDE_FACTOR {
        let handle = manager.msm(PolyId::TPart(coset_idx))?;
        msm_handles.push(handle);
    }

    Ok(msm_handles)
}

pub fn schedule_monomial_copyings_for_last_rounds<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    setup: &mut AsyncSetup,
    worker: &Worker,
) -> Result<(), ProvingError> {
    if S::PRODUCE_SETUP {
        manager.copy_from_host_pinned_to_device(PolyId::QLookupSelector, PolyForm::Monomial)?;
        manager.free_host_slot(PolyId::QLookupSelector, PolyForm::Monomial);

        manager.copy_from_host_pinned_to_device(PolyId::QTableType, PolyForm::Monomial)?;
        manager.free_host_slot(PolyId::QTableType, PolyForm::Monomial);

        for i in 0..8 {
            copying_setup_poly(manager, &assembly, i);
            // manager.multigpu_ifft(GATE_SETUP_LIST[i], false)?;
        }

        get_gate_selector_values_from_assembly(manager, assembly, worker, 0)?;
        get_gate_selector_values_from_assembly(manager, assembly, worker, 1)?;
        manager.multigpu_ifft(PolyId::QMainSelector, false)?;
    } else {
        // manager.async_copy_to_device(
        //     &mut setup.lookup_selector_monomial,
        //     PolyId::QLookupSelector,
        //     PolyForm::Monomial,
        //     0..MC::FULL_SLOT_SIZE,
        // )?;
        compute_values_from_bitvec(
            manager,
            &setup.lookup_selector_bitvec,
            PolyId::QLookupSelector,
        );
        manager.multigpu_ifft(PolyId::QLookupSelector, false)?;

        manager.async_copy_to_device(
            &mut setup.lookup_table_type_monomial,
            PolyId::QTableType,
            PolyForm::Monomial,
            0..MC::FULL_SLOT_SIZE,
        )?;

        for (i, poly_id) in GATE_SETUP_LIST.iter().enumerate() {
            manager.async_copy_to_device(
                &mut setup.gate_setup_monomials[i],
                *poly_id,
                PolyForm::Monomial,
                0..MC::FULL_SLOT_SIZE,
            )?;
        }

        for (i, poly_id) in [PolyId::QMainSelector, PolyId::QCustomSelector]
            .into_iter()
            .enumerate()
        {
            // manager.async_copy_to_device(
            //     &mut setup.gate_selectors_monomials[i],
            //     poly_id,
            //     PolyForm::Monomial,
            //     0..MC::FULL_SLOT_SIZE,
            // )?;
            compute_values_from_bitvec(manager, &setup.gate_selectors_bitvecs[i], poly_id);
            manager.multigpu_ifft(poly_id, false)?;
        }
    }

    Ok(())
}

pub fn get_gate_selector_values_from_assembly<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
    idx: usize,
) -> Result<(), ProvingError> {
    let poly_id = if idx == 0 {
        PolyId::QMainSelector
    } else {
        PolyId::QCustomSelector
    };

    crate_selector_on_manager(manager, assembly, poly_id)?;

    Ok(())
}
