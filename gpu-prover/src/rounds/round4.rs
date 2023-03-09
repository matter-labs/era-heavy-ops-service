use super::*;

pub fn round4<C: Circuit<Bn256>, T: Transcript<Fr>, S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    proof: &mut Proof<Bn256, C>,
    constants: &mut ProverConstants<Fr>,
    transcript: &mut T,
) -> Result<(), ProvingError> {
    let z = transcript.get_challenge();
    constants.z = z;

    make_lin_comb_of_t_poly(manager, z)?;
    let handles = schedule_evaluation(manager, z)?;
    commit_all_poly_openings(manager, proof, transcript, handles)?;
    compute_linearization_poly::<_, S, _>(manager, proof, constants)?;
    evaluate_linearization_at_z(manager, proof, transcript, z)?;

    free_useless_round_4_slots::<S, _>(manager)?;
    Ok(())
}

fn make_lin_comb_of_t_poly<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    z: Fr,
) -> Result<(), ProvingError> {
    for i in 1..4 {
        manager.add_assign_scaled(
            PolyId::TPart(0),
            PolyId::TPart(i),
            PolyForm::Monomial,
            z.pow([(i * MC::FULL_SLOT_SIZE) as u64]),
        )?;
        manager.free_slot(PolyId::TPart(i), PolyForm::Monomial);
    }
    Ok(())
}

fn schedule_evaluation<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    z: Fr,
) -> Result<Vec<EvaluationHandle>, ProvingError> {
    let mut zw = domain_generator::<Fr>(MC::FULL_SLOT_SIZE);
    zw.mul_assign(&z);

    let mut handles = vec![];

    let polys = vec![
        (PolyId::TPart(0), 0),
        (PolyId::A, 0),
        (PolyId::B, 0),
        (PolyId::C, 0),
        (PolyId::D, 0),
        (PolyId::D, 1),
        (PolyId::QMainSelector, 0),
        (PolyId::Sigma(0), 0),
        (PolyId::Sigma(1), 0),
        (PolyId::Sigma(2), 0),
        (PolyId::ZPerm, 1),
        (PolyId::T, 0),
        (PolyId::QLookupSelector, 0),
        (PolyId::QTableType, 0),
        (PolyId::S, 1),
        (PolyId::ZLookup, 1),
        (PolyId::T, 1),
    ];

    for (i, (id, dilation)) in polys.iter().enumerate() {
        // let device_id = i % NUM_GPUS;
        if *dilation == 0 {
            handles.push(manager.evaluate_at(*id, z)?)
        } else {
            handles.push(manager.evaluate_at(*id, zw)?)
        }
    }

    Ok(handles)
}

pub fn commit_all_poly_openings<C: Circuit<Bn256>, T: Transcript<Fr>, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    proof: &mut Proof<Bn256, C>,
    transcript: &mut T,
    handles: Vec<EvaluationHandle>,
) -> Result<(), ProvingError> {
    for (i, handle) in handles.into_iter().enumerate() {
        // let device_id = i % NUM_GPUS;

        let value = handle.get_result(manager)?;
        transcript.commit_field_element(&value);
        match i {
            0 => proof.quotient_poly_opening_at_z = value,
            1 => proof.state_polys_openings_at_z.push(value),
            2 => proof.state_polys_openings_at_z.push(value),
            3 => proof.state_polys_openings_at_z.push(value),
            4 => proof.state_polys_openings_at_z.push(value),
            5 => proof.state_polys_openings_at_dilations.push((1, 3, value)),
            6 => proof.gate_selectors_openings_at_z.push((0, value)),
            7 => proof.copy_permutation_polys_openings_at_z.push(value),
            8 => proof.copy_permutation_polys_openings_at_z.push(value),
            9 => proof.copy_permutation_polys_openings_at_z.push(value),
            10 => proof.copy_permutation_grand_product_opening_at_z_omega = value,
            11 => proof.lookup_t_poly_opening_at_z = Some(value),
            12 => proof.lookup_selector_poly_opening_at_z = Some(value),
            13 => proof.lookup_table_type_poly_opening_at_z = Some(value),
            14 => proof.lookup_s_poly_opening_at_z_omega = Some(value),
            15 => proof.lookup_grand_product_opening_at_z_omega = Some(value),
            16 => proof.lookup_t_poly_opening_at_z_omega = Some(value),
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn compute_linearization_poly<C: Circuit<Bn256>, S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    proof: &mut Proof<Bn256, C>,
    constants: &ProverConstants<Fr>,
) -> Result<(), ProvingError> {
    compute_main_gate_contribution_in_linearization_poly::<_, S, _>(manager, proof)?;
    compute_custom_gate_contribution_in_linearization_poly::<_, S, _>(
        manager,
        proof,
        constants.alpha[1],
    )?;
    compute_perm_arg_contribution_in_linearization_poly(manager, proof, constants)?;
    compute_lookup_arg_contribution_in_linearization_poly(manager, proof, constants)?;
    Ok(())
}

fn compute_main_gate_contribution_in_linearization_poly<
    C: Circuit<Bn256>,
    S: SynthesisMode,
    MC: ManagerConfigs,
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    proof: &mut Proof<Bn256, C>,
) -> Result<(), ProvingError> {
    let poly_form = if S::PRODUCE_SETUP {
        PolyForm::Values
    } else {
        PolyForm::Monomial
    };

    manager.rename_slot(PolyId::QConst, PolyId::R, poly_form);
    manager.add_assign_scaled(
        PolyId::R,
        PolyId::QA,
        poly_form,
        proof.state_polys_openings_at_z[0],
    )?;
    manager.add_assign_scaled(
        PolyId::R,
        PolyId::QB,
        poly_form,
        proof.state_polys_openings_at_z[1],
    )?;
    manager.add_assign_scaled(
        PolyId::R,
        PolyId::QC,
        poly_form,
        proof.state_polys_openings_at_z[2],
    )?;
    manager.add_assign_scaled(
        PolyId::R,
        PolyId::QD,
        poly_form,
        proof.state_polys_openings_at_z[3],
    )?;
    let mut ab_at_z = proof.state_polys_openings_at_z[0];
    ab_at_z.mul_assign(&proof.state_polys_openings_at_z[1]);
    manager.add_assign_scaled(PolyId::R, PolyId::QMab, poly_form, ab_at_z)?;
    let mut ac_at_z = proof.state_polys_openings_at_z[0];
    ac_at_z.mul_assign(&proof.state_polys_openings_at_z[2]);
    manager.add_assign_scaled(PolyId::R, PolyId::QMac, poly_form, ac_at_z)?;
    manager.add_assign_scaled(
        PolyId::R,
        PolyId::QDNext,
        poly_form,
        proof.state_polys_openings_at_dilations[0].2,
    )?;

    manager.mul_constant(
        PolyId::R,
        poly_form,
        proof.gate_selectors_openings_at_z[0].1,
    )?;

    Ok(())
}

fn compute_custom_gate_contribution_in_linearization_poly<
    C: Circuit<Bn256>,
    S: SynthesisMode,
    MC: ManagerConfigs,
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    proof: &mut Proof<Bn256, C>,
    alpha: Fr,
) -> Result<(), ProvingError> {
    let (a, b, c, d) = (
        proof.state_polys_openings_at_z[0],
        proof.state_polys_openings_at_z[1],
        proof.state_polys_openings_at_z[2],
        proof.state_polys_openings_at_z[3],
    );

    let mut coeff = a;
    coeff.mul_assign(&a);
    coeff.sub_assign(&b);
    let mut tmp = b;
    tmp.mul_assign(&b);
    tmp.sub_assign(&c);
    tmp.mul_assign(&alpha);
    coeff.add_assign(&tmp);
    let mut tmp = a;
    tmp.mul_assign(&c);
    tmp.sub_assign(&d);
    tmp.mul_assign(&alpha);
    tmp.mul_assign(&alpha);
    coeff.add_assign(&tmp);
    coeff.mul_assign(&alpha);

    let poly_form = if S::PRODUCE_SETUP {
        manager.add_assign_scaled(PolyId::R, PolyId::QCustomSelector, PolyForm::Values, coeff)?;
        manager.multigpu_ifft(PolyId::R, false)?;
    } else {
        manager.add_assign_scaled(
            PolyId::R,
            PolyId::QCustomSelector,
            PolyForm::Monomial,
            coeff,
        )?;
    };

    Ok(())
}

fn compute_perm_arg_contribution_in_linearization_poly<C: Circuit<Bn256>, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    proof: &mut Proof<Bn256, C>,
    constants: &ProverConstants<Fr>,
) -> Result<(), ProvingError> {
    let coeff = first_coeff_for_perm_arg_in_linearization_poly(proof, constants);
    manager.add_assign_scaled(PolyId::R, PolyId::ZPerm, PolyForm::Monomial, coeff)?;

    let coeff = second_coeff_for_perm_arg_in_linearization_poly(proof, constants);
    manager.sub_assign_scaled(PolyId::R, PolyId::Sigma(3), PolyForm::Monomial, coeff)?;

    let mut coeff = lagrange_0_poly_at_z::<MC>(constants.z);

    coeff.mul_assign(&constants.alpha[5]);
    manager.add_assign_scaled(PolyId::R, PolyId::ZPerm, PolyForm::Monomial, coeff)?;

    Ok(())
}

fn first_coeff_for_perm_arg_in_linearization_poly<C: Circuit<Bn256>>(
    proof: &mut Proof<Bn256, C>,
    constants: &ProverConstants<Fr>,
) -> Fr {
    let (a, b, c, d) = (
        proof.state_polys_openings_at_z[0],
        proof.state_polys_openings_at_z[1],
        proof.state_polys_openings_at_z[2],
        proof.state_polys_openings_at_z[3],
    );

    let mut coeff = constants.z;
    coeff.mul_assign(&constants.beta);
    coeff.add_assign(&a);
    coeff.add_assign(&constants.gamma);
    let mut tmp = constants.z;
    tmp.mul_assign(&constants.beta);
    tmp.mul_assign(&constants.non_residues[0]);
    tmp.add_assign(&b);
    tmp.add_assign(&constants.gamma);
    coeff.mul_assign(&tmp);
    tmp = constants.z;
    tmp.mul_assign(&constants.beta);
    tmp.mul_assign(&constants.non_residues[1]);
    tmp.add_assign(&c);
    tmp.add_assign(&constants.gamma);
    coeff.mul_assign(&tmp);
    tmp = constants.z;
    tmp.mul_assign(&constants.beta);
    tmp.mul_assign(&constants.non_residues[2]);
    tmp.add_assign(&d);
    tmp.add_assign(&constants.gamma);
    coeff.mul_assign(&tmp);
    coeff.mul_assign(&constants.alpha[4]);

    coeff
}

fn second_coeff_for_perm_arg_in_linearization_poly<C: Circuit<Bn256>>(
    proof: &mut Proof<Bn256, C>,
    constants: &ProverConstants<Fr>,
) -> Fr {
    let (a, b, c, sigma_0, sigma_1, sigma_2, z_perm_zw) = (
        proof.state_polys_openings_at_z[0],
        proof.state_polys_openings_at_z[1],
        proof.state_polys_openings_at_z[2],
        proof.copy_permutation_polys_openings_at_z[0],
        proof.copy_permutation_polys_openings_at_z[1],
        proof.copy_permutation_polys_openings_at_z[2],
        proof.copy_permutation_grand_product_opening_at_z_omega,
    );

    let mut coeff = z_perm_zw;
    let mut tmp = sigma_0;
    tmp.mul_assign(&constants.beta);
    tmp.add_assign(&a);
    tmp.add_assign(&constants.gamma);
    coeff.mul_assign(&tmp);
    tmp = sigma_1;
    tmp.mul_assign(&constants.beta);
    tmp.add_assign(&b);
    tmp.add_assign(&constants.gamma);
    coeff.mul_assign(&tmp);
    tmp = sigma_2;
    tmp.mul_assign(&constants.beta);
    tmp.add_assign(&c);
    tmp.add_assign(&constants.gamma);
    coeff.mul_assign(&tmp);
    coeff.mul_assign(&constants.beta);
    coeff.mul_assign(&constants.alpha[4]);

    coeff
}

fn lagrange_0_poly_at_z<MC: ManagerConfigs>(z: Fr) -> Fr {
    let mut num = z.pow(&[MC::FULL_SLOT_SIZE as u64]);
    num.sub_assign(&Fr::one());

    let size_as_fe = Fr::from_str(&format!("{}", MC::FULL_SLOT_SIZE)).unwrap();

    let mut den = z;
    den.sub_assign(&Fr::one());
    den.mul_assign(&size_as_fe);
    let den = den.inverse().ok_or(SynthesisError::DivisionByZero).unwrap();

    num.mul_assign(&den);
    num
}

fn compute_lookup_arg_contribution_in_linearization_poly<C: Circuit<Bn256>, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    proof: &mut Proof<Bn256, C>,
    constants: &ProverConstants<Fr>,
) -> Result<(), ProvingError> {
    let coeff = first_coeff_for_lookup_arg_in_linearization_poly::<C, MC>(proof, constants);
    manager.add_assign_scaled(PolyId::R, PolyId::S, PolyForm::Monomial, coeff)?;

    let mut coeff = second_coeff_for_lookup_arg_in_linearization_poly::<C, MC>(proof, constants);

    let mut tmp = lagrange_0_poly_at_z::<MC>(constants.z);
    tmp.mul_assign(&constants.alpha[7]);
    coeff.add_assign(&tmp);

    let mut tmp = lagrange_last_poly_at_z::<MC>(constants);
    tmp.mul_assign(&constants.alpha[8]);
    coeff.add_assign(&tmp);

    manager.add_assign_scaled(PolyId::R, PolyId::ZLookup, PolyForm::Monomial, coeff)?;

    Ok(())
}

fn first_coeff_for_lookup_arg_in_linearization_poly<C: Circuit<Bn256>, MC: ManagerConfigs>(
    proof: &mut Proof<Bn256, C>,
    constants: &ProverConstants<Fr>,
) -> Fr {
    let z_lookup_zw = proof.lookup_grand_product_opening_at_z_omega.unwrap();
    let w_last = constants.omega.pow(&[(MC::FULL_SLOT_SIZE - 1) as u64]);

    let mut coeff = constants.z;
    coeff.sub_assign(&w_last);
    coeff.mul_assign(&z_lookup_zw);
    coeff.mul_assign(&constants.alpha[6]);

    coeff
}

fn second_coeff_for_lookup_arg_in_linearization_poly<C: Circuit<Bn256>, MC: ManagerConfigs>(
    proof: &mut Proof<Bn256, C>,
    constants: &ProverConstants<Fr>,
) -> Fr {
    let (f, t, t_next) = (
        count_f_poly_at_z(proof, constants.eta),
        proof.lookup_t_poly_opening_at_z.unwrap(),
        proof.lookup_t_poly_opening_at_z_omega.unwrap(),
    );

    let w_last = constants.omega.pow(&[(MC::FULL_SLOT_SIZE - 1) as u64]);

    let mut beta_plus_one = constants.beta_for_lookup;
    beta_plus_one.add_assign(&Fr::one());
    let mut gamma_beta = beta_plus_one;
    gamma_beta.mul_assign(&constants.gamma_for_lookup);

    let mut coeff = t_next;
    coeff.mul_assign(&constants.beta_for_lookup);
    coeff.add_assign(&t);
    coeff.add_assign(&gamma_beta);
    let mut tmp = f;
    tmp.add_assign(&constants.gamma_for_lookup);
    coeff.mul_assign(&tmp);
    tmp = constants.z;
    tmp.sub_assign(&w_last);
    coeff.mul_assign(&tmp);
    coeff.mul_assign(&beta_plus_one);
    coeff.mul_assign(&constants.alpha[6]);
    coeff.negate();

    coeff
}

fn count_f_poly_at_z<C: Circuit<Bn256>>(proof: &mut Proof<Bn256, C>, eta: Fr) -> Fr {
    let (a, b, c, q_table_type, q_lookup_selector) = (
        proof.state_polys_openings_at_z[0],
        proof.state_polys_openings_at_z[1],
        proof.state_polys_openings_at_z[2],
        proof.lookup_table_type_poly_opening_at_z.unwrap(),
        proof.lookup_selector_poly_opening_at_z.unwrap(),
    );

    let mut f = a;
    let mut tmp = b;
    tmp.mul_assign(&eta);
    f.add_assign(&tmp);
    let mut tmp = c;
    tmp.mul_assign(&eta);
    tmp.mul_assign(&eta);
    f.add_assign(&tmp);
    let mut tmp = q_table_type;
    tmp.mul_assign(&eta);
    tmp.mul_assign(&eta);
    tmp.mul_assign(&eta);
    f.add_assign(&tmp);
    f.mul_assign(&q_lookup_selector);

    f
}

fn lagrange_last_poly_at_z<MC: ManagerConfigs>(constants: &ProverConstants<Fr>) -> Fr {
    let w_last = constants.omega.pow(&[(MC::FULL_SLOT_SIZE - 1) as u64]);

    let mut num = constants.z.pow(&[MC::FULL_SLOT_SIZE as u64]);
    num.sub_assign(&Fr::one());
    num.mul_assign(&w_last);

    let size_as_fe = Fr::from_str(&format!("{}", MC::FULL_SLOT_SIZE)).unwrap();

    let mut den = constants.z;
    den.sub_assign(&w_last);
    den.mul_assign(&size_as_fe);
    let den = den.inverse().ok_or(SynthesisError::DivisionByZero).unwrap();

    num.mul_assign(&den);
    num
}

fn evaluate_linearization_at_z<C: Circuit<Bn256>, T: Transcript<Fr>, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    proof: &mut Proof<Bn256, C>,
    transcript: &mut T,
    z: Fr,
) -> Result<(), ProvingError> {
    let handle = manager.evaluate_at(PolyId::R, z)?;
    let linearization_at_z = handle.get_result(manager)?;

    transcript.commit_field_element(&linearization_at_z);
    proof.linearization_poly_opening_at_z = linearization_at_z;
    Ok(())
}

pub fn free_useless_round_4_slots<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
) -> Result<(), ProvingError> {
    let poly_ids = [
        PolyId::QCustomSelector,
        PolyId::QMab,
        PolyId::QMac,
        PolyId::QA,
        PolyId::QB,
        PolyId::QC,
        PolyId::QD,
        PolyId::QDNext,
    ];

    let poly_form = if S::PRODUCE_SETUP {
        PolyForm::Values
    } else {
        PolyForm::Monomial
    };

    for id in poly_ids.iter() {
        manager.free_slot(*id, poly_form);
    }
    manager.free_slot(PolyId::Sigma(3), PolyForm::Monomial);

    Ok(())
}
