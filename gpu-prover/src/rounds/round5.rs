use super::*;

pub fn round5<C: Circuit<Bn256>, T: Transcript<Fr>, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    proof: &mut Proof<Bn256, C>,
    constants: &mut ProverConstants<Fr>,
    transcript: &mut T,
) -> Result<(), ProvingError> {
    get_round_5_challenges(constants, transcript);
    compute_proof_opening_at_z(manager, constants)?;
    compute_proof_opening_at_z_omega(manager, constants)?;
    free_useless_round_5_slots(manager)?;

    commit_proof_openings(manager, proof)?;

    let poly_ids = [PolyId::W, PolyId::W1];

    for id in poly_ids.iter() {
        manager.free_slot(*id, PolyForm::Monomial);
    }

    Ok(())
}

fn get_round_5_challenges<T: Transcript<Fr>>(
    constants: &mut ProverConstants<Fr>,
    transcript: &mut T,
) {
    let v = transcript.get_challenge();
    let mut tmp = v;
    constants.v = vec![];

    for _ in 0..17 {
        constants.v.push(tmp);
        tmp.mul_assign(&v);
    }
}

fn compute_proof_opening_at_z<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
) -> Result<(), ProvingError> {
    manager.rename_slot(PolyId::TPart(0), PolyId::W, PolyForm::Monomial);

    for (coeff, id) in compute_coeffs_and_ids_for_opening_at_z(constants).iter() {
        manager.add_assign_scaled(PolyId::W, *id, PolyForm::Monomial, *coeff)?;
    }

    manager.devide_monomial_by_degree_one_monomial(PolyId::W, constants.z)?;
    Ok(())
}

fn compute_coeffs_and_ids_for_opening_at_z(constants: &ProverConstants<Fr>) -> Vec<(Fr, PolyId)> {
    let mut coeffs = vec![];

    for v_power in constants.v[0..12].iter() {
        coeffs.push(*v_power);
    }

    let ids = [
        PolyId::R,
        PolyId::A,
        PolyId::B,
        PolyId::C,
        PolyId::D,
        PolyId::QMainSelector,
        PolyId::Sigma(0),
        PolyId::Sigma(1),
        PolyId::Sigma(2),
        PolyId::T,
        PolyId::QLookupSelector,
        PolyId::QTableType,
    ];

    coeffs.into_iter().zip(ids.into_iter()).collect()
}

fn compute_proof_opening_at_z_omega<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    constants: &ProverConstants<Fr>,
) -> Result<(), ProvingError> {
    manager.copy_from_device_to_free_device(PolyId::ZPerm, PolyId::W1, PolyForm::Monomial)?;
    manager.mul_constant(PolyId::W1, PolyForm::Monomial, constants.v[12])?;

    let ids = [PolyId::D, PolyId::S, PolyId::ZLookup, PolyId::T];
    for (i, id) in ids.into_iter().enumerate() {
        manager.add_assign_scaled(PolyId::W1, id, PolyForm::Monomial, constants.v[i + 13])?;
    }

    let mut zw = domain_generator::<Fr>(MC::FULL_SLOT_SIZE);
    zw.mul_assign(&constants.z);

    manager.devide_monomial_by_degree_one_monomial(PolyId::W1, zw)?;
    Ok(())
}

fn commit_proof_openings<C: Circuit<Bn256>, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    proof: &mut Proof<Bn256, C>,
) -> Result<(), ProvingError> {
    let opening_at_z_handle = manager.msm(PolyId::W)?;
    let opening_at_z_omega_handle = manager.msm(PolyId::W1)?;

    proof.opening_proof_at_z = opening_at_z_handle.get_result::<MC>(manager)?;
    proof.opening_proof_at_z_omega = opening_at_z_omega_handle.get_result::<MC>(manager)?;
    Ok(())
}

pub fn free_useless_round_5_slots<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
) -> Result<(), ProvingError> {
    let poly_ids = [
        PolyId::Sigma(0),
        PolyId::Sigma(1),
        PolyId::Sigma(2),
        PolyId::QMainSelector,
        PolyId::QLookupSelector,
        PolyId::QTableType,
        PolyId::S,
        PolyId::T,
        PolyId::ZPerm,
        PolyId::ZLookup,
        PolyId::A,
        PolyId::B,
        PolyId::C,
        PolyId::D,
        PolyId::R,
    ];

    for id in poly_ids.iter() {
        manager.free_slot(*id, PolyForm::Monomial);
    }
    Ok(())
}
