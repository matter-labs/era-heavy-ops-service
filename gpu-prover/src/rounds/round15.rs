use super::*;

pub fn round15<
    S: SynthesisMode,
    C: Circuit<Bn256>,
    T: Transcript<Fr>,
    MC: ManagerConfigs,
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
    proof: &mut Proof<Bn256, C>,
    constants: &mut ProverConstants<Fr>,
    transcript: &mut T,
    setup: &mut AsyncSetup,
    msm_handles_round1: Vec<MSMHandle>,
) -> Result<(), ProvingError> {

    for (i, commitment) in msm_handles_round1.into_iter().enumerate() {
        let s_commitment = commitment.get_result::<MC>(manager)?;
        // println!("GPU COMMITMENT {:?}", s_commitment);

        commit_point_as_xy::<Bn256, T>(transcript, &s_commitment);
        proof.state_polys_commitments.push(s_commitment);
    }

    constants.eta = transcript.get_challenge();

    compute_lookup_s_values(manager, assembly, constants.eta)?;
    compute_f_values_t_monomial(manager, constants.eta)?;
    compute_s_monomial_t_values(manager)?;

    let handle = manager.msm(PolyId::S)?;

    let s_commitment = handle.get_result::<MC>(manager)?;
    commit_point_as_xy::<Bn256, T>(transcript, &s_commitment);
    proof.lookup_s_poly_commitment = Some(s_commitment);

    free_useless_lookup_slots(manager)?;
    Ok(())
}

pub fn compute_f_values_t_monomial<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    eta: Fr,
) -> Result<(), ProvingError> {
    manager.rename_slot(PolyId::Col(0), PolyId::T, PolyForm::Values);
    manager.copy_from_device_to_free_device(PolyId::A, PolyId::F, PolyForm::Values)?;
    
    let mut tmp = eta;
    manager.add_assign_scaled(PolyId::T, PolyId::Col(1), PolyForm::Values, tmp)?;
    manager.add_assign_scaled(PolyId::F, PolyId::B, PolyForm::Values, tmp)?;

    tmp.mul_assign(&eta);
    manager.add_assign_scaled(PolyId::T, PolyId::Col(2), PolyForm::Values, tmp)?;
    manager.add_assign_scaled(PolyId::F, PolyId::C, PolyForm::Values, tmp)?;

    tmp.mul_assign(&eta);
    manager.add_assign_scaled(PolyId::T, PolyId::TableType, PolyForm::Values, tmp)?;
    manager.add_assign_scaled(PolyId::F, PolyId::QTableType, PolyForm::Values, tmp)?;

    manager.mul_assign(PolyId::F, PolyId::QLookupSelector, PolyForm::Values)?;
    Ok(())
}

pub fn compute_s_monomial_t_values<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
) -> Result<(), ProvingError> {
    manager.multigpu_ifft_to_free_slot(PolyId::S, false)?;
    manager.multigpu_ifft_to_free_slot(PolyId::T, false)?;
    Ok(())
}

pub fn free_useless_lookup_slots<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>
) -> Result<(), ProvingError> {
    let poly_ids = [
        PolyId::QTableType,
        PolyId::QLookupSelector,
    ];

    for id in poly_ids.iter() {
        manager.free_slot(*id, PolyForm::Values);
    }

    let poly_ids = [
        PolyId::Col(1),
        PolyId::Col(2),
        PolyId::TableType,
    ];

    for id in poly_ids.iter() {
        manager.free_slot(*id, PolyForm::Values);
    }
    Ok(())
}