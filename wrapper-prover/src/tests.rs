use super::*;

#[test]
fn test_wrapper_prover() {
    let mut prover = allocate_default_prover();

    let scheduler_vk = get_scheduler_vk_from_local_source();
    let scheduler_proof = get_scheduler_proof_from_local_source();

    prover.generate_setup_data(scheduler_vk.into_inner());
    prover.generate_proofs(scheduler_proof.into_inner());

    let snark_vk = prover.get_wrapper_vk();
    let snark_proof = prover.get_wrapper_proof();

    let is_valid =
        verify::<_, _, RollingKeccakTranscript<Fr>>(&snark_vk, &snark_proof, None).unwrap();
    assert!(is_valid);
}

#[test]
fn test_vk_generation() {
    let mut prover = allocate_default_prover();

    let scheduler_vk = get_scheduler_vk_from_local_source();
    prover.generate_setup_data(scheduler_vk.into_inner());
    
    let scheduler_vk = get_scheduler_vk_from_local_source();
    use zkevm_test_harness::proof_wrapper_utils::get_wrapper_setup_and_vk_from_scheduler_vk;
    let wrapper_config = zkevm_test_harness::proof_wrapper_utils::DEFAULT_WRAPPER_CONFIG;

    let snark_vk_from_prover = prover.get_wrapper_vk();
    let (_, snark_vk) = get_wrapper_setup_and_vk_from_scheduler_vk(scheduler_vk, wrapper_config);

    dbg!(snark_vk.into_inner(), snark_vk_from_prover);
}

/// You should have two global variables
/// 1. Your trusted setup location: CRS_FILE="../../setup_2^24.key"
/// 2. Bellman-cuda location: BELLMAN_CUDA_DIR=$PWD/../era-bellman-cuda
fn allocate_default_prover() -> WrapperProver<GPUWrapperConfigs> {
    let crs = get_trusted_setup();
    let wrapper_config = zkevm_test_harness::proof_wrapper_utils::DEFAULT_WRAPPER_CONFIG;
    WrapperProver::<GPUWrapperConfigs>::new(&crs, wrapper_config)
}

fn get_scheduler_vk_from_local_source() -> ZkSyncRecursionLayerVerificationKey {
    let source = LocalFileDataSource;
    source.get_recursion_layer_vk(ZkSyncRecursionLayerStorageType::SchedulerCircuit as u8)
        .expect("There should be scheduler vk in local storage")
}

fn get_scheduler_proof_from_local_source() -> ZkSyncRecursionLayerProof {
    let source = LocalFileDataSource;
    source.get_scheduler_proof()
        .expect("There should be scheduler proof in local storage")
}
