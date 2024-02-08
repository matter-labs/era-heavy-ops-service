use super::*;

#[test]
#[should_panic]
fn test_wrapper_prover() {
    // Here is a test with detailed description how to use Wrapper Prover

    // In order to initialize WrapperProver you need to have to compile bellman-cuda 
    // (https://github.com/matter-labs/era-bellman-cuda)
    // and create a global variable with its path:
    //     BELLMAN_CUDA_DIR=$PWD/../era-bellman-cuda
    // Also you need to download trusted setup (or CRS) and create similar variable:
    //     CRS_FILE="../../setup_2^24.key"
    // Trusted setup can be downloaded with
    //     wget https://universal-setup.ams3.digitaloceanspaces.com/setup_2^24.key
    let mut prover = allocate_default_prover();

    // For now we use default GPUWrapperConfigs witch is for single L4 GPU,
    // but you can chage it if you want to use 2 or 4 GPUs for one prover.
    // Also we use DEFAULT_WRAPPER_CONFIG, that could be chenged 
    // if you want to add more compression layers:
    let wrapper_config = zkevm_test_harness::proof_wrapper_utils::DEFAULT_WRAPPER_CONFIG;

    // In order to use prover we need scheduler_vk (for generating setup) 
    // and scheduler_proof (for generating proofs)
    let scheduler_vk = get_scheduler_vk_from_local_source();
    let scheduler_proof = get_scheduler_proof_from_local_source();

    // Also we have two bad ones for negative examples
    let bad_scheduler_vk = get_bad_scheduler_vk();
    let bad_scheduler_proof = get_bad_scheduler_proof();

    // There are two main function:
    //     generate_setup_data
    //     generate_proofs

    // The first one generates setup needed for proving from scheduler vk:
    prover.generate_setup_data(scheduler_vk.into_inner()).unwrap();

    // The second one generates proofs from scheduler proof:
    prover.generate_proofs(scheduler_proof.clone().into_inner()).unwrap();

    // We can get final Wrapper proof and vk with:
    let snark_vk = prover.get_wrapper_vk().unwrap();
    let snark_proof = prover.get_wrapper_proof().unwrap();

    // You can also get a link to source with intermediate vks and proofs:
    let inner_source = prover.source();
    let wrapper_type = wrapper_config.get_wrapper_type();
    let _compression_for_wrapper_vk = inner_source
        .get_compression_for_wrapper_vk(wrapper_type)
        .unwrap();

    // And verify correctness:
    let is_valid = verify::<_, _, RollingKeccakTranscript<Fr>>(
        &snark_vk, 
        &snark_proof, 
        None
    ).unwrap();
    assert!(is_valid);

    // We also can generate proofs multiple times without regenerating setup:
    prover.generate_proofs(scheduler_proof.clone().into_inner()).unwrap();

    // If generating proof fails the WrapperError is returned:
    let result = prover.generate_proofs(bad_scheduler_proof.into_inner());
    assert!(result.is_err());

    // The same happens when generating setup fails:
    let result = prover.generate_setup_data(bad_scheduler_vk.into_inner());
    assert!(result.is_err());

    // Note that after prover is just created or after generating setup fails there 
    // is no valid setup inside prover:
    assert!(!prover.setup_is_ready());

    // In this case generating proofs will panic:
    prover.generate_proofs(scheduler_proof.into_inner()).unwrap();
}

#[test]
fn test_vk_generation() {
    let mut prover = allocate_default_prover();

    let scheduler_vk = get_scheduler_vk_from_local_source();
    prover.generate_setup_data(scheduler_vk.into_inner()).unwrap();
    
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
    WrapperProver::<GPUWrapperConfigs>::new(&crs, wrapper_config).unwrap()
}

fn get_scheduler_vk_from_local_source() -> ZkSyncRecursionLayerVerificationKey {
    let source = LocalFileDataSource;
    source.get_recursion_layer_vk(ZkSyncRecursionLayerStorageType::SchedulerCircuit as u8)
        .expect("There should be scheduler vk in local storage")
}

fn get_bad_scheduler_vk() -> ZkSyncRecursionLayerVerificationKey {
    let mut bad_scheduler_vk = get_scheduler_vk_from_local_source().into_inner();
    bad_scheduler_vk.setup_merkle_tree_cap.pop();
    ZkSyncRecursionLayerStorage::SchedulerCircuit(bad_scheduler_vk)
}

fn get_scheduler_proof_from_local_source() -> ZkSyncRecursionLayerProof {
    let source = LocalFileDataSource;
    source.get_scheduler_proof()
        .expect("There should be scheduler proof in local storage")
}

fn get_bad_scheduler_proof() -> ZkSyncRecursionLayerProof {
    let mut bad_scheduler_proof = get_scheduler_proof_from_local_source().into_inner();
    bad_scheduler_proof.stage_2_oracle_cap.pop();
    ZkSyncRecursionLayerStorage::SchedulerCircuit(bad_scheduler_proof)
}
