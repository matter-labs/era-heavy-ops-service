use api::bellman::plonk::better_better_cs::cs::SynthesisModeTesting;
use api::bellman::plonk::better_better_cs::cs::{
    Circuit, PlonkCsWidth4WithNextStepAndCustomGatesParams, TrivialAssembly,
};
use api::bellman::plonk::better_better_cs::proof::Proof;
use api::bellman::plonk::better_better_cs::setup::VerificationKey;
use api::bellman::plonk::cs::variable::Index;
use api::bellman::{
    self,
    plonk::better_better_cs::gates::selector_optimized_with_d_next::SelectorOptimizedWidth4MainGateWithDNext,
};
use api::Prover;

use api::bellman::bn256::{Bn256, Fr};
use api::bellman::plonk::commitments::transcript::keccak_transcript::RollingKeccakTranscript;
use api::bellman::worker::Worker;
use non_trivial_circuit::NonTrivialCircuit;
use rand::thread_rng;

mod non_trivial_circuit;

type TestCircuit = NonTrivialCircuit<Bn256, SelectorOptimizedWidth4MainGateWithDNext>;
type TestTranscript = RollingKeccakTranscript<Fr>;

#[test]
fn test_prover_without_precomputed_setup() {
    let mut prover = if let Ok(crs_file) = std::env::var("CRS_FILE") {
        Prover::new()
    } else {
        Prover::new_with_dummy_crs()
    };

    let circuit = TestCircuit::new();
    let mut assembly = Prover::new_assembly();
    circuit.synthesize(&mut assembly).unwrap();
    assembly.is_satisfied();
    // pad to the full size
    assembly.finalize_to_size_log_2(Prover::get_max_domain_size_log());
    let proof = prover
        .create_proof_with_assembly_and_transcript::<TestCircuit, TestTranscript>(&assembly, None)
        .expect("create proof");
    // create verification key
    let vk = prover
        .create_vk_from_assembly(&assembly)
        .expect("create vk");

    assert!(Prover::verify_proof::<TestCircuit, TestTranscript>(&proof, &vk, None).unwrap());
}

#[test]
fn test_prover_with_precomputed_setup() {
    let mut prover = if let Ok(crs_file) = std::env::var("CRS_FILE") {
        Prover::new()
    } else {
        Prover::new_with_dummy_crs()
    };

    let circuit = TestCircuit::new();
    let mut assembly = Prover::new_assembly();
    circuit.synthesize(&mut assembly).unwrap();
    assembly.is_satisfied();
    assembly.finalize_to_size_log_2(Prover::get_max_domain_size_log());

    // create specialized setup for gpu
    let proof = prover
        .create_setup_from_assembly::<TestCircuit, SynthesisModeTesting>(&assembly)
        .expect("create setup");

    // create proof
    let proof = prover
        .create_proof_with_assembly_and_transcript::<TestCircuit, TestTranscript>(&assembly, None)
        .expect("create proof");

    // create verification key
    let vk = prover
        .create_vk_from_assembly(&assembly)
        .expect("create vk");

    // verify proof
    assert!(Prover::verify_proof::<TestCircuit, TestTranscript>(&proof, &vk, None).unwrap());
}
