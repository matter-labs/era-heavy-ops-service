use crate::simple::simple_artifact_manager::SETUP_FILE_NAME;

use super::*;
use crate::setup::ZkSyncSetup;
use prover::Prover;
use std::path::PathBuf;
use zkevm_test_harness::{
    bellman::{
        plonk::{
            better_better_cs::{cs::SynthesisModeGenerateSetup, proof::Proof},
            commitments::transcript::keccak_transcript::RollingKeccakTranscript,
        },
        worker::Worker,
    },
    sync_vm::utils::bn254_rescue_params,
};

pub fn get_artifacts_dir() -> PathBuf {
    let log_size = Prover::get_max_domain_size().trailing_zeros();
    #[cfg(feature = "legacy")]
    let artifacts_dir = format!(
        "{}/cpu/{}",
        std::env::var("ARTIFACTS_DIR").expect("ARTIFACTS_DIR"),
        log_size,
    );
    #[cfg(not(feature = "legacy"))]
    let artifacts_dir = format!(
        "{}/gpu/{}",
        std::env::var("ARTIFACTS_DIR").expect("ARTIFACTS_DIR"),
        log_size,
    );
    println!("{}", &artifacts_dir);
    std::path::Path::new(&artifacts_dir).to_owned()
}

pub fn read_circuits_from_directory(artifacts_dir: &PathBuf) -> Vec<ZkSyncCircuit> {
    let mut circuits = vec![];
    for entry in std::fs::read_dir(artifacts_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains("circuit")
            == false
        {
            continue;
        }

        let circuit = decode_circuit_from_file(&path);
        circuits.push(circuit);
    }
    circuits
}

pub fn decode_circuit_from_file(path: &PathBuf) -> ZkSyncCircuit {
    let file_name = path.file_name().unwrap().to_string_lossy();
    // println!("{}", &file_name);
    let is_json = file_name.contains(".json");
    let is_bin = file_name.contains(".bin");

    let data = std::fs::read(path).unwrap();
    if is_json {
        serde_json::from_reader(&data[..]).unwrap()
    } else if is_bin {
        bincode::deserialize(&data[..]).unwrap()
    } else {
        panic!("unknown serialization format");
    }
}

pub fn generate_setup_and_vk_for_circuit(
    prover: &mut Prover,
    circuit: &ZkSyncCircuit,
) -> (Setup, ZkSyncVerificationKey<Bn256>) {
    println!(
        "creating setup for {} {}",
        circuit.numeric_circuit_type(),
        circuit.short_description()
    );

    let log_size = (Prover::get_max_domain_size()).trailing_zeros();
    let mut setup_assembly = Prover::new_setup_assembly();
    circuit.synthesize(&mut setup_assembly).unwrap();
    setup_assembly.finalize_to_size_log_2(log_size as usize);

    let start = std::time::Instant::now();
    let setup = prover
        .create_setup_from_assembly::<ZkSyncCircuit, _>(&setup_assembly)
        .unwrap();
    println!("setup generation takes {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let vk = prover
        .inner_create_vk_from_assembly::<ZkSyncCircuit, _>(&setup_assembly)
        .unwrap();
    println!("vk generation takes {:?}", start.elapsed());
    let vk = ZkSyncVerificationKey::from_verification_key_and_numeric_type(
        circuit.numeric_circuit_type(),
        vk,
    );

    (setup, vk)
}
pub fn generate_setup_for_circuit(prover: &mut Prover, circuit: &ZkSyncCircuit) -> Setup {
    println!(
        "creating setup for {} {}",
        circuit.numeric_circuit_type(),
        circuit.short_description()
    );

    let log_size = (Prover::get_max_domain_size()).trailing_zeros();
    let mut setup_assembly = Prover::new_setup_assembly();
    circuit.synthesize(&mut setup_assembly).unwrap();
    setup_assembly.finalize_to_size_log_2(log_size as usize);

    let start = std::time::Instant::now();
    let setup = prover
        .create_setup_from_assembly::<ZkSyncCircuit, _>(&setup_assembly)
        .unwrap();
    println!("setup generation takes {:?}", start.elapsed());
    setup
}

pub fn read_vk_from_file(circuit_id: u8) -> ZkSyncVerificationKey<Bn256> {
    let artifact_dir = get_artifacts_dir();
    let vk_file_path = format!("{}/vk_{}.json", artifact_dir.to_str().unwrap(), circuit_id);
    let vk_file_path = std::path::Path::new(&vk_file_path);
    let vk_file = std::fs::File::open(&vk_file_path).unwrap();
    let vk = serde_json::from_reader(&vk_file).expect(vk_file_path.to_str().unwrap());

    vk
}

#[cfg(feature = "legacy")]
pub fn read_setup_from_file(setup_file_path: &std::path::Path) -> Setup {
    if !setup_file_path.exists() {
        panic!("{} not exists", setup_file_path.to_str().unwrap());
    }
    let setup_file = std::fs::File::open(&setup_file_path).unwrap();
    Setup::read(&setup_file).unwrap()
}

#[cfg(not(feature = "legacy"))]
pub fn read_setup_from_file(setup_file_path: &std::path::Path) -> Setup {
    use prover::NUM_LOOKUP_TABLE_NONZERO_VALUES;

    if !setup_file_path.exists() {
        panic!("{} not exists", setup_file_path.to_str().unwrap());
    }
    let setup_file = std::fs::File::open(&setup_file_path).unwrap();
    let mut setup = Setup::allocate_optimized(
        Prover::get_max_domain_size(),
        NUM_LOOKUP_TABLE_NONZERO_VALUES,
    );
    setup.read(&setup_file).unwrap();
    setup
}

pub fn save_setup_into_file(setup: &Setup, setup_file_path: &std::path::Path) {
    if setup_file_path.exists() {
        println!("{} exists", setup_file_path.to_str().unwrap());
        return;
    }
    let setup_file = std::fs::File::create(&setup_file_path).unwrap();
    #[cfg(feature = "legacy")]
    setup.write(&setup_file).unwrap();
    #[cfg(not(feature = "legacy"))]
    setup.write(&setup_file).unwrap();
}

pub fn generate_vk_for_circuit(
    prover: &mut Prover,
    circuit: &ZkSyncCircuit,
) -> ZkSyncVerificationKey<Bn256> {
    println!(
        "creating vk for {} {}",
        circuit.numeric_circuit_type(),
        circuit.short_description()
    );
    let circuit_id = circuit.numeric_circuit_type();

    let log_size = (Prover::get_max_domain_size()).trailing_zeros();
    let mut setup_assembly = Prover::new_setup_assembly();
    circuit.synthesize(&mut setup_assembly).unwrap();
    setup_assembly.finalize_to_size_log_2(log_size as usize);

    let start = std::time::Instant::now();
    let vk = prover.create_vk_from_assembly(&setup_assembly).unwrap();
    println!("vk generation takes {:?}", start.elapsed());

    ZkSyncVerificationKey::from_verification_key_and_numeric_type(circuit_id, vk)
}

pub fn save_vk_into_file(vk: &ZkSyncVerificationKey<Bn256>, vk_file_path: &Path) {
    if vk_file_path.exists() {
        return;
    }
    let vk_file = std::fs::File::create(&vk_file_path).unwrap();
    serde_json::to_writer(vk_file, &vk).unwrap();
}

pub fn read_json_circuit(artifacts_dir: &str, circuit_id: u8) -> ZkSyncCircuit {
    let circuit_file_name = format!("{}/circuit_{}.json", artifacts_dir, circuit_id);
    let circuit_file_path = std::path::Path::new(&circuit_file_name);
    let circuit_file = std::fs::File::open(circuit_file_path).expect(&circuit_file_name);
    serde_json::from_reader(circuit_file).unwrap()
}

pub fn read_binary_circuit(artifacts_dir: &str, circuit_id: u8) -> ZkSyncCircuit {
    let circuit_file_name = format!("{}/circuit_{}.bin", artifacts_dir, circuit_id);
    let circuit_file_path = std::path::Path::new(&circuit_file_name);
    let encoding = std::fs::read(&circuit_file_path).unwrap();
    bincode::deserialize(&encoding).unwrap()
}

pub fn prove_for_circuit(
    prover: &mut Prover,
    circuit: &ZkSyncCircuit,
    setup: &Setup,
) -> Proof<Bn256, ZkSyncCircuit> {
    println!(
        "create proof for {} {}",
        circuit.numeric_circuit_type(),
        circuit.short_description()
    );
    let log_size = Prover::get_max_domain_size().trailing_zeros();
    let mut assembly = Prover::new_proving_assembly();
    circuit.synthesize(&mut assembly).unwrap();
    assembly.finalize_to_size_log_2(log_size as usize);
    assert!(assembly.is_satisfied());

    let transcript_params = (bn254_rescue_params(), get_prefered_rns_params());
    let transcript_params = Some((&transcript_params.0, &transcript_params.1));

    let start = std::time::Instant::now();

    let proof = match circuit{
        OriginalZkSyncCircuit::Scheduler(_) => {
            prover.create_proof_with_proving_assembly_and_transcript::<_, RollingKeccakTranscript<Fr>>(&assembly, &setup, None)
        },
        _ => {
            prover.create_proof_with_proving_assembly_and_transcript::<_, RescueTranscriptForRecursion<'_>>(&assembly, setup, transcript_params)
        }
    };
    println!(
        "proof generation takes {:?} {} {}",
        start.elapsed(),
        circuit.numeric_circuit_type(),
        circuit.short_description()
    );

    proof.expect(&format!(
        "proof failed {} {}",
        circuit.numeric_circuit_type(),
        circuit.short_description()
    ))
}

#[cfg(feature = "legacy")]
pub fn decode_setup(circuit_id: u8, mut encoding: Box<dyn Read>) -> ZkSyncSetup {
    let setup = Setup::read(encoding).unwrap();

    ZkSyncSetup::from_setup_and_numeric_type(circuit_id, setup)
}

#[cfg(not(feature = "legacy"))]
pub fn decode_setup(circuit_id: u8, mut encoding: Box<dyn Read>) -> ZkSyncSetup {
    use std::io::Read;

    use prover::{Setup, NUM_LOOKUP_TABLE_NONZERO_VALUES};

    let mut setup = Setup::allocate_optimized(
        Prover::get_max_domain_size(),
        NUM_LOOKUP_TABLE_NONZERO_VALUES,
    );
    setup.read(encoding).unwrap();

    let wrapped_setup = ZkSyncSetup::from_setup_and_numeric_type(circuit_id, setup);
    wrapped_setup
    // Arc::new(wrapped_setup)
}

#[cfg(feature = "legacy")]
pub(crate) fn decode_setup_into_buf(setup: &mut ZkSyncSetup, mut encoding: Box<dyn Read>) {
    *setup.as_setup_mut() = Setup::read(encoding).unwrap();
}

#[cfg(not(feature = "legacy"))]
pub(crate) fn decode_setup_into_buf(setup: &mut ZkSyncSetup, encoding: Box<dyn Read>) {
    setup.as_setup_mut().read(encoding).unwrap();
}
