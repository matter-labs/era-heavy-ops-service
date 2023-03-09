#![feature(get_mut_unchecked)]
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::sync::Arc;

use api::Prover;
use zkevm_test_harness::franklin_crypto::bellman::plonk::better_better_cs::cs::{
    Circuit, PlonkCsWidth4WithNextStepAndCustomGatesParams, TrivialAssembly,
};
use zkevm_test_harness::franklin_crypto::bellman::plonk::better_better_cs::proof::Proof;
use zkevm_test_harness::franklin_crypto::bellman::plonk::better_better_cs::setup::VerificationKey;
use zkevm_test_harness::franklin_crypto::bellman::plonk::cs::variable::Index;
use zkevm_test_harness::franklin_crypto::bellman::{
    self,
    plonk::better_better_cs::gates::selector_optimized_with_d_next::SelectorOptimizedWidth4MainGateWithDNext,
};

use non_trivial_circuit::NonTrivialCircuit;
use rand::thread_rng;
use zkevm_test_harness::bellman::plonk::commitments::transcript::keccak_transcript::RollingKeccakTranscript;
use zkevm_test_harness::bellman::worker::Worker;
use zkevm_test_harness::franklin_crypto::bellman::bn256::{Bn256, Fr};
use zkevm_test_harness::witness::oracle::VmWitnessOracle;

mod circuit;
mod non_trivial_circuit;

const DEFAULT_CRS_FILE: &str = "/tmp/circuit.crs";

enum Proving<'a> {
    Dummy,
    File,
    Prover(&'a mut Prover),
}

#[derive(Clone, Copy)]
enum Encoding {
    Json,
    Binary,
}
#[test]
fn test_single_circuit_prover() {
    let circuit = NonTrivialCircuit::<Bn256, SelectorOptimizedWidth4MainGateWithDNext>::new();
    let mut prover = Prover::new_with_dummy_crs();

    let mut assembly = Prover::new_assembly();
    circuit.synthesize(&mut assembly).unwrap();
    assembly.is_satisfied();
    assembly.finalize();
    dbg!((assembly.n() + 1).trailing_zeros());
    let proof = prover.create_proof_with_assembly_and_transcript::<NonTrivialCircuit<Bn256, SelectorOptimizedWidth4MainGateWithDNext>, RollingKeccakTranscript<Fr>>(&assembly, None).expect("create proof");
    dbg!(proof);
}

#[test]
fn test_parallel_provers() {
    let prover = Prover::new_with_dummy_crs();
    let prover = Arc::new(prover);
    let mut handles = vec![];
    for idx in 0..2 {
        let prover = prover.clone();
        let handle = std::thread::spawn(move || {
            let circuit =
                NonTrivialCircuit::<Bn256, SelectorOptimizedWidth4MainGateWithDNext>::new();
            let mut prover = prover;
            let prover = unsafe { Arc::get_mut_unchecked(&mut prover) };
            let mut trivial_assembly = Prover::new_assembly();
            circuit.synthesize(&mut trivial_assembly).unwrap();
            trivial_assembly.finalize();

            let proof = prover.create_proof_with_assembly_and_transcript::<NonTrivialCircuit<Bn256, SelectorOptimizedWidth4MainGateWithDNext>, RollingKeccakTranscript<Fr>>(&trivial_assembly, None).expect("create proof");
            println!("proof generated for circuit {}", idx);
        });
        handles.push(handle);
    }
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_single_prover_with_json_encoding() {
    println!("using json file");
    run_single_prover_with_encoded_data(Encoding::Json)
}

#[test]
fn test_single_prover_with_binary_encoding() {
    println!("using binary file");
    run_single_prover_with_encoded_data(Encoding::Binary)
}

#[test]
fn test_parallel_prover_with_json_encoding() {
    println!("using json file");
    run_parallel_provers_with_circuit_encoding(Encoding::Json)
}

#[test]
fn test_parallel_prover_with_binary_encoding() {
    println!("using binary file");
    run_parallel_provers_with_circuit_encoding(Encoding::Binary)
}

fn run_single_prover_with_encoded_data(encoding: Encoding) {
    let encoding_file_path = std::env::var("CIRCUIT_FILE").unwrap();
    let circuit_file = std::fs::File::open(encoding_file_path).expect("circuit encoding file");

    let vk_file_path = std::env::var("VK_FILE").expect("env variable for path of vk encoding file");
    let vk_file = std::fs::File::open(&vk_file_path).expect("open vk file");

    let (circuit, vk): (ZkSyncCircuit, VerificationKey<Bn256, ZkSyncCircuit>) = match encoding {
        Encoding::Json => {
            let vk = serde_json::from_reader(&vk_file).expect("deserialize vk");
            let circuit = serde_json::from_reader(&circuit_file).expect("deserialize circuit");
            (circuit, vk)
        }
        Encoding::Binary => {
            let mut reader = BufReader::new(circuit_file);
            let mut circuit_buffer = Vec::new();
            reader.read_to_end(&mut circuit_buffer).unwrap();
            let circuit = bincode::deserialize(&circuit_buffer).expect("deserialize circuit");

            let vk: VerificationKey<Bn256, ZkSyncCircuit> =
                VerificationKey::read(&vk_file).unwrap();
            (circuit, vk)
        }
    };
    println!("circuit and vk are deserialized");
    let proving_key_file_path =
        std::env::var("CRS_FILE").expect("env variable for path of encoding file");
    let crs_path = std::path::Path::new(&proving_key_file_path);
    let mut prover = Prover::new_with_crs(crs_path);
    println!("prover initialized");
    generate_proof(Proving::Prover(&mut prover), &circuit, Some(&vk))
}

fn run_parallel_provers_with_circuit_encoding(encoding: Encoding) {
    let encoding_file_dir = std::env::var("ENCODING_FILE_DIR")
        .expect("env variable for directory path of encoding files");
    let proving_key_file_dir =
        std::env::var("CRS_FILE").expect("env variable for directory path of encoding files");
    let crs_path = std::path::Path::new(&proving_key_file_dir);
    let prover = Prover::new_with_crs(crs_path);
    let prover = Arc::new(prover);

    let start = std::time::Instant::now();

    let mut handles = vec![];
    for circuit_id in 6..8 {
        let prover = prover.clone();
        let encoding_file_dir = encoding_file_dir.clone();
        let handle = std::thread::spawn(move || {
            println!("deserializing circuit {}", circuit_id);
            let circuit_file_path =
                format!("{}/circuit_encoding_{}.json", encoding_file_dir, circuit_id);
            let circuit_file =
                std::fs::File::open(circuit_file_path).expect("circuit encoding file");

            let circuit: ZkSyncCircuit = match encoding {
                Encoding::Json => {
                    serde_json::from_reader(&circuit_file).expect("deserialize circuit")
                }
                Encoding::Binary => {
                    let mut reader = BufReader::new(circuit_file);
                    let mut buffer = Vec::new();
                    reader.read_to_end(&mut buffer).unwrap();
                    bincode::deserialize(&buffer).expect("deserialize circuit")
                }
            };
            println!("deserializing vk {}", circuit_id);
            let vk_file_path =
                format!("{}/basic_circuit_vk_{}.json", encoding_file_dir, circuit_id);
            let vk_file = std::fs::File::open(vk_file_path).expect("vk encoding file");
            let vk = match encoding {
                Encoding::Json => serde_json::from_reader(&vk_file).expect("deserialize circuit"),
                Encoding::Binary => {
                    let mut reader = BufReader::new(vk_file);
                    let mut buffer = Vec::new();
                    reader.read_to_end(&mut buffer).unwrap();
                    bincode::deserialize(&buffer).expect("deserialize circuit")
                }
            };

            let mut prover = prover;
            let prover = unsafe { Arc::get_mut_unchecked(&mut prover) };
            generate_proof(Proving::Prover(prover), &circuit, Some(&vk));
            println!("proof is done for circuit {}", circuit_id);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
    println!("2 proofs takes {:?}", start.elapsed());
}

fn generate_proof<C: Circuit<Bn256>>(
    proving: Proving,
    circuit: &C,
    vk: Option<&VerificationKey<Bn256, C>>,
) {
    println!("synthesizing");
    let mut assembly = Prover::new_assembly();
    circuit.synthesize(&mut assembly).expect("synthesis");
    assert!(assembly.is_satisfied());
    assembly.finalize();
    println!("assembly finalized");
    println!("assembly input variables {}", assembly.num_inputs);
    println!("assembly aux variables {}", assembly.num_aux);
    println!("lookup table queries {}", assembly.num_table_lookups);
    for (table_name, individual_table) in assembly.individual_table_entries.iter() {
        if individual_table.len() == 0 {
            continue;
        }
        // println!("{} table queries {}", table_name, individual_table[0].len());
    }
    println!(
        "assembly aux storage {}",
        assembly.aux_storage.state_map.len()
    );
    println!(
        "assembly setup storage {}",
        assembly.aux_storage.setup_map.len()
    );
    let proof = match proving {
        Proving::Dummy => {
            let mut prover = Prover::new_with_dummy_crs();

            prover
                .create_proof_with_assembly_and_transcript::<_, RollingKeccakTranscript<Fr>>(
                    &assembly, None,
                )
                .expect("create proof with assembly")
        }
        Proving::File => {
            let crs_file = std::env::var(DEFAULT_CRS_FILE).expect("proving key env var");
            let crs_path = std::path::Path::new(&crs_file);
            let mut prover = Prover::new_with_crs(crs_path);

            prover
                .create_proof_with_assembly_and_transcript::<_, RollingKeccakTranscript<Fr>>(
                    &assembly, None,
                )
                .expect("create proof with assembly")
        }
        Proving::Prover(prover) => prover
            .create_proof_with_assembly_and_transcript::<_, RollingKeccakTranscript<Fr>>(
                &assembly, None,
            )
            .expect("create proof with assembly"),
    };
    match vk {
        Some(vk) => {
            assert_eq!(vk.n, proof.n);
            assert_eq!(vk.num_inputs, proof.inputs.len());
            let valid_proof =
                Prover::verify_proof::<C, RollingKeccakTranscript<Fr>>(&proof, vk, None).unwrap();
            assert!(valid_proof);
        }
        None => todo!(),
    }

    println!("done");
}

pub type ZkSyncCircuit =
    zkevm_test_harness::abstract_zksync_circuit::concrete_circuits::ZkSyncCircuit<
        Bn256,
        VmWitnessOracle<Bn256>,
    >;

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

#[test]
fn run_circuit_statistics() {
    let encoding = Encoding::Json;

    let artifacts_dir =
        std::env::var("ARTIFACTS_DIR").expect("env variable for directory path of encoding files");

    let circuits = read_circuits_from_directory(&PathBuf::from(&artifacts_dir));
    for circuit in circuits.iter() {
        let circuit_id = circuit.numeric_circuit_type();

        if circuit_id != 3 {
            continue;
        }

        let mut assembly = TrivialAssembly::<
            Bn256,
            PlonkCsWidth4WithNextStepAndCustomGatesParams,
            SelectorOptimizedWidth4MainGateWithDNext,
        >::new();
        circuit.synthesize(&mut assembly).unwrap();
        println!("============================================");
        println!("CIRCUIT {}", circuit_id);
        println!("Assembly has {} input gates", assembly.num_input_gates);
        println!("Assembly has {} aux gates", assembly.num_aux_gates);
        println!("Assembly has {} total gates", assembly.n());
        println!(
            "Assembly {} has input assignments(Fr)",
            assembly.input_assingments.len()
        );
        println!(
            "Assembly {} has aux assignments(Fr)",
            assembly.aux_assingments.len()
        );

        for (poly_id, variables) in assembly.inputs_storage.state_map.iter() {
            let poly_id = match poly_id{
                bellman::plonk::better_better_cs::data_structures::PolyIdentifier::VariablesPolynomial(idx) =>idx,
                _ => unreachable!(),
            };
            println!(
                "state poly {:?} has {} input variables(Variable)",
                poly_id,
                variables.len()
            );
        }
        println!("====");
        for (poly_id, variables) in assembly.inputs_storage.setup_map.iter() {
            let poly_id = match poly_id{
                bellman::plonk::better_better_cs::data_structures::PolyIdentifier::GateSetupPolynomial(_, idx) =>idx,
                _ => unreachable!(),
            };
            println!(
                "setup poly {} has {} input coefficients(Fr)",
                poly_id,
                variables.len()
            );
        }
        println!("====");
        for (poly_id, variables) in assembly.aux_storage.state_map.iter() {
            let poly_id = match poly_id{
                bellman::plonk::better_better_cs::data_structures::PolyIdentifier::VariablesPolynomial(idx) =>idx,
                _ => unreachable!(),
            };
            println!(
                "state poly {:?} has {} aux variables(Variable)",
                poly_id,
                variables.len()
            );
        }
        println!("====");
        for (poly_id, variables) in assembly.aux_storage.setup_map.iter() {
            let poly_id = match poly_id{
                bellman::plonk::better_better_cs::data_structures::PolyIdentifier::GateSetupPolynomial(_, idx) =>idx,
                _ => unreachable!(),
            };
            println!(
                "setup poly {}  has {} aux coefficients(Fr)",
                poly_id,
                variables.len()
            );
        }
        println!("====");
        println!(
            "Assembly {} has loookup queries (Fr)",
            assembly.num_table_lookups
        );

        for (table, entries) in assembly.individual_table_entries {
            println!("{} table has {} entries(Fr)", table, entries.len())
        }
        for (table, selectors) in assembly.table_selectors.iter() {
            println!("table {} has {} selectors(Bit) ", table, selectors.len())
        }
        println!("====");
        println!(
            "table id poly has {} items(Fr)",
            assembly.table_ids_poly.len()
        );
        println!("============================================");
    }
}

#[test]
fn run_generate_vk_and_save_file() {
    let circuit = NonTrivialCircuit::<Bn256, SelectorOptimizedWidth4MainGateWithDNext>::new();
    let crs_file =
        std::env::var("CRS_FILE").expect("env variable for directory path of encoding files");
    let crs_path = std::path::Path::new(&crs_file);

    let worker = Prover::new_worker(None);

    let mut prover = Prover::new_with_crs(&crs_path);

    let mut assembly = Prover::new_assembly();
    circuit.synthesize(&mut assembly).unwrap();
    assert!(assembly.is_satisfied());
    assembly.finalize();

    let start = std::time::Instant::now();
    let proof = prover.create_proof_with_assembly_and_transcript::<NonTrivialCircuit<Bn256, SelectorOptimizedWidth4MainGateWithDNext>, RollingKeccakTranscript<Fr>>(&assembly, None).expect("create proof");
    println!("proof generation takes {:?}", start.elapsed());

    println!("generating vk");
    let start = std::time::Instant::now();
    let vk = prover.create_vk_from_assembly::<NonTrivialCircuit<_, SelectorOptimizedWidth4MainGateWithDNext>, _>(&assembly).expect("create vk");
    println!("vk generation takes {:?}", start.elapsed());

    assert!(Prover::verify_proof::<_, RollingKeccakTranscript<Fr>>(&proof, &vk, None).unwrap());
    #[cfg(feature = "legacy")]
    let vk_file_name = format!("cpu.key");
    #[cfg(not(feature = "legacy"))]
    let vk_file_name = format!("gpu.key");

    #[cfg(feature = "legacy")]
    let proof_file_name = format!("cpu.proof");
    #[cfg(not(feature = "legacy"))]
    let proof_file_name = format!("gpu.proof");

    let vk_file =
        std::fs::File::create(std::path::Path::new(&vk_file_name)).expect("create vk file");
    vk.write(&vk_file).expect("write vk into file");

    let proof_file =
        std::fs::File::create(std::path::Path::new(&proof_file_name)).expect("create vk file");
    proof.write(&proof_file).expect("proof vk into file");

    let valid = Prover::verify_proof::<_, RollingKeccakTranscript<Fr>>(&proof, &vk, None)
        .expect("verify proof");
    assert!(valid);
}

#[test]
fn compare_vks() {
    let vk_file_cpu = std::fs::File::open("cpu.key").unwrap();
    let vk_file_gpu = std::fs::File::open("gpu.key").unwrap();

    let vk_cpu: VerificationKey<
        Bn256,
        NonTrivialCircuit<Bn256, SelectorOptimizedWidth4MainGateWithDNext>,
    > = VerificationKey::read(&vk_file_cpu).expect("read vk cpu");
    let vk_gpu: VerificationKey<
        Bn256,
        NonTrivialCircuit<Bn256, SelectorOptimizedWidth4MainGateWithDNext>,
    > = VerificationKey::read(&vk_file_gpu).expect("read vk gpu");

    assert_eq!(vk_cpu.n, vk_gpu.n);
    assert_eq!(vk_cpu.num_inputs, vk_gpu.num_inputs);
    assert_eq!(vk_cpu.state_width, vk_gpu.state_width);

    assert_eq!(vk_cpu.gate_setup_commitments, vk_gpu.gate_setup_commitments);
    assert_eq!(
        vk_cpu.gate_selectors_commitments,
        vk_gpu.gate_selectors_commitments
    );
    assert_eq!(
        vk_cpu.permutation_commitments,
        vk_gpu.permutation_commitments
    );

    assert_eq!(
        vk_cpu.total_lookup_entries_length,
        vk_gpu.total_lookup_entries_length
    );
    assert_eq!(
        vk_cpu.lookup_selector_commitment,
        vk_gpu.lookup_selector_commitment
    );
    assert_eq!(
        vk_cpu.lookup_tables_commitments,
        vk_gpu.lookup_tables_commitments
    );
    assert_eq!(
        vk_cpu.lookup_table_type_commitment,
        vk_gpu.lookup_table_type_commitment
    );

    assert_eq!(vk_cpu.non_residues, vk_gpu.non_residues);
    assert_eq!(vk_cpu.g2_elements, vk_gpu.g2_elements);

    let proof_file_cpu = std::fs::File::open("cpu.proof").unwrap();
    let proof_file_gpu = std::fs::File::open("gpu.proof").unwrap();

    let proof_cpu: Proof<
        Bn256,
        NonTrivialCircuit<Bn256, SelectorOptimizedWidth4MainGateWithDNext>,
    > = Proof::read(&proof_file_cpu).expect("read vk cpu");
    let proof_gpu: Proof<
        Bn256,
        NonTrivialCircuit<Bn256, SelectorOptimizedWidth4MainGateWithDNext>,
    > = Proof::read(&proof_file_gpu).expect("read vk gpu");

    // compare each field individually
    assert_eq!(proof_cpu.n, proof_gpu.n);
    assert_eq!(proof_cpu.inputs, proof_gpu.inputs);
    assert_eq!(
        proof_cpu.state_polys_commitments,
        proof_gpu.state_polys_commitments
    );
    assert_eq!(
        proof_cpu.copy_permutation_grand_product_commitment,
        proof_gpu.copy_permutation_grand_product_commitment
    );
    assert_eq!(
        proof_cpu.lookup_s_poly_commitment,
        proof_gpu.lookup_s_poly_commitment
    );
    assert_eq!(
        proof_cpu.lookup_grand_product_commitment,
        proof_gpu.lookup_grand_product_commitment
    );
    assert_eq!(
        proof_cpu.quotient_poly_parts_commitments,
        proof_gpu.quotient_poly_parts_commitments
    );
    assert_eq!(
        proof_cpu.state_polys_openings_at_z,
        proof_gpu.state_polys_openings_at_z
    );
    assert_eq!(
        proof_cpu.state_polys_openings_at_dilations,
        proof_gpu.state_polys_openings_at_dilations
    );
    assert_eq!(
        proof_cpu.gate_setup_openings_at_z,
        proof_gpu.gate_setup_openings_at_z
    );
    assert_eq!(
        proof_cpu.gate_selectors_openings_at_z,
        proof_gpu.gate_selectors_openings_at_z
    );
    assert_eq!(
        proof_cpu.copy_permutation_polys_openings_at_z,
        proof_gpu.copy_permutation_polys_openings_at_z
    );
    assert_eq!(
        proof_cpu.copy_permutation_grand_product_opening_at_z_omega,
        proof_gpu.copy_permutation_grand_product_opening_at_z_omega
    );
    assert_eq!(
        proof_cpu.lookup_s_poly_opening_at_z_omega,
        proof_gpu.lookup_s_poly_opening_at_z_omega
    );
    assert_eq!(
        proof_cpu.lookup_grand_product_opening_at_z_omega,
        proof_gpu.lookup_grand_product_opening_at_z_omega
    );
    assert_eq!(
        proof_cpu.lookup_t_poly_opening_at_z,
        proof_gpu.lookup_t_poly_opening_at_z
    );
    assert_eq!(
        proof_cpu.lookup_t_poly_opening_at_z_omega,
        proof_gpu.lookup_t_poly_opening_at_z_omega
    );
    assert_eq!(
        proof_cpu.lookup_selector_poly_opening_at_z,
        proof_gpu.lookup_selector_poly_opening_at_z
    );
    assert_eq!(
        proof_cpu.lookup_table_type_poly_opening_at_z,
        proof_gpu.lookup_table_type_poly_opening_at_z
    );
    assert_eq!(
        proof_cpu.quotient_poly_opening_at_z,
        proof_gpu.quotient_poly_opening_at_z
    );
    assert_eq!(
        proof_cpu.linearization_poly_opening_at_z,
        proof_gpu.linearization_poly_opening_at_z
    );
    assert_eq!(proof_cpu.opening_proof_at_z, proof_gpu.opening_proof_at_z);
    assert_eq!(
        proof_cpu.opening_proof_at_z_omega,
        proof_gpu.opening_proof_at_z_omega
    );

    assert!(Prover::verify_proof::<
        NonTrivialCircuit<Bn256, SelectorOptimizedWidth4MainGateWithDNext>,
        RollingKeccakTranscript<_>,
    >(&proof_cpu, &vk_cpu, None)
    .unwrap());

    assert!(Prover::verify_proof::<
        NonTrivialCircuit<Bn256, SelectorOptimizedWidth4MainGateWithDNext>,
        RollingKeccakTranscript<_>,
    >(&proof_gpu, &vk_cpu, None)
    .unwrap());

    assert!(Prover::verify_proof::<
        NonTrivialCircuit<Bn256, SelectorOptimizedWidth4MainGateWithDNext>,
        RollingKeccakTranscript<_>,
    >(&proof_cpu, &vk_gpu, None)
    .unwrap());

    assert!(Prover::verify_proof::<
        NonTrivialCircuit<Bn256, SelectorOptimizedWidth4MainGateWithDNext>,
        RollingKeccakTranscript<_>,
    >(&proof_gpu, &vk_gpu, None)
    .unwrap());
}
