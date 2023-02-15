use std::{
    collections::HashMap,
    io::{Cursor, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    str::FromStr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{channel, Receiver, Sender, SyncSender},
    },
    time::Duration,
};

use super::*;

use prover::{ProvingAssembly, SetupAssembly};
use rand::Rng;
use zkevm_test_harness::{
    bellman::plonk::{
        better_better_cs::{
            cs::SynthesisModeProve, data_structures::PolyIdentifier,
            lookup_tables::table_id_from_string, setup::VerificationKey,
        },
        commitments::transcript::keccak_transcript::RollingKeccakTranscript,
    },
    sync_vm::utils::bn254_rescue_params,
};

use crate::{
    remote_synth::{
        calculate_serialization_capacity_for_proving_assembly, custom_assembly_deserialization,
        custom_assembly_serialization, deserialize_job, run_remote_synthesizer, serialize_job,
        EncodedArtifactSender,
    },
    run_prover::{
        create_prover_instances, run_prover_with_local_synthesizer,
        run_prover_with_remote_synthesizer,
    },
    simple::{
        simple_artifact_manager::{SimpleArtifactManager, SETUP_FILE_NAME},
        simple_job_manager::{JobState, SimpleJobManager, SimpleJobReporter},
    },
};

use super::utils::*;

pub(crate) struct TestingParams;

impl Params for TestingParams {
    fn number_of_parallel_synthesis(&self) -> u8 {
        3
    }

    fn number_of_setup_slots(&self) -> u8 {
        4
    }

    fn polling_duration(&self) -> Duration {
        Duration::from_millis(1)
    }
}

pub(crate) struct TestingParamsForSpecialized;

impl Params for TestingParamsForSpecialized {
    fn number_of_parallel_synthesis(&self) -> u8 {
        8
    }

    fn number_of_setup_slots(&self) -> u8 {
        3
    }

    fn polling_duration(&self) -> Duration {
        Duration::from_millis(1)
    }
}

struct SimpleRemoteSynthesizer {
    sender: std::sync::mpsc::Sender<Box<dyn Read + Send + Sync>>,
    receiver: std::sync::mpsc::Receiver<Box<dyn Read + Send + Sync>>,
}
unsafe impl Send for SimpleRemoteSynthesizer {}

impl RemoteSynthesizer for SimpleRemoteSynthesizer {
    fn try_next(&mut self) -> Option<Box<dyn Read + Send + Sync>> {
        if let Ok(buf) = self.receiver.try_recv() {
            Some(buf)
        } else {
            None
        }
    }
}

impl SimpleRemoteSynthesizer {
    fn new(assembly_encodings: Vec<Box<dyn Read + Send + Sync>>) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        for encoding in assembly_encodings {
            sender.send(encoding).unwrap();
        }
        Self { sender, receiver }
    }
}

#[test]
fn test_prover_service_with_shuffled_circuits() {
    assert!(std::env::var("CRS_FILE").is_ok());
    let artifacts_dir = get_artifacts_dir();

    let circuits = read_circuits_from_directory(&artifacts_dir);
    assert!(!circuits.is_empty());
    let mut selected_circuits = vec![];
    for circuit in circuits {
        selected_circuits.push(circuit.clone());
    }

    rand::thread_rng().shuffle(&mut selected_circuits);

    let jobs: Vec<(usize, ZkSyncCircuit, JobState)> = selected_circuits
        .into_iter()
        .enumerate()
        .map(|(idx, c)| (idx, c, JobState::Created(idx)))
        .collect();
    let jobs = Arc::new(Mutex::new(jobs));
    let job_manager = SimpleJobManager::new(jobs.clone());
    let job_reporter = SimpleJobReporter::new(jobs);
    let artifact_manager = SimpleArtifactManager;

    run_prover_with_local_synthesizer(
        artifact_manager,
        job_manager,
        job_reporter,
        None,
        TestingParams,
    );
}

#[test]
fn test_prover_service_with_external_synthesizer_and_shuffled_circuits() {
    assert!(std::env::var("CRS_FILE").is_ok());
    let artifacts_dir = get_artifacts_dir();

    let circuit_ids = vec![3, 5, 8, 10];
    let circuits = read_circuits_from_directory(&artifacts_dir);
    assert!(!circuits.is_empty());

    let mut selected_circuits = vec![];
    for circuit in circuits {
        if !circuit_ids.contains(&circuit.numeric_circuit_type()) {
            continue;
        }

        selected_circuits.push(circuit.clone());
    }
    rand::thread_rng().shuffle(&mut selected_circuits);
    let mut assembly_encodings = vec![];

    for circuit in selected_circuits.iter() {
        let circuit_id = circuit.numeric_circuit_type();
        let assembly_file_path = format!(
            "{}/{}_{}.bin",
            artifacts_dir.to_str().unwrap(),
            "assembly",
            circuit_id
        );
        let assembly_file_path = std::path::Path::new(&assembly_file_path);
        let assembly_file = std::fs::File::open(assembly_file_path).unwrap();
        let assembly_encoding = Box::new(assembly_file) as Box<dyn Read + Send + Sync>;
        assembly_encodings.push(assembly_encoding);
    }

    let external_synthesizer = SimpleRemoteSynthesizer::new(assembly_encodings);

    let jobs = vec![];
    let jobs = Arc::new(Mutex::new(jobs));
    let job_reporter = SimpleJobReporter::new(jobs);
    let artifact_manager = SimpleArtifactManager;

    assert!(
        selected_circuits.len() <= TestingParams::number_of_setup_slots(&TestingParams) as usize
    );

    run_prover_with_remote_synthesizer(
        external_synthesizer,
        artifact_manager,
        job_reporter,
        Some(circuit_ids),
        TestingParams,
    );
}
#[test]
fn test_generic_prover_with_external_synthesizer_and_shuffled_circuits() {
    assert!(std::env::var("CRS_FILE").is_ok());
    let artifacts_dir = get_artifacts_dir();

    let circuit_ids = (0u8..20).collect::<Vec<u8>>();
    let circuits = read_circuits_from_directory(&artifacts_dir);
    assert!(!circuits.is_empty());

    let mut selected_circuits = vec![];
    for circuit in circuits {
        if !circuit_ids.contains(&circuit.numeric_circuit_type()) {
            continue;
        }

        for _ in 0..3 {
            selected_circuits.push(circuit.clone());
        }
    }
    rand::thread_rng().shuffle(&mut selected_circuits);
    let mut assembly_encodings = vec![];

    for circuit in selected_circuits.iter() {
        let circuit_id = circuit.numeric_circuit_type();
        let assembly_file_path = format!(
            "{}/{}_{}.bin",
            artifacts_dir.to_str().unwrap(),
            "assembly",
            circuit_id
        );
        let assembly_file_path = std::path::Path::new(&assembly_file_path);
        let assembly_file = std::fs::File::open(assembly_file_path).unwrap();
        let assembly_encoding = Box::new(assembly_file) as Box<dyn Read + Send + Sync>;
        assembly_encodings.push(assembly_encoding);
    }

    pub struct TestingParams;

    impl Params for TestingParams {
        fn number_of_parallel_synthesis(&self) -> u8 {
            20
        }
        fn number_of_setup_slots(&self) -> u8 {
            4
        }
    }

    let params = TestingParams;

    let external_synthesizer = SimpleRemoteSynthesizer::new(assembly_encodings);

    let jobs = vec![];
    let jobs = Arc::new(Mutex::new(jobs));
    let job_reporter = SimpleJobReporter::new(jobs);
    let artifact_manager = SimpleArtifactManager;

    run_prover_with_remote_synthesizer(
        external_synthesizer,
        artifact_manager,
        job_reporter,
        None,
        params,
    );
}

#[test]
fn test_specialized_prover_server() {
    assert!(std::env::var("CRS_FILE").is_ok());
    let artifacts_dir = get_artifacts_dir();

    let num_circuit_per_server = 10;

    let server_id = if let Ok(server_id) = std::env::var("SERVER_ID") {
        server_id.parse().unwrap()
    } else {
        0
    };

    pub struct TestingParamsForServer(u8);

    impl Params for TestingParamsForServer {
        fn number_of_parallel_synthesis(&self) -> u8 {
            4
        }
        fn number_of_setup_slots(&self) -> u8 {
            self.0
        }
    }
    let params = TestingParamsForServer(num_circuit_per_server as u8);

    #[derive(Clone)]
    struct SimpleNetworkRemoteSynthesizer {
        sender: SyncSender<Box<dyn Read + Send + Sync>>,
        receiver: Arc<Receiver<Box<dyn Read + Send + Sync>>>,
    }
    unsafe impl Send for SimpleNetworkRemoteSynthesizer {}

    impl RemoteSynthesizer for SimpleNetworkRemoteSynthesizer {
        fn try_next(&mut self) -> Option<Box<dyn Read + Send + Sync>> {
            if let Ok(buf) = self.receiver.try_recv() {
                Some(buf)
            } else {
                None
            }
        }
    }

    impl SimpleNetworkRemoteSynthesizer {
        fn new(bound: usize) -> Self {
            let (sender, receiver) = std::sync::mpsc::sync_channel(bound);
            Self {
                sender: sender,
                receiver: Arc::new(receiver),
            }
        }

        fn listen(&self) {
            let bind_addr = if let Ok(bind_addr) = std::env::var("BIND_ADDR") {
                bind_addr
            } else {
                "0.0.0.0:8080".to_string()
            };

            let listener = TcpListener::bind(&bind_addr)
                .unwrap_or_else(|_| panic!("Failed binding address: {:?}", bind_addr));
            println!("server started on {}", bind_addr);
            for conn in listener.incoming() {
                println!("received assembly encoding");
                let conn = Box::new(conn.unwrap());
                self.sender.send(conn).unwrap();
            }
        }
    }

    let external_synthesizer =
        SimpleNetworkRemoteSynthesizer::new(params.number_of_parallel_synthesis() as usize);
    let external_synthesizer_listener = external_synthesizer.clone();
    std::thread::spawn(move || loop {
        external_synthesizer_listener.listen();
    });

    let jobs = vec![];
    let jobs = Arc::new(Mutex::new(jobs));
    let job_reporter = SimpleJobReporter::new(jobs);
    let artifact_manager = SimpleArtifactManager;

    let circuit_ids =
        (server_id * num_circuit_per_server..(server_id + 1) * num_circuit_per_server).collect();
    dbg!(server_id);
    dbg!(&circuit_ids);

    run_prover_with_remote_synthesizer(
        external_synthesizer,
        artifact_manager,
        job_reporter,
        Some(circuit_ids),
        params,
    );
}

#[test]
fn test_run_remote_synthesizer() {
    struct HttpArtifactSender {
        url: String,
        circuits_per_server: usize,
        server_id: usize,
        sender: Sender<Box<dyn Read>>,
        receiver: Receiver<Box<dyn Read>>,
    }

    unsafe impl Send for HttpArtifactSender {}
    unsafe impl Sync for HttpArtifactSender {}

    impl HttpArtifactSender {
        fn new(url: String, server_id: usize, circuits_per_server: usize) -> Self {
            let (sender, receiver) = channel();
            Self {
                url,
                server_id,
                circuits_per_server,
                sender,
                receiver,
            }
        }

        fn try_connect(&self) {
            // let addr = SocketAddr::from_str(&self.server_url_first).unwrap();
            // let timeout = std::time::Duration::from_secs(5);
            // TcpStream::connect_timeout(&addr, timeout).expect("Server 1 is not running");

            // let addr = SocketAddr::from_str(&self.server_url_second).unwrap();
            // let timeout = std::time::Duration::from_secs(5);
            // TcpStream::connect_timeout(&addr, timeout).expect("Server 2 is not running");
        }
    }

    impl EncodedArtifactSender for HttpArtifactSender {
        fn send(
            &mut self,
            mut artifact: Box<dyn Read>,
            circuit_id: u8,
        ) -> Result<(), std::io::Error> {
            let circuit_id_bound = (self.server_id + 1) * self.circuits_per_server;
            let circuit_id = circuit_id as usize;
            assert!(circuit_id < circuit_id_bound);
            let mut socket = TcpStream::connect(self.url.clone())?;
            let _ = std::io::copy(&mut artifact, &mut socket)?;
            println!(
                "{} has been sent to the server {}",
                circuit_id, self.server_id
            );
            Ok(())
        }
    }

    pub struct TestingParamsForSynthesizer;

    impl Params for TestingParamsForSynthesizer {
        fn number_of_parallel_synthesis(&self) -> u8 {
            46
        }
        fn number_of_setup_slots(&self) -> u8 {
            0
        }
    }

    let params = TestingParamsForSynthesizer;

    let artifacts_dir = get_artifacts_dir();
    let circuits = read_circuits_from_directory(&artifacts_dir);
    assert!(!circuits.is_empty());
    let mut selected_circuits = vec![];
    'outer: loop {
        for circuit in circuits.iter() {
            selected_circuits.push(circuit.clone());
            if selected_circuits.len() == params.number_of_parallel_synthesis() as usize {
                break 'outer;
            }
        }
    }

    rand::thread_rng().shuffle(&mut selected_circuits);

    let jobs: Vec<(usize, ZkSyncCircuit, JobState)> = selected_circuits
        .into_iter()
        .enumerate()
        .map(|(idx, c)| (idx, c, JobState::Created(idx)))
        .collect();
    let jobs = Arc::new(Mutex::new(jobs));
    let job_manager = SimpleJobManager::new(jobs.clone());
    let job_reporter = SimpleJobReporter::new(jobs);
    let artifact_manager = SimpleArtifactManager;

    let server_url_first = std::env::var("SERVER_URL1").unwrap();
    let server_url_second = std::env::var("SERVER_URL2").unwrap();

    let mut artifact_senders = vec![];
    for (idx, url) in [server_url_first, server_url_second]
        .into_iter()
        .enumerate()
    {
        let artifact_sender = HttpArtifactSender::new(url, idx, 10);
        artifact_sender.try_connect();
        artifact_senders.push(artifact_sender);
    }

    run_remote_synthesizer(job_manager, job_reporter, artifact_senders, params);
}

#[test]
fn generate_setup_and_vk_from_json_file() {
    assert!(std::env::var("CRS_FILE").is_ok());
    let artifacts_dir = get_artifacts_dir();

    let mut circuits = read_circuits_from_directory(&artifacts_dir);

    let circuit_ids: Vec<u8> = circuits.iter().map(|c| c.numeric_circuit_type()).collect();
    dbg!(circuit_ids);

    let mut provers = create_prover_instances();
    assert!(!provers.is_empty());

    if provers.len() == 1 {
        println!("Generate artifacts sequentially");
        let (_, prover) = provers.pop().unwrap();
        run_serial_generate_setup_and_vk_for_circuits(circuits, prover);
    } else {
        println!("Generate artifacts in parallel");
        let provers = provers.into_iter().map(|p| p.1).collect::<Vec<_>>();
        run_parallel_generate_setu_and_vk_for_circuits(circuits, provers);
    }
}

fn run_serial_generate_setup_and_vk_for_circuits(circuits: Vec<ZkSyncCircuit>, mut prover: Prover) {
    let artifacts_dir = get_artifacts_dir();
    assert!(!circuits.is_empty());
    for circuit in circuits {
        dbg!(circuit.short_description());
        let circuit_id = circuit.numeric_circuit_type();
        let setup_file_path = format!(
            "{}/{}_{}.bin",
            artifacts_dir.to_str().unwrap(),
            SETUP_FILE_NAME,
            circuit_id
        );
        let setup_file_path = std::path::Path::new(&setup_file_path);

        if setup_file_path.exists() {
            continue;
        }
        let (setup, vk) = generate_setup_and_vk_for_circuit(&mut prover, &circuit);
        save_setup_into_file(&setup, setup_file_path);

        let vk_file_path = format!("{}/vk_{}.json", artifacts_dir.to_str().unwrap(), circuit_id);
        let vk_file_path = std::path::Path::new(&vk_file_path);
        if vk_file_path.exists() {
            continue;
        }
        save_vk_into_file(&vk, vk_file_path);
    }
}

fn run_parallel_generate_setu_and_vk_for_circuits(
    mut circuits: Vec<ZkSyncCircuit>,
    mut provers: Vec<Prover>,
) {
    let mut handles = vec![];

    let start_all_jobs = std::time::Instant::now();
    let (assemebly_sender, assembly_receiver) =
        std::sync::mpsc::channel::<(SetupAssembly, ZkSyncCircuit)>();

    let num_parallel_jobs_outer = Arc::new(std::sync::atomic::AtomicUsize::new(16));

    let num_parallel_jobs = num_parallel_jobs_outer.clone();
    let handle = std::thread::spawn(move || {
        let (prover_sender, prover_receiver) = std::sync::mpsc::channel();

        for prover in provers {
            prover_sender.send(prover).unwrap();
        }

        loop {
            let num_parallel_jobs = num_parallel_jobs.clone();
            if let mut prover = prover_receiver.recv().unwrap() {
                if let ((setup_assembly, circuit)) = assembly_receiver.recv().unwrap() {
                    let circuit_id = circuit.numeric_circuit_type();
                    let prover_sender = prover_sender.clone();
                    std::thread::spawn(move || {
                        println!("generating setup for {}", circuit.short_description());
                        let artifacts_dir = get_artifacts_dir();
                        let setup_file_path = format!(
                            "{}/{}_{}.bin",
                            artifacts_dir.to_str().unwrap(),
                            SETUP_FILE_NAME,
                            circuit_id
                        );
                        let setup_file_path = std::path::Path::new(&setup_file_path);
                        let setup = prover
                            .create_setup_from_assembly::<ZkSyncCircuit, _>(&setup_assembly)
                            .unwrap();
                        save_setup_into_file(&setup, setup_file_path);
                        println!("setup saved for {}", circuit.short_description());

                        let vk_file_path =
                            format!("{}/vk_{}.json", artifacts_dir.to_str().unwrap(), circuit_id);
                        let vk_file_path = std::path::Path::new(&vk_file_path);
                        let vk = prover
                            .inner_create_vk_from_assembly::<ZkSyncCircuit, _>(&setup_assembly)
                            .unwrap();
                        let vk = ZkSyncVerificationKey::from_verification_key_and_numeric_type(
                            circuit_id, vk,
                        );
                        save_vk_into_file(&vk, vk_file_path);
                        println!("vk saved for {}", circuit.short_description());
                        prover_sender.send(prover).unwrap();
                        num_parallel_jobs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    });
                }
            }
        }
    });
    handles.push(handle);

    let num_parallel_jobs = num_parallel_jobs_outer.clone();

    loop {
        if num_parallel_jobs.load(std::sync::atomic::Ordering::SeqCst) > 0 {
            let _ = num_parallel_jobs.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            let num_parallel_jobs = num_parallel_jobs.clone();
            let circuit = circuits.pop().unwrap();

            let assemebly_sender = assemebly_sender.clone();
            let handle = std::thread::spawn(move || {
                println!("synthesizing {}", circuit.short_description());
                let log_size = (Prover::get_max_domain_size()).trailing_zeros();
                let mut setup_assembly = Prover::new_setup_assembly();
                circuit.synthesize(&mut setup_assembly).unwrap();
                setup_assembly.finalize_to_size_log_2(log_size as usize);

                assemebly_sender.send((setup_assembly, circuit)).unwrap();
            });
            handles.push(handle);
        }
        if circuits.is_empty() {
            break;
        }
    }
    for handle in handles {
        handle.join().unwrap();
    }
    println!("all jobs takes {:?}", start_all_jobs.elapsed());
}
#[test]
fn transform_binary_to_json() {
    let artifacts_dir = get_artifacts_dir();

    for entry in std::fs::read_dir(artifacts_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let file_name = String::from(path.file_name().unwrap().to_str().unwrap());
        let is_circuit = file_name.contains("circuit");
        let is_bin = file_name.contains(".bin") && is_circuit;

        if is_bin {
            dbg!(file_name);
            let mut new_path = path.to_owned();
            new_path.set_extension("json");
            let file = std::fs::read(path).unwrap();
            let decoded: ZkSyncCircuit = bincode::deserialize(&file).unwrap();
            let new_file = std::fs::File::create(new_path).unwrap();
            serde_json::to_writer(&new_file, &decoded).unwrap();
        }
    }
}

#[test]
fn test_single_circuit_with_proving_assembly() {
    let circuit_file_path = std::env::var("CIRCUIT_FILE").expect("circuit file");
    let circuit = decode_circuit_from_file(&circuit_file_path.into());

    println!("{}", circuit.short_description());
    let mut prover = Prover::new();

    let (setup, vk) = generate_setup_and_vk_for_circuit(&mut prover, &circuit);
    let proof = prove_for_circuit(&mut prover, &circuit, &setup);
    let wrapped_proof =
        ZkSyncProof::from_proof_and_numeric_type(circuit.numeric_circuit_type(), proof);
    if !vk.verify_proof(&wrapped_proof) {
        println!("invalid proof");
    }
}

use zkevm_test_harness::bellman::plonk::better_better_cs::proof::Proof;

#[test]
fn test_proof_correctness_with_expected_proof() {
    let circuit_file_path = std::env::var("CIRCUIT_FILE").expect("circuit file");
    let circuit = decode_circuit_from_file(&circuit_file_path.into());

    let expected_proof_file_path = std::env::var("PROOF_FILE").expect("proof file");
    let expected_proof_file =
        std::fs::File::open(expected_proof_file_path).expect("open proof file");
    let expected_proof: Proof<Bn256, ZkSyncCircuit> =
        serde_json::from_reader(expected_proof_file).unwrap();

    println!("Circuit: {}", circuit.short_description());

    let mut prover = Prover::new();

    let (setup, vk) = generate_setup_and_vk_for_circuit(&mut prover, &circuit);
    let actual_proof = prove_for_circuit(&mut prover, &circuit, &setup);
    assert_proof(&expected_proof, &actual_proof);

    let wrapped_proof =
        ZkSyncProof::from_proof_and_numeric_type(circuit.numeric_circuit_type(), actual_proof);

    if !vk.verify_proof(&wrapped_proof) {
        panic!("invalid proof");
    }
}

// couldn't check equality with binary == operation and had this function
fn assert_proof(expected: &Proof<Bn256, ZkSyncCircuit>, actual: &Proof<Bn256, ZkSyncCircuit>) {
    assert_eq!(expected.n, actual.n, "proof mismatch: n");
    assert_eq!(expected.inputs, actual.inputs, "proof mismatch: inputs");
    assert_eq!(
        expected.state_polys_commitments, actual.state_polys_commitments,
        "proof mismatch: state_polys_commitments"
    );
    assert_eq!(
        expected.witness_polys_commitments, actual.witness_polys_commitments,
        "proof mismatch: witness_polys_commitments"
    );
    assert_eq!(
        expected.lookup_s_poly_commitment, actual.lookup_s_poly_commitment,
        "proof mismatch: lookup_s_poly_commitment"
    );
    assert_eq!(
        expected.copy_permutation_grand_product_commitment,
        actual.copy_permutation_grand_product_commitment,
        "proof mismatch: copy_permutation_grand_product_commitment"
    );
    assert_eq!(
        expected.lookup_grand_product_commitment, actual.lookup_grand_product_commitment,
        "proof mismatch: lookup_grand_product_commitment"
    );
    assert_eq!(
        expected.quotient_poly_parts_commitments, actual.quotient_poly_parts_commitments,
        "proof mismatch: quotient_poly_parts_commitments"
    );

    assert_eq!(
        expected.state_polys_openings_at_z, actual.state_polys_openings_at_z,
        "proof mismatch: state_polys_openings_at_z"
    );
    assert_eq!(
        expected.state_polys_openings_at_dilations, actual.state_polys_openings_at_dilations,
        "proof mismatch: state_polys_openings_at_dilations"
    );
    assert_eq!(
        expected.witness_polys_openings_at_z, actual.witness_polys_openings_at_z,
        "proof mismatch: witness_polys_openings_at_z"
    );
    assert_eq!(
        expected.witness_polys_openings_at_dilations, actual.witness_polys_openings_at_dilations,
        "proof mismatch: witness_polys_openings_at_dilations"
    );
    assert_eq!(
        expected.gate_setup_openings_at_z, actual.gate_setup_openings_at_z,
        "proof mismatch: gate_setup_openings_at_z"
    );
    assert_eq!(
        expected.gate_selectors_openings_at_z, actual.gate_selectors_openings_at_z,
        "proof mismatch: gate_selectors_openings_at_z"
    );
    assert_eq!(
        expected.copy_permutation_polys_openings_at_z, actual.copy_permutation_polys_openings_at_z,
        "proof mismatch: copy_permutation_polys_openings_at_z"
    );
    assert_eq!(
        expected.copy_permutation_grand_product_opening_at_z_omega,
        actual.copy_permutation_grand_product_opening_at_z_omega,
        "proof mismatch: copy_permutation_grand_product_opening_at_z_omega"
    );
    assert_eq!(
        expected.lookup_s_poly_opening_at_z_omega, actual.lookup_s_poly_opening_at_z_omega,
        "proof mismatch: lookup_s_poly_opening_at_z_omega"
    );
    assert_eq!(
        expected.lookup_grand_product_opening_at_z_omega,
        actual.lookup_grand_product_opening_at_z_omega,
        "proof mismatch: lookup_grand_product_opening_at_z_omega"
    );
    assert_eq!(
        expected.lookup_t_poly_opening_at_z, actual.lookup_t_poly_opening_at_z,
        "proof mismatch: lookup_t_poly_opening_at_z"
    );
    assert_eq!(
        expected.lookup_t_poly_opening_at_z_omega, actual.lookup_t_poly_opening_at_z_omega,
        "proof mismatch: lookup_t_poly_opening_at_z_omega"
    );
    assert_eq!(
        expected.lookup_selector_poly_opening_at_z, actual.lookup_selector_poly_opening_at_z,
        "proof mismatch: lookup_selector_poly_opening_at_z"
    );
    assert_eq!(
        expected.lookup_table_type_poly_opening_at_z, actual.lookup_table_type_poly_opening_at_z,
        "proof mismatch: lookup_table_type_poly_opening_at_z"
    );
    assert_eq!(
        expected.quotient_poly_opening_at_z, actual.quotient_poly_opening_at_z,
        "proof mismatch: quotient_poly_opening_at_z"
    );
    assert_eq!(
        expected.linearization_poly_opening_at_z, actual.linearization_poly_opening_at_z,
        "proof mismatch: linearization_poly_opening_at_z"
    );

    assert_eq!(
        expected.opening_proof_at_z, actual.opening_proof_at_z,
        "proof mismatch: opening_proof_at_z"
    );
    assert_eq!(
        expected.opening_proof_at_z_omega, actual.opening_proof_at_z_omega,
        "proof mismatch: opening_proof_at_z_omega"
    );
}

#[test]
fn test_custom_serialization_of_multiple_assemblies() {
    let artifacts_dir = get_artifacts_dir();

    let mut circuits = read_circuits_from_directory(&artifacts_dir);
    assert!(!circuits.is_empty());
    let mut selected_circuits = vec![];
    let mut circuit_ids = (1..20).collect::<Vec<u8>>();
    for circuit in circuits {
        if !circuit_ids.contains(&circuit.numeric_circuit_type()) {
            continue;
        }

        selected_circuits.push(circuit);
    }

    let mut handles = vec![];

    let num_physical = 8;
    let num_cores = Arc::new(AtomicUsize::new(num_physical));

    let capacity = calculate_serialization_capacity_for_proving_assembly();
    loop {
        let num_cores = num_cores.clone();
        if num_cores.load(Ordering::SeqCst) > 0 {
            if let Some(circuit) = selected_circuits.pop() {
                num_cores.fetch_sub(1, Ordering::SeqCst);
                let expeted_circuit_id = circuit.numeric_circuit_type();
                let assembly_file_path = format!(
                    "{}/{}_{}.bin",
                    artifacts_dir.to_str().unwrap(),
                    "assembly",
                    expeted_circuit_id
                );
                let handle = std::thread::spawn(move || {
                    let circuit_id = circuit.numeric_circuit_type();
                    let job_id = 42 + circuit_id as usize;
                    println!("synthesizing {}", circuit.short_description());

                    let mut assembly = prover::Prover::new_proving_assembly();
                    circuit.synthesize(&mut assembly).unwrap();

                    for table in assembly.tables.iter() {
                        let internal = table.as_internal();
                        let expected_table_id = table.table_id();
                        let actual_table_id = table_id_from_string(&internal.name());
                        assert_eq!(expected_table_id, actual_table_id);

                        let sorted_table = assembly
                            .individual_table_canonical_sorted_entries
                            .get(&table.functional_name())
                            .unwrap();
                        assert_eq!(table.size(), sorted_table.len());
                    }

                    let mut assembly_encoding = Vec::with_capacity(capacity);
                    serialize_job(&assembly, job_id, circuit_id, &mut assembly_encoding);

                    let mut file = std::fs::File::create(&assembly_file_path).unwrap();
                    file.write_all(&assembly_encoding).unwrap();

                    let mut decoded_assembly = Prover::new_proving_assembly();
                    let mut buffer = Cursor::new(assembly_encoding);
                    deserialize_job(&mut buffer, &mut decoded_assembly);

                    compare_assemblies(&assembly, &decoded_assembly);

                    num_cores.fetch_add(1, Ordering::SeqCst);
                });
                handles.push(handle);
            } else {
                break;
            }
        }
    }
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_custom_serialization_of_assembly() {
    let circuit_file = std::env::var("CIRCUIT_FILE").unwrap();
    let circuit = decode_circuit_from_file(&circuit_file.into());
    let circuit_id = circuit.numeric_circuit_type();

    println!(
        "{} {} synthesizing",
        circuit.numeric_circuit_type(),
        circuit.short_description()
    );

    let mut expected_assembly = prover::Prover::new_proving_assembly();
    circuit.synthesize(&mut expected_assembly).unwrap();

    let mut buffer = Vec::with_capacity(calculate_serialization_capacity_for_proving_assembly());
    let start = std::time::Instant::now();
    custom_assembly_serialization(&expected_assembly, &mut buffer);
    println!(
        "{} {} custom serialization takes {:?}",
        circuit.numeric_circuit_type(),
        circuit.short_description(),
        start.elapsed()
    );

    let mut actual_assembly = Prover::new_proving_assembly();
    let start = std::time::Instant::now();
    let mut encoding = Cursor::new(buffer);
    custom_assembly_deserialization(&mut encoding, &mut actual_assembly);
    println!(
        "{} {} custom deserialization takes {:?}",
        circuit.numeric_circuit_type(),
        circuit.short_description(),
        start.elapsed()
    );

    compare_assemblies(&expected_assembly, &actual_assembly);
}

fn compare_assemblies(this: &ProvingAssembly, other: &ProvingAssembly) {
    if this.aux_assingments.len() != other.aux_assingments.len() {
        panic!(
            "aux_assingments lengths are not equal {} {}",
            this.aux_assingments.len(),
            other.aux_assingments.len()
        );
    }
    if this.aux_assingments != other.aux_assingments {
        panic!("aux_assingments are not equal");
    }
    for idx in 0..4 {
        let poly_idx = PolyIdentifier::VariablesPolynomial(idx);
        if this.aux_storage.state_map.get(&poly_idx) != other.aux_storage.state_map.get(&poly_idx) {
            panic!("aux_storage are not equal {}", idx);
        }
        if this.inputs_storage.state_map.get(&poly_idx)
            != other.inputs_storage.state_map.get(&poly_idx)
        {
            panic!("inputs_storage are not equal {}", idx);
        }
    }

    let table_names = this
        .tables
        .iter()
        .map(|t| t.functional_name())
        .collect::<Vec<_>>();

    for table_name in table_names {
        let a = this
            .individual_table_canonical_sorted_entries
            .get(&table_name)
            .unwrap();
        let b = other
            .individual_table_canonical_sorted_entries
            .get(&table_name)
            .unwrap();
        if a != b {
            panic!(
                "individual_table_canonical_sorted_entries are not equal {}",
                table_name
            )
        }

        let a = this.individual_table_entries.get(&table_name).unwrap();
        let b = other.individual_table_entries.get(&table_name).unwrap();

        if a != b {
            panic!("individual_table_entries are not equal {}", table_name)
        }

        let a = this.known_table_ids.get(&table_name).unwrap();
        let b = other.known_table_ids.get(&table_name).unwrap();
        if a != b {
            panic!("known_table_ids are not equal");
        }
    }
    for (a, b) in this
        .known_table_names
        .iter()
        .zip(other.known_table_names.iter())
    {
        if a != b {
            panic!("known_table_names are not equal");
        }
    }

    assert_eq!(this.num_input_gates, other.num_input_gates);
    assert_eq!(this.num_aux_gates, other.num_aux_gates);
    assert_eq!(this.num_aux, other.num_aux);
    assert_eq!(this.num_inputs, other.num_inputs);
}

#[test]
fn test_create_proof_for_deserialized_assembly() {
    assert!(std::env::var("CRS_FILE").is_ok());

    let assembly_file_path = std::env::var("ASSEMBLY_FILE").unwrap();
    let mut assembly_file = std::fs::File::open(&assembly_file_path).unwrap();
    let mut assembly = Prover::new_proving_assembly();
    let (job_id, circuit_id) = deserialize_job(&mut assembly_file, &mut assembly);

    for (table_name, table) in assembly.individual_table_canonical_sorted_entries.iter() {
        let num_rows_of_witnesses = assembly
            .individual_table_entries
            .get(table_name)
            .expect(&format!("assembly doesn't contain table: {}", table_name))
            .len();
    }
    let mut prover = Prover::new();
    let artifact_manager = SimpleArtifactManager;
    let worker = Prover::new_worker(None);
    let vk = artifact_manager.get_vk(circuit_id).unwrap();
    let setup_encoding = artifact_manager.get_setup(circuit_id).unwrap();
    let setup = decode_setup(circuit_id, setup_encoding);

    println!("create proof for {}", circuit_id);
    let log_size = Prover::get_max_domain_size().trailing_zeros();
    assembly.finalize_to_size_log_2(log_size as usize);
    assert!(assembly.is_satisfied());

    let transcript_params = (bn254_rescue_params(), get_prefered_rns_params());
    let transcript_params = Some((&transcript_params.0, &transcript_params.1));

    let start = std::time::Instant::now();

    let proof = if circuit_id == 0 {
        prover
        .create_proof_with_proving_assembly_and_transcript::<ZkSyncCircuit, RollingKeccakTranscript<Fr>>(
            &assembly,
            &setup.as_setup(),
            None,
        )
        .unwrap()
    } else {
        prover
        .create_proof_with_proving_assembly_and_transcript::<ZkSyncCircuit, RescueTranscriptForRecursion<'_>>(
            &assembly,
            &setup.as_setup(),
            transcript_params,
        )
        .unwrap()
    };

    let proof = ZkSyncProof::from_proof_and_numeric_type(circuit_id, proof);
    assert!(vk.verify_proof(&proof));
}
