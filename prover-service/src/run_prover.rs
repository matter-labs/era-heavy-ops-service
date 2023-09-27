use prover::ProvingAssembly;
use std::sync::mpsc::channel;
use std::sync::RwLock;
use std::time::Duration;
use std::{
    collections::HashMap,
    sync::mpsc::{Receiver, Sender},
};
use zkevm_test_harness::bellman::bn256::Fr;

use zkevm_test_harness::{
    bellman::plonk::commitments::transcript::keccak_transcript::RollingKeccakTranscript,
    sync_vm::utils::bn254_rescue_params,
};

use crate::remote_synth::{
    calculate_serialization_capacity_for_proving_assembly, deserialize_job, serialize_job,
};
use crate::setup::ZkSyncSetup;

pub struct GenericReceiver<T>(Receiver<T>);
unsafe impl<T> Send for GenericReceiver<T> {}
unsafe impl<T> Sync for GenericReceiver<T> {}

use super::*;

pub struct ThreadGuard {
    job_id: usize,
    thread: u8,
    report_sender: Sender<JobResult>,
    thread_status_sender: Sender<u8>,
}

// we don't need to handle panic per job id
impl ThreadGuard {
    pub fn new(
        thread: u8,
        job_id: usize,
        report_sender: Sender<JobResult>,
        thread_status_sender: Sender<u8>,
    ) -> Self {
        Self {
            thread,
            job_id,
            report_sender,
            thread_status_sender,
        }
    }
}

pub(crate) const PROVER_THREAD_HANDLE: u8 = 0;
pub(crate) const SETUP_THREAD_HANDLE: u8 = 1;
pub(crate) const SYNTH_THREAD_HANDLE: u8 = 2;
pub(crate) const ENCODER_THREAD_HANDLE: u8 = 3;
pub(crate) const PROOF_GENERATION_THREAD_HANDLE: u8 = 4;

impl Drop for ThreadGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let msg = match self.thread {
                PROVER_THREAD_HANDLE => "proving thread panicked",
                SETUP_THREAD_HANDLE => "setup thread panicked",
                SYNTH_THREAD_HANDLE => "synthesize thread panicked",
                ENCODER_THREAD_HANDLE => "assembly decoding job panicked",
                PROOF_GENERATION_THREAD_HANDLE => "proof generation child thread panicked",
                _ => "unknown panick",
            };

            if self.thread == PROVER_THREAD_HANDLE || self.thread == SETUP_THREAD_HANDLE {
                if let Err(e) = self.thread_status_sender.send(self.thread) {
                    // most likely main thread is panicked and cant handle messages
                    // if main thread isn't available then there is no reason to
                    // run infellible threads at all
                    // so we can process::exit(1) here
                    println!("{}", format!("[{}] {}: {}", self.job_id, msg, e));
                };
            }
            self.report_sender
                .send(JobResult::Failure(self.job_id, msg.to_string()))
                .unwrap();
        }
    }
}

pub(crate) struct ProverContext {
    setup_input_receiver: Option<Receiver<ProverMessage>>,
    prover_input_sender: Sender<ProverMessage>,
    prover_instance_receiver: Receiver<(usize, Prover, std::time::Instant)>,
    prover_instance_sender: Sender<(usize, Prover, std::time::Instant)>,
    prover_input_receiver: Receiver<ProverMessage>,
    main_prover_input_sender: Sender<ProverMessage>,
    reusable_assembly_sender: Sender<ProvingAssembly>,
    reusable_assembly_receiver: Receiver<ProvingAssembly>,
    pub(crate) thread_status_sender: Sender<u8>,
    thread_status_receiver: Receiver<u8>,
    pub(crate) report_sender: Sender<JobResult>,
    report_receiver: Receiver<JobResult>,
    num_provers: usize,
    specialized_circuit_ids: Option<Vec<u8>>,
}

unsafe impl Send for ProverContext {}
unsafe impl Sync for ProverContext {}

impl ProverContext {
    fn init(circuit_ids: Option<Vec<u8>>, num_parallel_synthesis: u8) -> Self {
        let (prover_input_sender, prover_input_receiver) = channel();
        let (prover_instance_sender, prover_instance_receiver) = channel();
        let (reusable_assembly_sender, reusable_assembly_receiver) = channel();
        let (report_sender, report_receiver) = channel();
        let (thread_status_sender, thread_status_receiver) = channel();

        let mut setup_input_receiver = None;
        let main_prover_input_sender = if circuit_ids.is_some() {
            // this is a specialized prover so that send prover inputs directly to the prover thread
            prover_input_sender.clone()
        } else {
            // this is the generic prover so that send prover inputs to the setup thread first then prover thread
            let (sender, receiver) = channel();
            setup_input_receiver = Some(receiver);
            sender
        };

        for _ in 0..num_parallel_synthesis as usize {
            reusable_assembly_sender
                .send(Prover::new_proving_assembly())
                .unwrap()
        }

        let provers = create_prover_instances();
        let num_provers = provers.len();
        for (prover_idx, prover) in provers {
            let prover_instance = (prover_idx, prover, std::time::Instant::now());
            prover_instance_sender.send(prover_instance).unwrap();
        }

        Self {
            setup_input_receiver,
            prover_input_sender,
            prover_input_receiver,
            prover_instance_sender,
            prover_instance_receiver,
            reusable_assembly_sender,
            reusable_assembly_receiver,
            thread_status_sender,
            thread_status_receiver,
            report_sender,
            report_receiver,
            main_prover_input_sender,
            num_provers,
            specialized_circuit_ids: circuit_ids,
        }
    }
}

fn check_job_is_allowed(ctx: &ProverContext, job_id: usize, circuit_id: u8) -> bool {
    if let Some(ref circuit_ids) = ctx.specialized_circuit_ids {
        if !circuit_ids.contains(&circuit_id) {
            ctx.report_sender
                .send(JobResult::Failure(
                    job_id,
                    format!("unknown circuit type {}", circuit_id),
                ))
                .unwrap();
            return false;
        }
    }
    true
}

pub fn run_prover_with_local_synthesizer<
    AM: ArtifactProvider + 'static,
    JM: JobManager,
    JR: JobReporter + 'static,
    P: Params + 'static,
>(
    artifact_manager: AM,
    mut job_manager: JM,
    job_reporter: JR,
    circuit_ids: Option<Vec<u8>>,
    params: P,
) {
    let ctx = ProverContext::init(circuit_ids.clone(), params.number_of_parallel_synthesis());
    let ctx = Arc::new(ctx);

    let params = Arc::new(params);

    let artifact_manager = Arc::new(artifact_manager);

    thread_liveness_tracker(
        ctx.clone(),
        artifact_manager.clone(),
        job_reporter,
        circuit_ids.clone(),
        params.clone(),
    );

    main_prover_handler(
        ctx.clone(),
        artifact_manager.clone(),
        circuit_ids,
        params.clone(),
    );
    let mut scheduler_is_idle = std::time::Instant::now();
    loop {
        let reusable_assembly = ctx.reusable_assembly_receiver.recv().unwrap();
        let (job_id, circuit) = job_manager.get_next_job();
        if check_job_is_allowed(ctx.as_ref(), job_id, circuit.numeric_circuit_type()) == false {
            continue;
        }
        let scheduler_received_input = scheduler_is_idle.elapsed();
        ctx.report_sender
            .send(JobResult::SchedulerWaitedIdle(scheduler_received_input))
            .unwrap();
        spawn_new_synthesize(ctx.clone(), reusable_assembly, job_id, circuit);
        scheduler_is_idle = std::time::Instant::now();
    }
}

pub fn run_prover_with_remote_synthesizer<
    RS: RemoteSynthesizer + 'static,
    AM: ArtifactProvider + 'static,
    JR: JobReporter + 'static,
    P: Params + 'static,
>(
    mut remote_synthesizer: RS,
    artifact_manager: AM,
    job_reporter: JR,
    circuit_ids: Option<Vec<u8>>,
    params: P,
) {
    let ctx = ProverContext::init(circuit_ids.clone(), params.number_of_parallel_synthesis());
    let ctx = Arc::new(ctx);

    let params = Arc::new(params);

    let artifact_manager = Arc::new(artifact_manager);
    thread_liveness_tracker(
        ctx.clone(),
        artifact_manager.clone(),
        job_reporter,
        circuit_ids.clone(),
        params.clone(),
    );

    main_prover_handler(
        ctx.clone(),
        artifact_manager.clone(),
        circuit_ids,
        params.clone(),
    );
    let mut scheduler_is_idle = std::time::Instant::now();
    loop {
        if let Some(mut encoded_assembly) = remote_synthesizer.try_next() {
            let reusable_assembly = ctx.reusable_assembly_receiver.recv().unwrap();
            let scheduler_received_input = scheduler_is_idle.elapsed();
            ctx.report_sender
                .send(JobResult::SchedulerWaitedIdle(scheduler_received_input))
                .unwrap();
            scheduler_is_idle = std::time::Instant::now();

            let mut job_id_bytes = [0u8; 8];
            encoded_assembly.read_exact(&mut job_id_bytes[..]).unwrap();
            let job_id = usize::from_le_bytes(job_id_bytes);
            let mut circuit_id = [0u8; 1];
            encoded_assembly.read_exact(&mut circuit_id[..]).unwrap();
            let circuit_id = circuit_id[0];
            if check_job_is_allowed(ctx.as_ref(), job_id, circuit_id) == false {
                continue;
            }
            spawn_new_assembly_decoding(
                ctx.clone(),
                job_id,
                circuit_id,
                encoded_assembly,
                reusable_assembly,
            );
        } else {
            sleep_for_duration(params.polling_duration());
        }
    }
}

fn sleep_for_duration(duration: Duration) {
    std::thread::sleep(duration);
}

fn thread_liveness_tracker<
    AM: ArtifactProvider + 'static,
    JR: JobReporter + 'static,
    P: Params + 'static,
>(
    ctx: Arc<ProverContext>,
    artifact_manager: Arc<AM>,
    mut job_reporter: JR,
    circuit_ids: Option<Vec<u8>>,
    params: Arc<P>,
) {
    let duration = params.polling_duration();
    std::thread::spawn(move || loop {
        for report in ctx.report_receiver.try_iter() {
            job_reporter.send_report(report);
        }

        for thread_id in ctx.thread_status_receiver.try_iter() {
            match thread_id {
                SETUP_THREAD_HANDLE => {
                    println!("re-spawning setup loading thread");
                    setup_loader(ctx.clone(), artifact_manager.clone(), params.clone());
                }
                PROVER_THREAD_HANDLE => {
                    println!("re-spawning prover thread");
                    proof_handler(
                        ctx.clone(),
                        artifact_manager.clone(),
                        circuit_ids.clone(),
                        params.clone(),
                    );
                }
                _ => (),
            }
        }

        sleep_for_duration(duration);
    });
}

fn spawn_new_assembly_decoding(
    ctx: Arc<ProverContext>,
    job_id: usize,
    circuit_id: u8,
    mut encoded_assembly: Box<dyn Read + Send + Sync>,
    mut reusable_assembly: ProvingAssembly,
) {
    let log_degree = Prover::get_max_domain_size_log();
    std::thread::spawn(move || {
        let assembly_decoded = std::time::Instant::now();

        use super::remote_synth::custom_assembly_deserialization;
        custom_assembly_deserialization(&mut encoded_assembly, &mut reusable_assembly);
        drop(encoded_assembly);

        ctx.report_sender
            .send(JobResult::AssemblyDecoded(
                job_id,
                assembly_decoded.elapsed(),
            ))
            .unwrap();
        let assembly_finalized = std::time::Instant::now();
        reusable_assembly.finalize_to_size_log_2(log_degree);
        ctx.report_sender
            .send(JobResult::AssemblyFinalized(
                job_id,
                assembly_finalized.elapsed(),
            ))
            .unwrap();

        ctx.main_prover_input_sender
            .send(ProverMessage(reusable_assembly, job_id, circuit_id, None))
            .unwrap();
    });
}

fn spawn_new_synthesize(
    ctx: Arc<ProverContext>,
    mut assembly: ProvingAssembly,
    job_id: usize,
    circuit: ZkSyncCircuit,
) {
    std::thread::spawn(move || {
        let guard = ThreadGuard::new(
            SYNTH_THREAD_HANDLE,
            job_id,
            ctx.report_sender.clone(),
            ctx.thread_status_sender.clone(),
        );
        let circuit_id = circuit.numeric_circuit_type();
        println!("synthesizing circuit {}", circuit.short_description());

        let synth_started = std::time::Instant::now();
        circuit.synthesize(&mut assembly).unwrap();
        ctx.report_sender
            .send(JobResult::Synthesized(job_id, synth_started.elapsed()))
            .unwrap();

        let assembly_finalized = std::time::Instant::now();
        let log_size = (prover::Prover::get_max_domain_size()).trailing_zeros();
        assembly.finalize_to_size_log_2(log_size as usize);
        ctx.report_sender
            .send(JobResult::AssemblyFinalized(
                job_id,
                assembly_finalized.elapsed(),
            ))
            .unwrap();

        let setup_loaded = std::time::Instant::now();

        ctx.report_sender
            .send(JobResult::SetupLoaded(job_id, setup_loaded.elapsed(), true))
            .unwrap();

        let prover_input = ProverMessage(assembly, job_id, circuit_id, None);
        ctx.main_prover_input_sender.send(prover_input).unwrap()
    });
}

fn init_setup_cache(
    circuit_ids: Vec<u8>,
    number_of_setup_slots: usize,
) -> HashMap<u8, RwLock<Arc<ZkSyncSetup>>> {
    println!("allocating {} setup buffers", circuit_ids.len());
    assert!(circuit_ids.len() <= number_of_setup_slots);
    let mut setup_map = HashMap::new();
    for circuit_id in circuit_ids.iter() {
        let setup = RwLock::new(Arc::new(ZkSyncSetup::empty(Prover::new_setup())));
        setup_map.insert(*circuit_id, setup);
    }
    println!(
        "all({} items) setup buffers are allocated",
        circuit_ids.len()
    );

    setup_map
}

fn load_setup_or_read_from_cache<AM: ArtifactProvider>(
    ctx: Arc<ProverContext>,
    cache: &HashMap<u8, RwLock<Arc<ZkSyncSetup>>>,
    circuit_id: u8,
    artifact_manager: Arc<AM>,
    job_id: usize,
) -> Arc<ZkSyncSetup> {
    let guarded_setup = cache.get(&circuit_id).expect("setup in cache");
    let mut setup = guarded_setup.write().unwrap();
    let setup_started = std::time::Instant::now();
    if setup.is_free() {
        println!("setup isn't in cache, loading.");
        // setup initially contains dummy setup, so we need to load actual setup
        let inner_setup = unsafe { Arc::get_mut_unchecked(&mut setup) };
        if let Ok(setup_encoding) = artifact_manager.get_setup(circuit_id) {
            inner_setup.reload(setup_encoding, circuit_id);
            ctx.report_sender
                .send(JobResult::SetupLoaded(
                    job_id,
                    setup_started.elapsed(),
                    false,
                ))
                .unwrap();
        } else {
            ctx.report_sender
                .send(JobResult::Failure(
                    job_id,
                    format!("setup encoding for circuit {} not found", circuit_id),
                ))
                .unwrap();
        }
    } else {
        // cache hit
        ctx.report_sender
            .send(JobResult::SetupLoaded(
                job_id,
                setup_started.elapsed(),
                true,
            ))
            .unwrap();
    };

    setup.clone()
}

struct ProverMessage(ProvingAssembly, usize, u8, Option<Arc<ZkSyncSetup>>);
unsafe impl Send for ProverMessage {}
unsafe impl Sync for ProverMessage {}

fn main_prover_handler<AM: ArtifactProvider + 'static, P: Params>(
    ctx: Arc<ProverContext>,
    artifact_manager: Arc<AM>,
    circuit_ids: Option<Vec<u8>>,
    params: Arc<P>,
) {
    proof_handler(
        ctx.clone(),
        artifact_manager.clone(),
        circuit_ids.clone(),
        params.clone(),
    );

    if circuit_ids.is_none() {
        setup_loader(ctx.clone(), artifact_manager.clone(), params);
    }
}

fn setup_loader<AM: ArtifactProvider + 'static, P: Params>(
    ctx: Arc<ProverContext>,
    artifact_manager: Arc<AM>,
    params: Arc<P>,
) {
    let num_setup_slots = params.number_of_setup_slots();

    std::thread::spawn(move || loop {
        let mut cache: Vec<Arc<ZkSyncSetup>> = (0..num_setup_slots)
            .map(|_| Arc::new(ZkSyncSetup::empty(Prover::new_setup())))
            .collect::<Vec<Arc<ZkSyncSetup>>>();
        println!("setup handler started");

        let mut setup_loader_is_idle = std::time::Instant::now();

        'outer: loop {
            if let Ok(setup_input) = ctx
                .setup_input_receiver
                .as_ref()
                .expect("setup receiver")
                .recv()
            {
                let setup_input_received = setup_loader_is_idle.elapsed();
                ctx.report_sender
                    .send(JobResult::SetupLoaderWaitedIdle(setup_input_received))
                    .unwrap();

                let ProverMessage(assembly, job_id, circuit_id, _) = setup_input;
                let guard = ThreadGuard::new(
                    SETUP_THREAD_HANDLE,
                    job_id,
                    ctx.report_sender.clone(),
                    ctx.thread_status_sender.clone(),
                );

                let mut assembly = Some(assembly);

                let setup_started = std::time::Instant::now();
                loop {
                    // first try to hit cache
                    let mut cache_hit = false;
                    'inner: for entry in cache.iter() {
                        if entry.is_busy() {
                            if entry.numeric_circuit_type() == circuit_id {
                                ctx.prover_input_sender
                                    .send(ProverMessage(
                                        assembly.take().unwrap(),
                                        job_id,
                                        circuit_id,
                                        Some(entry.clone()),
                                    ))
                                    .unwrap();
                                cache_hit = true;
                                break 'inner;
                            }
                        }
                    }

                    // invalidate stale entries
                    // TODO: should we invalidate all stale entries or single one?
                    let mut invalidated = false;
                    for slot in cache.iter_mut() {
                        if invalidated {
                            continue;
                        }
                        if slot.is_busy() {
                            if Arc::strong_count(slot) == 1 {
                                let inner = unsafe { Arc::get_mut_unchecked(slot) };
                                inner.free();
                                invalidated = true;
                            }
                        }
                    }

                    if cache_hit {
                        ctx.report_sender
                            .send(JobResult::SetupLoaded(
                                job_id,
                                setup_started.elapsed(),
                                cache_hit,
                            ))
                            .unwrap();
                        setup_loader_is_idle = std::time::Instant::now();
                        continue 'outer;
                    }

                    // then look for free slots
                    for slot in cache.iter_mut() {
                        if slot.is_free() {
                            match artifact_manager.get_setup(circuit_id) {
                                Ok(setup_encoding) => {
                                    let inner = unsafe { Arc::get_mut_unchecked(slot) };
                                    inner.reload(setup_encoding, circuit_id);
                                    assert_eq!(inner.numeric_circuit_type(), circuit_id);
                                    ctx.prover_input_sender
                                        .send(ProverMessage(
                                            assembly.take().unwrap(),
                                            job_id,
                                            circuit_id,
                                            Some(slot.clone()),
                                        ))
                                        .unwrap();
                                    ctx.report_sender
                                        .send(JobResult::SetupLoaded(
                                            job_id,
                                            setup_started.elapsed(),
                                            false,
                                        ))
                                        .unwrap();
                                    setup_loader_is_idle = std::time::Instant::now();
                                    continue 'outer;
                                }
                                Err(e) => {
                                    ctx.report_sender
                                        .send(JobResult::Failure(job_id, e.to_string()))
                                        .unwrap();
                                }
                            }
                        }
                    }
                    setup_loader_is_idle = std::time::Instant::now();
                }
            }
        }
    });
}

fn proof_handler<AM: ArtifactProvider + 'static, P: Params>(
    ctx: Arc<ProverContext>,
    artifact_manager: Arc<AM>,
    circuit_ids: Option<Vec<u8>>,
    params: Arc<P>,
) {
    let polling_duration = params.polling_duration();
    let number_of_setup_slots = params.number_of_setup_slots() as usize;

    std::thread::spawn(move || {
        let setup_cache = if let Some(circuit_ids) = circuit_ids {
            let cache = init_setup_cache(circuit_ids, number_of_setup_slots);
            let cache = Arc::new(cache);
            Some(cache)
        } else {
            None
        };

        loop {
            let artifact_manager = artifact_manager.clone();
            let job_reporter = ctx.report_sender.clone();
            let reusable_assembly_sender = ctx.reusable_assembly_sender.clone();
            let prover_instance_sender = ctx.prover_instance_sender.clone();
            let setup_cache = setup_cache.clone();

            if let Ok((prover_idx, prover, prover_instance_become_idle)) =
                ctx.prover_instance_receiver.try_recv()
            {
                let input = ctx.prover_input_receiver.recv().unwrap();
                let prover = (prover_idx, prover);
                let prover_input_received = prover_instance_become_idle.elapsed();
                ctx.report_sender
                    .send(JobResult::ProverWaitedIdle(
                        prover_idx,
                        prover_input_received,
                    ))
                    .unwrap();
                let ctx = ctx.clone();
                std::thread::spawn(move || {
                    create_proof(ctx, artifact_manager, input, prover, setup_cache)
                });
            } else {
                sleep_for_duration(polling_duration);
            }
        }
    });
}

fn create_proof<AM: ArtifactProvider>(
    ctx: Arc<ProverContext>,
    artifact_manager: Arc<AM>,
    input: ProverMessage,
    mut prover: (usize, Prover),
    setup_cache: Option<Arc<HashMap<u8, RwLock<Arc<ZkSyncSetup>>>>>,
) {
    let job_id = input.1;
    let guard = ThreadGuard::new(
        PROOF_GENERATION_THREAD_HANDLE,
        job_id,
        ctx.report_sender.clone(),
        ctx.thread_status_sender.clone(),
    );

    let ProverMessage(assembly, job_id, circuit_id, mut setup) = input;

    let setup = if let Some(setup_cache) = setup_cache {
        load_setup_or_read_from_cache(
            ctx.clone(),
            setup_cache.as_ref(),
            circuit_id,
            artifact_manager.clone(),
            job_id,
        )
    } else {
        setup.take().unwrap()
    };

    let (prover_idx, mut prover) = prover;

    let report = if let Ok(vk) = artifact_manager.get_vk(circuit_id) {
        let proof_generated = std::time::Instant::now();
        println!("Creating proof for job-id: {}", job_id);
        let result = if circuit_id == 0 {
            prover.create_proof_with_proving_assembly_and_transcript::<_, RollingKeccakTranscript<Fr>>(&assembly, setup.as_setup(), None)
        } else {
            let rescue_params = bn254_rescue_params();
            let rns_params = get_prefered_rns_params();
            let transcript_params = Some((&rescue_params, &rns_params));
            prover.create_proof_with_proving_assembly_and_transcript::<_, RescueTranscriptForRecursion>(&assembly, setup.as_setup(), transcript_params)
        };
        let proof_generated = proof_generated.elapsed();
        match result {
            Ok(proof) => {
                let proof = ZkSyncProof::from_proof_and_numeric_type(circuit_id, proof);
                if vk.verify_proof(&proof) {
                    JobResult::ProofGenerated(job_id, proof_generated, proof, prover_idx)
                } else {
                    JobResult::Failure(job_id, format!("{} proof verification failed", prover_idx))
                }
            }
            Err(msg) => {
                // JobResult::Failure(job_id, format!("proof generation failed: {}", msg))
                let mut assembly_encoding =
                    Vec::with_capacity(calculate_serialization_capacity_for_proving_assembly());
                serialize_job(&assembly, job_id, circuit_id, &mut assembly_encoding);
                JobResult::FailureWithDebugging(
                    job_id,
                    circuit_id,
                    assembly_encoding,
                    format!("{} proof generation failed: {}", prover_idx, msg),
                )
            }
        }
    } else {
        JobResult::Failure(job_id, format!("{} couldn't get a vk", prover_idx))
    };

    ctx.prover_instance_sender
        .send((prover_idx, prover, std::time::Instant::now()))
        .unwrap();

    let recycled_assembly = recycle_assembly(assembly);
    ctx.reusable_assembly_sender
        .send(recycled_assembly)
        .unwrap();

    ctx.report_sender.send(report).unwrap();
}

pub(crate) fn recycle_assembly(assembly: ProvingAssembly) -> ProvingAssembly {
    // reuse aux assignments and aux storage since they are already allocated on pinned memory
    let ProvingAssembly {
        mut aux_assingments,
        mut aux_storage,
        mut individual_table_entries,
        mut reusable_buffer_for_lookup_entries,
        ..
    } = assembly;

    let max_aux_assignments = Prover::get_max_num_variables();
    let domain_size = Prover::get_max_domain_size();

    unsafe {
        aux_assingments.set_len(0);
        assert_eq!(aux_assingments.capacity(), max_aux_assignments);
        for (_, variables) in aux_storage.state_map.iter_mut() {
            variables.set_len(0);
            assert_eq!(variables.capacity(), domain_size);
        }
    }

    // also reuse lookup related fields
    let num_lookup_tables = Prover::get_num_lookup_tables();
    let max_num_lookup_entries = Prover::get_max_num_lookup_entries();

    let num_used_lookup_buffer = individual_table_entries.len();
    for (_, buffer) in individual_table_entries
        .drain()
        .take(num_used_lookup_buffer)
    {
        reusable_buffer_for_lookup_entries.push(buffer);
    }

    for buffer in reusable_buffer_for_lookup_entries.iter_mut() {
        unsafe {
            buffer.set_len(0);
        }
        assert_eq!(buffer.capacity(), max_num_lookup_entries);
    }
    assert_eq!(reusable_buffer_for_lookup_entries.len(), num_lookup_tables);

    // we are recycling assembly so it is okey to construct an empty assembly with no preallocated memory
    // then assign pinned buffers to it
    let mut new_assembly = ProvingAssembly::new();
    new_assembly.aux_assingments = aux_assingments;
    new_assembly.aux_storage = aux_storage;
    new_assembly.reusable_buffer_for_lookup_entries = reusable_buffer_for_lookup_entries;

    new_assembly
}

#[cfg(feature = "legacy")]
pub(crate) fn create_prover_instances() -> Vec<(usize, Prover)> {
    vec![(0, Prover::new())]
}

#[cfg(not(feature = "legacy"))]
pub(crate) fn create_prover_instances() -> Vec<(usize, Prover)> {
    let actual_num_gpus = prover::gpu_prover::cuda_bindings::devices().unwrap() as usize;

    let info = prover::gpu_prover::cuda_bindings::device_info(0).unwrap();
    let available_memory_in_bytes = info.total as usize;
    let available_memory = available_memory_in_bytes / 1024 / 1024 / 1024;

    let num_gpus_per_prover_instance = if available_memory <= 40 { 2 } else { 1 };
    dbg!(available_memory);
    println!("actual num gpus: {}", actual_num_gpus);

    let mut prover_instances = vec![];
    for idx in 0..actual_num_gpus / num_gpus_per_prover_instance {
        let start = num_gpus_per_prover_instance * idx;
        let end = start + num_gpus_per_prover_instance;
        let device_ids: Vec<usize> = (start..end).collect();
        println!("loading prover {}", idx);
        prover_instances.push((idx, Prover::new_gpu_with_affinity(&device_ids)));
    }

    println!("num provers: {}", prover_instances.len());

    prover_instances
}
