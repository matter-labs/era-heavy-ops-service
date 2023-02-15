use std::{
    io::Write,
    sync::mpsc::{Receiver, Sender},
};

use prover::ProvingAssembly;
use zkevm_test_harness::{
    bellman::plonk::better_better_cs::{
        cs::Gate, data_structures::PolyIdentifier,
        gates::selector_optimized_with_d_next::SelectorOptimizedWidth4MainGateWithDNext,
    },
    franklin_crypto::plonk::circuit::custom_rescue_gate::Rescue5CustomGate,
};

use super::*;
use crate::run_prover::{
    recycle_assembly, ThreadGuard, ENCODER_THREAD_HANDLE, SYNTH_THREAD_HANDLE,
};

#[cfg(feature = "gpu")]
use std::alloc::Allocator;

#[cfg(not(feature = "gpu"))]
macro_rules! new_vec_with_allocator {
    ($capacity:expr) => {
        Vec::with_capacity($capacity)
    };
}

#[cfg(feature = "gpu")]
macro_rules! new_vec_with_allocator {
    ($capacity:expr) => {
        Vec::with_capacity_in($capacity, prover::gpu_prover::cuda_bindings::CudaAllocator)
    };
}

struct SynthesizerContext {
    encoding_senders: Vec<Sender<(usize, u8, ProvingAssembly)>>,
    encoding_receivers: Vec<Receiver<(usize, u8, ProvingAssembly)>>,
    assembly_sender: Sender<ProvingAssembly>,
    assembly_receiver: Receiver<ProvingAssembly>,
    thread_status_sender: Sender<u8>,
    thread_status_receiver: Receiver<u8>,
    report_sender: Sender<JobResult>,
    report_receiver: Receiver<JobResult>,
}
unsafe impl Send for SynthesizerContext {}
unsafe impl Sync for SynthesizerContext {}

impl SynthesizerContext {
    fn new(num_senders: usize) -> Self {
        let mut encoding_senders = vec![];
        let mut encoding_receivers = vec![];
        for idx in 0..num_senders {
            let (sender, receiver) = std::sync::mpsc::channel();
            encoding_senders.push(sender);
            encoding_receivers.push(receiver);
        }
        let (assembly_sender, assembly_receiver) = std::sync::mpsc::channel();
        let (report_sender, report_receiver) = std::sync::mpsc::channel();
        let (thread_status_sender, thread_status_receiver) = std::sync::mpsc::channel();

        Self {
            encoding_senders,
            encoding_receivers,
            assembly_sender,
            assembly_receiver,
            report_sender,
            report_receiver,
            thread_status_sender,
            thread_status_receiver,
        }
    }
}

const REMOTE_SYNTH_UTILITY_THREADS: u8 = 2;

pub trait EncodedArtifactSender: Send + Sync {
    fn send(
        &mut self,
        encoded_artifact: Box<dyn Read>,
        circuit_id: u8,
    ) -> Result<(), std::io::Error>;
}

pub fn run_remote_synthesizer<
    JM: JobManager + 'static,
    JR: JobReporter + 'static,
    AS: EncodedArtifactSender + 'static,
    P: Params,
>(
    mut job_manager: JM,
    mut job_reporter: JR,
    artifact_senders: Vec<AS>,
    params: P,
) {
    assert!(
        (params.number_of_parallel_synthesis() + REMOTE_SYNTH_UTILITY_THREADS) as usize
            <= num_cpus::get_physical()
    );
    let num_senders = artifact_senders.len();
    let ctx = SynthesizerContext::new(num_senders);
    let ctx = Arc::new(ctx);

    // create reusable proving assemblies
    for _ in 0..params.number_of_parallel_synthesis() {
        ctx.assembly_sender
            .send(Prover::new_proving_assembly())
            .unwrap();
    }

    // utility thread that encodes assemblies
    assembly_encoder(ctx.clone(), artifact_senders);

    'outer: loop {
        // process collected reports here
        // note: avoid locking for job reporting and use a channel instead
        for report in ctx.report_receiver.try_iter() {
            job_reporter.send_report(report);
        }

        // check whether infallible threads are alive or not
        for thread_status in ctx.thread_status_receiver.try_iter() {
            match thread_status {
                ENCODER_THREAD_HANDLE => {
                    println!(
                        "encoder thread panicked at {:?}, respawning",
                        std::time::Instant::now()
                    );
                }
                _ => (),
            }
        }

        // block neither for job nor for assembly because we need to process the reports from the threads
        let mut assembly: ProvingAssembly = if let Ok(assembly) = ctx.assembly_receiver.try_recv() {
            assembly
        } else {
            std::thread::sleep(params.polling_duration());
            continue 'outer;
        };

        let (job_id, circuit) = if let Some(job) = job_manager.try_get_next_job() {
            job
        } else {
            ctx.assembly_sender.send(assembly).unwrap();
            std::thread::sleep(params.polling_duration());
            continue 'outer;
        };

        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let guard = ThreadGuard::new(
                SYNTH_THREAD_HANDLE,
                job_id,
                ctx.report_sender.clone(),
                ctx.thread_status_sender.clone(),
            );
            let circuit_id = circuit.numeric_circuit_type();
            println!("synthesizing circuit {}", circuit.short_description());

            let synthesized = std::time::Instant::now();
            circuit.synthesize(&mut assembly).unwrap();
            ctx.report_sender
                .send(JobResult::Synthesized(job_id, synthesized.elapsed()))
                .unwrap();

            let chunk_size = 20 / ctx.encoding_senders.len();
            let sender_idx = circuit_id as usize / chunk_size;

            if let Err(e) = ctx.encoding_senders[sender_idx].send((job_id, circuit_id, assembly)) {
                ctx.report_sender
                    .send(JobResult::Failure(
                        job_id,
                        format!("encoder handler failed: {}", e),
                    ))
                    .unwrap();
            };
        });
    }
}

fn assembly_encoder<AS: EncodedArtifactSender + 'static>(
    ctx: Arc<SynthesizerContext>,
    mut artifact_senders: Vec<AS>,
) {
    for (sender_idx, mut artifact_sender) in artifact_senders.into_iter().enumerate() {
        let ctx = ctx.clone();
        std::thread::spawn(move || loop {
            let (job_id, circuit_id, assembly) = ctx.encoding_receivers[sender_idx].recv().unwrap();
            let guard = ThreadGuard::new(
                ENCODER_THREAD_HANDLE,
                job_id,
                ctx.report_sender.clone(),
                ctx.thread_status_sender.clone(),
            );
            let capacity = calculate_serialization_capacity_for_proving_assembly();

            let start = std::time::Instant::now();
            let mut buffer = new_vec_with_allocator!(capacity);
            serialize_job(&assembly, job_id, circuit_id, &mut buffer);
            let recycled_assembly = recycle_assembly(assembly);
            ctx.assembly_sender.send(recycled_assembly).unwrap();

            let assembly_encoding = std::io::Cursor::new(buffer);
            let assembly_encoding = Box::new(assembly_encoding);

            let assembly_encoded = start.elapsed();
            ctx.report_sender
                .send(JobResult::AssemblyEncoded(job_id, assembly_encoded))
                .unwrap();

            let assembly_transferred = std::time::Instant::now();
            artifact_sender.send(assembly_encoding, circuit_id).unwrap();
            ctx.report_sender
                .send(JobResult::AssemblyTransferred(
                    job_id,
                    assembly_transferred.elapsed(),
                ))
                .unwrap();
        });
    }
}

pub fn serialize_job<W: Write>(
    assembly: &ProvingAssembly,
    job_id: usize,
    circuit_id: u8,
    buffer: &mut W,
) {
    buffer.write_all(&job_id.to_le_bytes()).unwrap();
    buffer.write(&[circuit_id]).unwrap();
    custom_assembly_serialization(assembly, buffer);
}

fn deserialize_job_with_guard<R: Read>(
    buffer: &mut R,
    assembly: &mut ProvingAssembly,
    ctx: Arc<super::run_prover::ProverContext>,
) -> (usize, u8) {
    add_gates_into_assembly(assembly);

    let mut job_id_bytes = [0u8; 8];
    buffer.read_exact(&mut job_id_bytes[..]).unwrap();
    let job_id = usize::from_le_bytes(job_id_bytes);
    let guard = ThreadGuard::new(
        ENCODER_THREAD_HANDLE,
        job_id,
        ctx.report_sender.clone(),
        ctx.thread_status_sender.clone(),
    );
    let mut circuit_id = [0u8; 1];
    buffer.read_exact(&mut circuit_id[..]).unwrap();
    let circuit_id = circuit_id[0];
    custom_assembly_deserialization(buffer, assembly);

    (job_id, circuit_id)
}
pub fn deserialize_job<R: Read>(buffer: &mut R, assembly: &mut ProvingAssembly) -> (usize, u8) {
    add_gates_into_assembly(assembly);

    let mut job_id_bytes = [0u8; 8];
    buffer.read_exact(&mut job_id_bytes[..]).unwrap();
    let job_id = usize::from_le_bytes(job_id_bytes);
    let mut circuit_id = [0u8; 1];
    buffer.read_exact(&mut circuit_id[..]).unwrap();
    let circuit_id = circuit_id[0];
    custom_assembly_deserialization(buffer, assembly);

    (job_id, circuit_id)
}

fn add_gates_into_assembly(assembly: &mut ProvingAssembly) {
    let main_gate = SelectorOptimizedWidth4MainGateWithDNext;
    let rescue_gate = Rescue5CustomGate;

    assembly.sorted_gates.push(main_gate.into_internal());
    assembly.sorted_gates.push(rescue_gate.into_internal());
}

pub fn calculate_serialization_capacity_for_proving_assembly() -> usize {
    // aux assignments
    let assembly = Prover::new_proving_assembly();
    let mut capacity = assembly.aux_assingments.capacity() * 32;

    // variables
    capacity += assembly
        .aux_storage
        .state_map
        .iter()
        .map(|(_, v)| v.capacity() * 16)
        .fold(0, |acc, x| acc + x);

    // table names
    let num_bytes_for_single_table_name = 1 << 16; // that is huge, use it until proper measurement
    let max_num_tables = Prover::get_num_lookup_tables();
    capacity += max_num_tables * num_bytes_for_single_table_name;

    // canonical tables
    let num_bytes_for_single_canonical_table = 327680 * 3 * 32;
    capacity += max_num_tables * num_bytes_for_single_canonical_table;

    // table indexes
    let max_num_lookup_entries = Prover::get_max_num_lookup_entries();
    capacity += max_num_tables * max_num_lookup_entries * 4;

    capacity
}

pub fn custom_assembly_serialization<W: Write>(assembly: &ProvingAssembly, buffer: &mut W) {
    serialize_assignments(assembly, buffer);
    serialize_variables(assembly, buffer);
    serialize_tables(assembly, buffer);
}

pub fn custom_assembly_deserialization<R: Read>(encoding: &mut R, assembly: &mut ProvingAssembly) {
    deserialize_assignments(encoding, assembly);
    deserialize_variables(encoding, assembly);
    deserialize_tables(encoding, assembly);
}

fn serialize_assignments<W: Write>(assembly: &ProvingAssembly, buffer: &mut W) {
    let num_aux_gates = assembly.num_aux_gates.to_le_bytes();
    serialize_generic(&num_aux_gates, buffer);
    let num_input_gates = assembly.num_input_gates.to_le_bytes();
    serialize_generic(&num_input_gates, buffer);
    let num_inputs = assembly.num_inputs.to_le_bytes();
    serialize_generic(&num_inputs, buffer);
    let num_aux = assembly.num_aux.to_le_bytes();
    serialize_generic(&num_aux, buffer);
    serialize_generic(&assembly.input_assingments, buffer);
    serialize_generic(&assembly.aux_assingments, buffer);
}

fn deserialize_assignments<R: Read>(encoding: &mut R, assembly: &mut ProvingAssembly) {
    let mut num_aux_gates = vec![0u8; 8];
    deserialize_generic(encoding, &mut num_aux_gates);
    assembly.num_aux_gates = usize::from_le_bytes(num_aux_gates.try_into().unwrap());
    let mut num_input_gates = vec![0u8; 8];
    deserialize_generic(encoding, &mut num_input_gates);
    assembly.num_input_gates = usize::from_le_bytes(num_input_gates.try_into().unwrap());
    let mut num_inputs = vec![0u8; 8];
    deserialize_generic(encoding, &mut num_inputs);
    assembly.num_inputs = usize::from_le_bytes(num_inputs.try_into().unwrap());
    let mut num_aux = vec![0u8; 8];
    deserialize_generic(encoding, &mut num_aux);
    assembly.num_aux = usize::from_le_bytes(num_aux.try_into().unwrap());

    deserialize_generic(encoding, &mut assembly.input_assingments);
    assert_eq!(assembly.num_inputs, assembly.num_input_gates);

    deserialize_generic(encoding, &mut assembly.aux_assingments);
}

fn serialize_variables<W: Write>(assembly: &ProvingAssembly, buffer: &mut W) {
    for idx in 0..4 {
        let poly_idx = PolyIdentifier::VariablesPolynomial(idx);
        let variables = assembly.aux_storage.state_map.get(&poly_idx).unwrap();
        serialize_generic(variables, buffer);
    }
    for idx in 0..4 {
        let poly_idx = PolyIdentifier::VariablesPolynomial(idx);
        let variables = assembly.inputs_storage.state_map.get(&poly_idx).unwrap();
        serialize_generic(variables, buffer);
    }
}

fn deserialize_variables<R: Read>(encoding: &mut R, assembly: &mut ProvingAssembly) {
    for idx in 0..4 {
        let idx = PolyIdentifier::VariablesPolynomial(idx);
        let variables = assembly
            .aux_storage
            .state_map
            .entry(idx)
            .or_insert(new_vec_with_allocator!(0));
        deserialize_generic(encoding, variables);
    }
    for idx in 0..4 {
        let idx = PolyIdentifier::VariablesPolynomial(idx);
        let variables = assembly
            .inputs_storage
            .state_map
            .entry(idx)
            .or_insert(new_vec_with_allocator!(0));
        deserialize_generic(encoding, variables);
    }
}

fn serialize_tables<W: Write>(assembly: &ProvingAssembly, buffer: &mut W) {
    let table_names = assembly.known_table_names.clone();

    let num_tables = table_names.len();
    if num_tables == 0 {
        return;
    }
    buffer.write(&num_tables.to_le_bytes()).unwrap();

    for table_name in table_names.iter() {
        let mut encoding = bincode::serialize(&table_name).unwrap();
        serialize_generic(&mut encoding, buffer);
    }

    let mut table_ids = vec![];
    for table_name in table_names.iter() {
        let table_id = assembly.known_table_ids.get(table_name).unwrap();
        table_ids.push(table_id.clone());
    }

    serialize_generic(&table_ids, buffer);

    for table_name in table_names.iter() {
        let table = assembly
            .individual_table_canonical_sorted_entries
            .get(table_name)
            .unwrap();

        serialize_generic(table, buffer)
    }

    for table_name in table_names.iter() {
        let table = assembly.individual_table_entries.get(table_name).unwrap();
        serialize_generic(table, buffer)
    }
}

fn deserialize_tables<R: Read>(encoding: &mut R, assembly: &mut ProvingAssembly) {
    let ProvingAssembly {
        individual_table_canonical_sorted_entries,
        individual_table_entries,
        reusable_buffer_for_lookup_entries,
        ..
    } = assembly;
    assert!(individual_table_canonical_sorted_entries.is_empty());
    assert!(individual_table_entries.is_empty());

    let mut num_tables_as_bytes = [0u8; 8];
    encoding.read(&mut num_tables_as_bytes[..]).unwrap();
    let num_tables = usize::from_le_bytes(num_tables_as_bytes);

    if num_tables == 0 {
        return;
    }

    for _ in 0..num_tables {
        let mut table_name_encoding: Vec<u8> = vec![];
        deserialize_generic(encoding, &mut table_name_encoding);
        let table_name: String = bincode::deserialize(&table_name_encoding).unwrap();
        assembly.known_table_names.push(table_name);
    }
    let table_names = assembly.known_table_names.clone();

    let mut table_ids = vec![];
    deserialize_generic(encoding, &mut table_ids);
    assert_eq!(table_names.len(), table_ids.len());

    for (table_name, table_id) in table_names.iter().zip(table_ids) {
        assembly
            .known_table_ids
            .insert(table_name.clone(), table_id);
    }
    assert_eq!(assembly.known_table_ids.len(), num_tables);

    for table_name in table_names.iter() {
        let table = individual_table_canonical_sorted_entries
            .entry(table_name.clone())
            .or_insert_with(|| Vec::with_capacity(0));
        deserialize_generic(encoding, table);
    }
    assert_eq!(individual_table_canonical_sorted_entries.len(), num_tables);

    for table_name in table_names.iter() {
        let table = individual_table_entries
            .entry(table_name.clone())
            .or_insert_with(|| {
                if let Some(buf) = reusable_buffer_for_lookup_entries.pop() {
                    buf
                } else {
                    new_vec_with_allocator!(0)
                }
            });
        deserialize_generic(encoding, table);
    }
    assert_eq!(individual_table_entries.len(), num_tables);
}

fn serialize_generic<T, W: Write>(data: &[T], buffer: &mut W) {
    let unit_len = std::mem::size_of::<T>();
    let ptr = data.as_ptr() as *const u8;
    let len = data.len() * unit_len;
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    buffer.write_all(len.to_le_bytes().as_ref()).unwrap();
    buffer.write_all(slice).unwrap();
}

#[cfg(feature = "gpu")]
fn deserialize_generic<T, A: Allocator, R: Read>(encoding: &mut R, buffer: &mut Vec<T, A>) {
    let unit_len = std::mem::size_of::<T>();

    let mut buf_num_bytes = [0u8; 8];
    encoding.read_exact(&mut buf_num_bytes).unwrap();
    let len = usize::from_le_bytes(buf_num_bytes);
    let actual_len = len / unit_len;
    assert!(len % unit_len == 0);
    if buffer.capacity() < actual_len {
        buffer.reserve(actual_len - buffer.capacity());
    }
    let ptr = buffer.as_mut_ptr() as *mut u8;
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    unsafe {
        buffer.set_len(actual_len);
    }
    encoding.read_exact(slice).unwrap();
}

#[cfg(not(feature = "gpu"))]
fn deserialize_generic<T, R: Read>(encoding: &mut R, buffer: &mut Vec<T>) {
    let unit_len = std::mem::size_of::<T>();

    let mut buf_num_bytes = [0u8; 8];
    encoding.read_exact(&mut buf_num_bytes).unwrap();
    let len = usize::from_le_bytes(buf_num_bytes);
    let actual_len = len / unit_len;
    assert!(len % unit_len == 0);
    if buffer.capacity() < actual_len {
        buffer.reserve(actual_len - buffer.capacity());
    }
    let ptr = buffer.as_mut_ptr() as *mut u8;
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    unsafe {
        buffer.set_len(actual_len);
    }
    encoding.read_exact(slice).unwrap();
}
