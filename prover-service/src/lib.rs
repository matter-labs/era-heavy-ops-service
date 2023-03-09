#![feature(get_mut_unchecked)]
#![cfg_attr(feature = "gpu", feature(allocator_api))]
pub mod remote_synth;
pub mod run_prover;
pub(crate) mod setup;
pub mod simple;
#[cfg(test)]
mod tests;
pub mod utils;

pub use bellman::bn256::{Bn256, Fr};

use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use zkevm_test_harness::abstract_zksync_circuit::concrete_circuits::ZkSyncProof;
pub use zkevm_test_harness::abstract_zksync_circuit::concrete_circuits::ZkSyncVerificationKey;
use zkevm_test_harness::bellman::plonk::better_better_cs::cs::Circuit;
use zkevm_test_harness::bellman::plonk::better_better_cs::setup::Setup as OriginalSetup;
use zkevm_test_harness::bellman::SynthesisError;
use zkevm_test_harness::franklin_crypto::bellman;
use zkevm_test_harness::sync_vm::recursion::{
    get_prefered_rns_params, RescueTranscriptForRecursion,
};
use zkevm_test_harness::{
    abstract_zksync_circuit::concrete_circuits::ZkSyncCircuit as OriginalZkSyncCircuit,
    witness::oracle::VmWitnessOracle,
};

pub type ZkSyncCircuit = OriginalZkSyncCircuit<Bn256, VmWitnessOracle<Bn256>>;
pub use prover;
use prover::{Prover, ProvingAssembly};

pub use utils::*;

#[cfg(not(feature = "legacy"))]
use prover::AsyncSetup;
#[cfg(feature = "gpu")]
use prover::CudaAllocator;

#[cfg(feature = "legacy")]
pub type Setup =
    zkevm_test_harness::bellman::plonk::better_better_cs::setup::Setup<Bn256, ZkSyncCircuit>;
#[cfg(not(feature = "legacy"))]
pub type Setup = AsyncSetup;

#[derive(Debug)]
pub enum ProverError {
    #[cfg(any(feature = "gpu", feature = "gpu_no_alloc"))]
    GpuError(prover::GpuError),
    SynthesisError(SynthesisError),
    Other(String),
}

pub type JobId = usize;
pub type ProverId = usize;

#[derive(Clone)]
pub enum JobResult {
    Synthesized(JobId, std::time::Duration),
    AssemblyFinalized(JobId, std::time::Duration),
    SetupLoaded(JobId, std::time::Duration, bool),
    ProofGenerated(JobId, std::time::Duration, ZkSyncProof<Bn256>, usize),
    Failure(JobId, String),
    AssemblyEncoded(JobId, std::time::Duration),
    AssemblyDecoded(JobId, std::time::Duration),
    AssemblyTransferred(JobId, std::time::Duration),
    FailureWithDebugging(JobId, u8, Vec<u8>, String),
    ProverWaitedIdle(ProverId, std::time::Duration),
    SetupLoaderWaitedIdle(std::time::Duration),
    SchedulerWaitedIdle(std::time::Duration),
}

impl std::fmt::Debug for JobResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Synthesized(arg0, arg1) => f
                .debug_tuple("Synthesized")
                .field(arg0)
                .field(&arg1)
                .finish(),
            Self::ProverWaitedIdle(arg0, arg1) => f
                .debug_tuple("ProverWaitedIdle")
                .field(arg0)
                .field(&arg1)
                .finish(),
            Self::SetupLoaderWaitedIdle(arg0) => {
                f.debug_tuple("SetupLoaderWaitedIdle").field(arg0).finish()
            }
            Self::SchedulerWaitedIdle(arg0) => {
                f.debug_tuple("SchedulerWaitedIdle").field(arg0).finish()
            }
            Self::AssemblyFinalized(arg0, arg1) => f
                .debug_tuple("AssemblyFinalized")
                .field(arg0)
                .field(&arg1)
                .finish(),
            Self::AssemblyEncoded(arg0, arg1) => f
                .debug_tuple("AssemblyEncoded")
                .field(arg0)
                .field(&arg1)
                .finish(),
            Self::AssemblyDecoded(arg0, arg1) => f
                .debug_tuple("AssemblyDecoded")
                .field(arg0)
                .field(&arg1)
                .finish(),
            Self::AssemblyTransferred(arg0, arg1) => f
                .debug_tuple("AssemblyTransferred")
                .field(arg0)
                .field(&arg1)
                .finish(),
            Self::SetupLoaded(arg0, arg1, arg2) => f
                .debug_tuple("SetupLoaded")
                .field(arg0)
                .field(&arg1)
                .field(&arg2)
                .finish(),
            Self::ProofGenerated(arg0, arg1, _, arg3) => f
                .debug_tuple("ProofGenerated")
                .field(arg3)
                .field(arg0)
                .field(&arg1)
                .finish(),
            Self::Failure(arg0, arg1) => f.debug_tuple("Failure").field(arg0).field(arg1).finish(),
            Self::FailureWithDebugging(arg0, arg1, arg2, arg3) => {
                f.debug_tuple("Failure").field(arg0).field(arg1).finish()
            }
        }
    }
}

pub trait JobReporter: Send {
    fn send_report(&mut self, report: JobResult);
}

pub trait JobManager: Send {
    /// This is a blocking function if there is no job then wait
    fn get_next_job(&mut self) -> (JobId, ZkSyncCircuit);
    /// This is a blocking function if there is no job for the given circuit type then wait
    fn get_next_job_by_circuit(&mut self, circuit_id: u8) -> (JobId, ZkSyncCircuit);
    /// This is a non-blocking function that yields some Job or None
    fn try_get_next_job(&mut self) -> Option<(JobId, ZkSyncCircuit)>;
    /// This is a non-blocking function that  yields  some Job for the given circuit type or None
    fn try_get_next_job_by_circuit(&mut self, circuit_id: u8) -> Option<(JobId, ZkSyncCircuit)>;
}

pub enum Encoding {
    Json,
    Binary,
}
pub trait ArtifactProvider: Send + Sync {
    type ArtifactError: std::fmt::Debug + std::string::ToString;
    fn get_setup(&self, circuit_id: u8) -> Result<Box<dyn Read>, Self::ArtifactError>;
    fn get_vk(&self, circuit_id: u8) -> Result<ZkSyncVerificationKey<Bn256>, Self::ArtifactError>;
}

pub trait RemoteSynthesizer: Send {
    fn try_next(&mut self) -> Option<Box<dyn Read + Send + Sync>>;
}

pub trait Params: Send + Sync {
    fn number_of_parallel_synthesis(&self) -> u8;
    fn number_of_setup_slots(&self) -> u8;
    fn polling_duration(&self) -> Duration {
        Duration::from_millis(1)
    }
}
