#![cfg_attr(feature = "gpu", feature(allocator_api))]
#![feature(get_mut_unchecked)]
use cfg_if::*;

cfg_if! {
    if #[cfg(not(feature = "legacy"))]{
        pub mod gpu;
        pub use self::gpu::*;
        pub use gpu_prover;
        #[cfg(feature = "gpu")]
        pub use gpu_prover::cuda_bindings::CudaAllocator;
        pub use gpu_prover::AsyncSetup;
        pub use gpu_prover::cuda_bindings::GpuError;
        use gpu_prover::ManagerConfigs;
    }else{
        mod legacy;
        pub use self::legacy::*;
    }
}
#[cfg(feature = "gpu")]
use std::alloc::Global;

use bellman::bn256::{Bn256, Fr};

use bellman::plonk::better_better_cs::cs::Circuit;
use bellman::plonk::better_better_cs::cs::{
    SynthesisMode, SynthesisModeGenerateSetup, SynthesisModeProve, SynthesisModeTesting,
};
use bellman::plonk::better_better_cs::proof::Proof;
use bellman::plonk::commitments::transcript::Transcript;
use bellman::worker::Worker as OldWorker;
use bellman::Engine;

use std::path::Path;
use std::sync::{Arc, Mutex};

use bellman::{
    kate_commitment::{Crs, CrsForMonomialForm},
    plonk::better_better_cs::{
        cs::{
            Assembly as OriginalAssembly, PlonkCsWidth4WithNextStepAndCustomGatesParams,
            PlonkCsWidth4WithNextStepParams, ProvingAssembly as OriginalProvingAssembly,
            SetupAssembly as OriginalSetupAssembly, TrivialAssembly as OriginalTrivialAssembly,
        },
        gates::selector_optimized_with_d_next::SelectorOptimizedWidth4MainGateWithDNext,
        setup::{Setup as OriginalSetup, VerificationKey},
        verifier::verify,
    },
    SynthesisError,
};

#[cfg(not(feature = "gpu"))]
pub type Assembly<S> = OriginalAssembly<
    Bn256,
    PlonkCsWidth4WithNextStepAndCustomGatesParams,
    SelectorOptimizedWidth4MainGateWithDNext,
    S,
>;
#[cfg(feature = "gpu")]
pub type Assembly<S> = OriginalAssembly<
    Bn256,
    PlonkCsWidth4WithNextStepAndCustomGatesParams,
    SelectorOptimizedWidth4MainGateWithDNext,
    S,
    CudaAllocator,
>;

#[cfg(feature = "legacy")]
type Setup<C> = OriginalSetup<Bn256, C>;
#[cfg(not(feature = "legacy"))]
pub type Setup = AsyncSetup;

pub type TrivialAssembly = Assembly<SynthesisModeTesting>;
pub type ProvingAssembly = Assembly<SynthesisModeProve>;
pub type SetupAssembly = Assembly<SynthesisModeGenerateSetup>;

pub(crate) const CRS_FILE_ENV_VAR: &str = "CRS_FILE";
pub(crate) const DOMAIN_SIZE_LOG: usize = 26;
pub(crate) const DOMAIN_SIZE: usize = 1 << DOMAIN_SIZE_LOG;
pub(crate) const NUM_LOOKUP_TABLES: usize = 10;
pub(crate) const MAX_NUM_LOOKUP_ENTRIES: usize = 41000000; // ~40M
pub(crate) const MAX_NUM_VARIABLES: usize = 200000000; // ~200M

pub struct Prover {
    context: ProverContext,
}

unsafe impl Send for Prover {}
unsafe impl Sync for Prover {}

impl Prover {
    pub fn new() -> Self {
        println!("num physical cores: {}\n", num_cpus::get_physical());
        let ctx = ProverContext::init();
        Self { context: ctx }
    }

    pub fn new_assembly() -> TrivialAssembly {
        TrivialAssembly::new()
    }

    pub fn new_proving_assembly() -> ProvingAssembly {
        ProvingAssembly::new_specialized_for_proving_assembly_and_state_4(
            DOMAIN_SIZE,
            MAX_NUM_VARIABLES,
            NUM_LOOKUP_TABLES,
            MAX_NUM_LOOKUP_ENTRIES,
        )
    }

    pub fn new_setup_assembly() -> SetupAssembly {
        SetupAssembly::new()
    }

    pub fn new_worker(num_cores: Option<usize>) -> OldWorker {
        match num_cores {
            Some(num_cores) => OldWorker::new_with_cpus(num_cores),
            None => OldWorker::new(),
        }
    }

    pub const fn get_max_domain_size_log() -> usize {
        DOMAIN_SIZE_LOG
    }

    pub const fn get_max_domain_size() -> usize {
        DOMAIN_SIZE
    }
    pub const fn get_max_num_variables() -> usize {
        MAX_NUM_VARIABLES
    }

    pub const fn get_num_lookup_tables() -> usize {
        NUM_LOOKUP_TABLES
    }

    pub const fn get_max_num_lookup_entries() -> usize {
        MAX_NUM_LOOKUP_ENTRIES
    }

    pub fn new_with_crs(crs: &Path) -> Self {
        #[cfg(not(feature = "legacy"))]
        let ctx = ProverContext::init_with_crs(None, crs);
        #[cfg(feature = "legacy")]
        let ctx = ProverContext::init_with_crs(crs);
        Self { context: ctx }
    }

    pub fn new_with_dummy_crs() -> Self {
        let ctx = ProverContext::init_with_dummy_crs();
        Self { context: ctx }
    }

    /// This function is useful for proof generation of basic circuits
    /// where they require `RescueTranscriptForRecursion`
    pub fn create_proof_with_assembly_and_transcript<C: Circuit<Bn256>, T: Transcript<Fr>>(
        &mut self,
        assembly: &TrivialAssembly,
        transcript_params: Option<T::InitializationParameters>,
    ) -> Result<Proof<Bn256, C>, SynthesisError> {
        self.inner_create_proof::<_, T, _>(assembly, None, transcript_params)
    }

    pub fn create_proof_with_proving_assembly_and_transcript<
        C: Circuit<Bn256>,
        T: Transcript<Fr>,
    >(
        &mut self,
        assembly: &ProvingAssembly,
        #[cfg(feature = "legacy")] setup: &Setup<C>,
        #[cfg(feature = "gpu")] setup: &AsyncSetup, // uses Default = CudaAllocator
        #[cfg(feature = "gpu_no_alloc")] setup: &AsyncSetup,
        transcript_params: Option<T::InitializationParameters>,
    ) -> Result<Proof<Bn256, C>, SynthesisError> {
        self.inner_create_proof::<_, T, _>(assembly, Some(setup), transcript_params)
    }

    // This function should be called for already synhtesized circuit.
    // Use as many core as you want for Worker which can be created `Prover::new_worker()`
    pub fn create_vk_from_assembly<C: Circuit<Bn256>, S: SynthesisMode>(
        &mut self,
        assembly: &Assembly<S>,
    ) -> Result<VerificationKey<Bn256, C>, SynthesisError> {
        assert!(S::PRODUCE_SETUP);

        assert!(assembly.is_finalized);
        assert!(assembly.n() + 1 <= Prover::get_max_domain_size());

        let vk = self.inner_create_vk_from_assembly(assembly)?;

        Ok(vk)
    }

    #[cfg(feature = "legacy")]
    pub fn create_setup_from_assembly<C: Circuit<Bn256>, S: SynthesisMode>(
        &mut self,
        assembly: &Assembly<S>,
    ) -> Result<Setup<C>, SynthesisError> {
        assert!(assembly.is_finalized);
        dbg!(Prover::get_max_domain_size());
        dbg!(assembly.n() + 1);
        assert!(assembly.n() + 1 <= Prover::get_max_domain_size());

        let setup = self.inner_create_setup_from_assembly(assembly)?;

        Ok(setup)
    }

    #[cfg(not(feature = "legacy"))]
    pub fn create_setup_from_assembly<C: Circuit<Bn256>, S: SynthesisMode>(
        &mut self,
        assembly: &Assembly<S>,
    ) -> Result<AsyncSetup, SynthesisError> {
        assert!(assembly.is_finalized);
        dbg!(Prover::get_max_domain_size());
        dbg!(assembly.n() + 1);
        assert!(assembly.n() + 1 <= Prover::get_max_domain_size());

        let setup = self.inner_create_setup_from_assembly::<C, _>(assembly)?;

        Ok(setup)
    }

    pub fn verify_proof<C: Circuit<Bn256>, T: Transcript<Fr>>(
        proof: &Proof<Bn256, C>,
        vk: &VerificationKey<Bn256, C>,
        transcript_params: Option<T::InitializationParameters>,
    ) -> Result<bool, SynthesisError> {
        assert_eq!(vk.n, proof.n);
        assert_eq!(vk.num_inputs, proof.inputs.len());

        let start = std::time::Instant::now();
        let valid = verify::<Bn256, C, T>(&vk, &proof, transcript_params)?;
        println!("verification takes {:?}", start.elapsed());
        if !valid {
            return Err(SynthesisError::Unsatisfiable);
        }

        Ok(valid)
    }
}
