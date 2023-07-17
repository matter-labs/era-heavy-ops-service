use std::path::Path;

use super::*;

use bellman::{
    bn256::{Bn256, Fr, G2Affine, G2},
    compact_bn256::Bn256 as CompactBn256,
    plonk::{better_better_cs::cs::SynthesisMode, commitments::transcript::Transcript},
    CurveProjective, EncodedPoint,
};
use gpu_prover::{create_proof, DeviceMemoryManager, ManagerConfigs};

struct GpuProverConfigFor1x80;

impl ManagerConfigs for GpuProverConfigFor1x80 {
    const NUM_GPUS_LOG: usize = 0;
    const FULL_SLOT_SIZE_LOG: usize = DOMAIN_SIZE_LOG;
    const NUM_SLOTS: usize = 28;
    const NUM_HOST_SLOTS: usize = 2;
}
struct GpuProverConfigFor2x40;

impl ManagerConfigs for GpuProverConfigFor2x40 {
    const NUM_GPUS_LOG: usize = 1;
    const FULL_SLOT_SIZE_LOG: usize = DOMAIN_SIZE_LOG;
    const NUM_SLOTS: usize = 28;
    const NUM_HOST_SLOTS: usize = 2;
}

pub const NUM_LOOKUP_TABLE_NONZERO_VALUES: usize = 1 << 19;

type MemoryManager1x80 = DeviceMemoryManager<Fr, GpuProverConfigFor1x80>;
type MemoryManager2x40 = DeviceMemoryManager<Fr, GpuProverConfigFor2x40>;

enum ManagerType {
    Forty(MemoryManager2x40),
    Eighty(MemoryManager1x80),
}

impl ManagerType {
    fn num_gpus(&self) -> usize {
        match self {
            ManagerType::Forty(_) => 2,
            ManagerType::Eighty(_) => 1,
        }
    }
}

pub struct ProverContext {
    worker: OldWorker,
    manager: ManagerType,
    crs_for_verification: Crs<Bn256, CrsForMonomialForm>,
}

impl ProverContext {
    pub fn init_with_affinity(device_ids: &[usize]) -> Self {
        let crs_file_str = std::env::var(CRS_FILE_ENV_VAR).expect("crs file env var");
        let crs_file_path = std::path::Path::new(&crs_file_str);
        Self::init_with_crs(Some(device_ids), crs_file_path)
    }
    pub fn init() -> Self {
        let crs_file_str = std::env::var(CRS_FILE_ENV_VAR).expect("crs file env var");
        let crs_file_path = std::path::Path::new(&crs_file_str);
        Self::init_with_crs(None, crs_file_path)
    }

    pub fn init_with_crs(device_ids: Option<&[usize]>, crs_file_path: &Path) -> Self {
        let worker = OldWorker::new();
        let crs_file = std::fs::File::open(&crs_file_path).expect("crs file to open");
        let crs = Crs::read(&crs_file).expect("crs file for bases");
        let crs_size = Prover::get_max_domain_size();
        assert_eq!(
            crs.g1_bases.len(),
            crs_size,
            "Proving key has {} but expected {}",
            crs.g1_bases.len(),
            crs_size,
        );

        let mut crs_for_verification = Crs::<Bn256, CrsForMonomialForm>::dummy_crs(1);
        crs_for_verification.g2_monomial_bases = Arc::new(crs.g2_monomial_bases.as_ref().clone());

        let transformed_crs = transform_crs(&crs, &worker);
        let manager = Self::init_manager(device_ids, &transformed_crs);
        Self {
            worker: OldWorker::new_with_cpus(2),
            manager,
            crs_for_verification,
        }
    }

    pub fn init_with_dummy_crs() -> Self {
        let worker = OldWorker::new();
        let crs_size = Prover::get_max_domain_size();
        let crs = Crs::<Bn256, CrsForMonomialForm>::dummy_crs(crs_size);
        assert_eq!(crs.g1_bases.len(), crs_size);

        let mut crs_for_verification = Crs::<Bn256, CrsForMonomialForm>::dummy_crs(1);
        crs_for_verification.g2_monomial_bases = Arc::new(crs.g2_monomial_bases.as_ref().clone());

        let transformed_crs = transform_crs(&crs, &worker);
        let manager = Self::init_manager(None, &transformed_crs);
        Self {
            worker: OldWorker::new_with_cpus(2),
            manager,
            crs_for_verification,
        }
    }

    fn init_manager(
        device_ids: Option<&[usize]>,
        crs: &Crs<CompactBn256, CrsForMonomialForm>,
    ) -> ManagerType {
        let info = gpu_prover::cuda_bindings::device_info(0).unwrap();
        let available_memory_in_bytes = info.total;
        let available_memory = available_memory_in_bytes / 1024 / 1024 / 1024;

        let device_ids = match device_ids {
            Some(device_ids) => device_ids.to_vec(),
            None => {
                let mut new_device_ids = vec![0];
                if available_memory <= 40 {
                    new_device_ids.push(1)
                }
                new_device_ids
            }
        };
        println!("num gpus: {}", device_ids.len());
        let manager = if available_memory <= 40 {
            assert_eq!(
                device_ids.len(),
                2,
                "prover supports only 2 gpus per prover instance"
            );
            let manager = MemoryManager2x40::init(&device_ids, &crs.g1_bases[..]).unwrap();
            ManagerType::Forty(manager)
        } else {
            assert_eq!(
                device_ids.len(),
                1,
                "prover supports only 1 gpu per prover instance"
            );
            let manager = MemoryManager1x80::init(&device_ids, &crs.g1_bases[..]).unwrap();
            ManagerType::Eighty(manager)
        };

        manager
    }
}

impl Prover {
    pub fn config() -> Vec<u64> {
        let actual_num_gpus = gpu_prover::cuda_bindings::devices().unwrap() as usize;
        let mut result = vec![];
        for device_id in 0..actual_num_gpus {
            let info = gpu_prover::cuda_bindings::device_info(device_id as i32).unwrap();
            result.push(info.total);
        }

        result
    }

    pub fn new_gpu_with_affinity(device_ids: &[usize]) -> Self {
        println!("num physical cores: {}\n", num_cpus::get_physical());
        let ctx = ProverContext::init_with_affinity(device_ids);
        Self { context: ctx }
    }

    pub fn inner_create_proof<C: Circuit<Bn256>, T: Transcript<Fr>, S: SynthesisMode + 'static>(
        &mut self,
        assembly: &Assembly<S>,
        #[cfg(feature = "gpu_no_alloc")] setup: Option<&AsyncSetup>,
        #[cfg(feature = "gpu")] setup: Option<&AsyncSetup<CudaAllocator>>,
        transcript_params: Option<T::InitializationParameters>,
    ) -> Result<Proof<Bn256, C>, SynthesisError> {
        assert!(S::PRODUCE_WITNESS);
        assert!(assembly.is_finalized);
        let n = assembly.n();
        let num_inputs = assembly.num_inputs;
        assert_eq!(
            n,
            Prover::get_max_domain_size() - 1,
            "GPU prover specialized for circuit 2^{}",
            Prover::get_max_domain_size()
        );
        // We should distinguish between ProvingAssembly and TrivialAssembly
        let mut empty_setup = AsyncSetup::empty();
        let pinned_setup = if S::PRODUCE_SETUP {
            // TrivialAssembly already contains setup values
            // so that we don't do any pre-computation
            // instead we compute and construct them  on the fly on the gpu
            &mut empty_setup
        } else {
            // ProvingAssembly contains only witness values
            // For setup values we need a precomputed setup
            // TODO: Setup will already be in the pinned memory with new version,
            unsafe { &mut *(setup.expect("setup") as &Setup as *const Setup as *mut Setup) }
        };

        let proof = match &mut self.context.manager {
            ManagerType::Forty(manager) => {
                let proof = create_proof::<_, _, T, _>(
                    assembly,
                    manager,
                    &self.context.worker,
                    pinned_setup,
                    transcript_params,
                )
                .map_err(|e| {
                    dbg!(e);
                    // we should free all slots in case of an error
                    manager.free_all_slots();
                    SynthesisError::Unsatisfiable
                })?;
                manager.free_all_slots();

                proof
            }
            ManagerType::Eighty(manager) => {
                let proof = create_proof::<_, _, T, _>(
                    assembly,
                    manager,
                    &self.context.worker,
                    pinned_setup,
                    transcript_params,
                )
                .map_err(|e| {
                    dbg!(e);
                    // we should free all slots in case of an error
                    manager.free_all_slots();
                    SynthesisError::Unsatisfiable
                })?;
                manager.free_all_slots();

                proof
            }
        };

        assert_eq!(n, proof.n);
        assert_eq!(num_inputs, proof.inputs.len());

        Ok(proof)
    }

    pub fn inner_create_vk_from_assembly<C: Circuit<Bn256>, S: SynthesisMode>(
        &mut self,
        assembly: &Assembly<S>,
    ) -> Result<VerificationKey<Bn256, C>, SynthesisError> {
        assert!(S::PRODUCE_SETUP);

        let n = assembly.n();
        let num_inputs = assembly.num_inputs;

        let vk = match &mut self.context.manager {
            ManagerType::Forty(manager) => {
                let vk = gpu_prover::compute_vk_from_assembly::<
                    C,
                    GpuProverConfigFor2x40,
                    PlonkCsWidth4WithNextStepAndCustomGatesParams,
                    _,
                >(manager, &assembly, &self.context.crs_for_verification)
                .map_err(|e| {
                    dbg!(e);
                    manager.free_all_slots();
                    SynthesisError::Unsatisfiable
                })?;
                manager.free_all_slots();

                vk
            }
            ManagerType::Eighty(manager) => {
                let vk = gpu_prover::compute_vk_from_assembly::<
                    C,
                    GpuProverConfigFor1x80,
                    PlonkCsWidth4WithNextStepAndCustomGatesParams,
                    _,
                >(manager, &assembly, &self.context.crs_for_verification)
                .map_err(|e| {
                    dbg!(e);
                    manager.free_all_slots();
                    SynthesisError::Unsatisfiable
                })?;
                manager.free_all_slots();

                vk
            }
        };

        assert_eq!(n, vk.n);
        assert_eq!(num_inputs, vk.num_inputs);

        Ok(vk)
    }

    pub fn inner_create_setup_from_assembly<C: Circuit<Bn256>, S: SynthesisMode>(
        &mut self,
        assembly: &Assembly<S>,
    ) -> Result<AsyncSetup, SynthesisError> {
        assert!(S::PRODUCE_SETUP);
        let mut setup = AsyncSetup::allocate_optimized(
            Self::get_max_domain_size(),
            NUM_LOOKUP_TABLE_NONZERO_VALUES,
        );

        match &mut self.context.manager {
            ManagerType::Forty(manager) => {
                setup
                    .generate_from_assembly(&self.context.worker, assembly, manager)
                    .unwrap();
                manager.free_all_slots();
            }
            ManagerType::Eighty(manager) => {
                setup
                    .generate_from_assembly(&self.context.worker, assembly, manager)
                    .unwrap();
                manager.free_all_slots();
            }
        }

        Ok(setup)
    }

    pub fn new_setup() -> AsyncSetup {
        let mut setup = AsyncSetup::allocate_optimized(
            Self::get_max_domain_size(),
            NUM_LOOKUP_TABLE_NONZERO_VALUES,
        );
        setup.zeroize();
        setup
    }
}

fn transform_crs(
    crs: &Crs<Bn256, CrsForMonomialForm>,
    worker: &OldWorker,
) -> Crs<CompactBn256, CrsForMonomialForm> {
    use bellman::CurveAffine;
    let bases_len = crs.g1_bases.len();
    let mut transformed_bases = vec![<CompactBn256 as Engine>::G1Affine::zero(); bases_len];
    println!("transforming CRS");
    worker.scope(bases_len, |scope, chunk_size| {
        for (src, dst) in crs
            .g1_bases
            .chunks(chunk_size)
            .zip(transformed_bases.chunks_mut(chunk_size))
        {
            scope.spawn(move |_| {
                assert_eq!(src.len(), dst.len());
                for (a, b) in src.iter().zip(dst.iter_mut()) {
                    let (x, y) = a.as_xy();
                    *b = <CompactBn256 as Engine>::G1Affine::from_xy_unchecked(*x, *y);
                }
            });
        }
    });

    let mut crs = Crs::<CompactBn256, CrsForMonomialForm>::dummy_crs(bases_len);
    crs.g1_bases = Arc::new(transformed_bases);

    crs
}
