use bellman::plonk::{better_better_cs::cs::SynthesisMode, commitments::transcript::Prng};

use super::*;
use std::{iter::empty, path::Path};

pub struct ProverContext {
    worker: OldWorker,
    crs: Crs<Bn256, CrsForMonomialForm>,
}

impl ProverContext {
    pub fn init() -> Self {
        let crs_file_str = std::env::var(CRS_FILE_ENV_VAR).expect("crs file env var");
        let crs_file_path = std::path::Path::new(&crs_file_str);

        let crs = Self::read_crs(crs_file_path);

        Self {
            worker: OldWorker::new(),
            crs,
        }
    }
    pub fn init_with_crs(crs_file_path: &Path) -> Self {
        let crs = Self::read_crs(crs_file_path);

        Self {
            worker: OldWorker::new(),
            crs,
        }
    }

    pub fn init_with_dummy_crs() -> Self {
        let domain_size = 1 << 26;
        let worker = OldWorker::new();
        let crs = Crs::<Bn256, CrsForMonomialForm>::dummy_crs(domain_size);
        Self { worker, crs }
    }

    fn read_crs(crs_file_path: &Path) -> Crs<Bn256, CrsForMonomialForm> {
        let crs_file = std::fs::File::open(&crs_file_path).expect("crs file to open");

        Crs::read(&crs_file).expect("crs file for bases")
    }
}

impl Prover {
    pub fn config() -> Vec<u64> {
        unimplemented!();
    }
    pub fn inner_create_proof<C: Circuit<Bn256>, T: Transcript<Fr>, S: SynthesisMode>(
        &mut self,
        assembly: &Assembly<S>,
        setup: Option<&Setup<C>>,
        transcript_params: Option<T::InitializationParameters>,
    ) -> Result<Proof<Bn256, C>, SynthesisError> {
        assert!(S::PRODUCE_WITNESS);
        dbg!("CPU PROVER");
        assert!(assembly.is_finalized);

        let empty_setup = Setup::empty();
        // If Assembly contains setup values then we don't need produce setup
        let setup = if !S::PRODUCE_SETUP {
            setup.expect("precomputed setup")
        } else {
            &empty_setup
        };

        assert!(
            setup.n <= self.context.crs.g1_bases.len(),
            "circuit polynomial degree is greater than number of bases"
        );

        let proof = assembly.create_proof_by_ref::<_, T>(
            &self.context.worker,
            &setup,
            &self.context.crs,
            transcript_params,
        )?;
        Ok(proof)
    }

    pub fn inner_create_proof_for_proving_assembly<C: Circuit<Bn256>, T: Transcript<Fr>>(
        &mut self,
        assembly: &ProvingAssembly,
        setup: &Setup<C>,
        transcript_params: Option<T::InitializationParameters>,
    ) -> Result<Proof<Bn256, C>, SynthesisError> {
        dbg!("CPU PROVER");
        assert!(assembly.is_finalized);

        assert!(
            setup.n <= self.context.crs.g1_bases.len(),
            "circuit polynomial degree is greater than number of bases"
        );

        let proof = assembly.create_proof_by_ref::<_, T>(
            &self.context.worker,
            &setup,
            &self.context.crs,
            transcript_params,
        )?;

        Ok(proof)
    }

    pub fn inner_create_vk_from_assembly<C: Circuit<Bn256>, S: SynthesisMode>(
        &mut self,
        assembly: &Assembly<S>,
    ) -> Result<VerificationKey<Bn256, C>, SynthesisError> {
        assert!(S::PRODUCE_SETUP);
        let setup = assembly.create_setup(&self.context.worker)?;

        let vk = VerificationKey::from_setup(&setup, &self.context.worker, &self.context.crs)?;

        Ok(vk)
    }

    pub fn inner_create_setup_from_assembly<C: Circuit<Bn256>, S: SynthesisMode>(
        &mut self,
        assembly: &Assembly<S>,
    ) -> Result<Setup<C>, SynthesisError> {
        assert!(S::PRODUCE_SETUP);

        let setup = assembly.create_setup(&self.context.worker)?;
        Ok(setup)
    }

    pub fn new_setup<C: Circuit<Bn256>>() -> Setup<C> {
        Setup::empty()
    }
}
