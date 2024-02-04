#![feature(generic_const_exprs)]

mod prover_storage;
mod tests;

use prover_storage::*;

use circuit_definitions::circuit_definitions::{
    aux_layer::{
        wrapper::ZkSyncCompressionWrapper, ZkSyncCompressionForWrapperCircuit,
        ZkSyncCompressionLayerCircuit, ZkSyncCompressionLayerStorage, ZkSyncSnarkWrapperCircuit,
    },
    recursion_layer::{
        ZkSyncRecursionLayerStorage, ZkSyncRecursionLayerStorageType, ZkSyncRecursionProof,
        ZkSyncRecursionVerificationKey, ZkSyncRecursionLayerProof, ZkSyncRecursionLayerVerificationKey,
    },
};
use gpu_prover::{
    compute_vk_from_assembly, AsyncSetup, DefaultAssembly,
    DeviceMemoryManager, ManagerConfigs,
};
use std::sync::Arc;
use std::time::Instant;
use zkevm_test_harness::{
    boojum::cs::implementations::pow::NoPow,
    boojum::worker::Worker as BoojumWorker,
    data_source::{
        in_memory_data_source::InMemoryDataSource, local_file_data_source::LocalFileDataSource,
        BlockDataSource, SetupDataSource,
    },
    franklin_crypto::bellman::{
        kate_commitment::{Crs, CrsForMonomialForm},
        pairing::bn256::{Bn256, Fr, G2Affine},
        pairing::compact_bn256::Bn256 as CompactBn256,
        plonk::{
            better_better_cs::cs::{
                Circuit, PlonkCsWidth4WithNextStepAndCustomGatesParams,
                SynthesisModeGenerateSetup, SynthesisModeProve,
            },
            better_better_cs::proof::Proof as SnarkProof,
            better_better_cs::setup::VerificationKey as SnarkVK,
            better_better_cs::verifier::verify,
            commitments::transcript::keccak_transcript::RollingKeccakTranscript,
        },
        worker::Worker,
        Engine,
    },
    proof_wrapper_utils::{
        get_proof_for_previous_circuit, get_trusted_setup, get_vk_for_previous_circuit,
        WrapperConfig, L1_VERIFIER_DOMAIN_SIZE_LOG,
    },
    prover_utils::{
        create_compression_for_wrapper_setup_data, prove_compression_for_wrapper_circuit,
        verify_compression_for_wrapper_proof, create_compression_layer_setup_data,
        verify_compression_layer_proof, prove_compression_layer_circuit,
    },
};

pub struct GPUWrapperConfigs;

impl ManagerConfigs for GPUWrapperConfigs {
    const NUM_GPUS_LOG: usize = 0;
    const FULL_SLOT_SIZE_LOG: usize = 24;
    const NUM_SLOTS: usize = 29;
    const NUM_HOST_SLOTS: usize = 2;
}

pub struct WrapperProver<MC: ManagerConfigs> {
    manager: DeviceMemoryManager<Fr, MC>,
    worker: Worker,
    boojum_worker: BoojumWorker,

    setup_data: ProverSetupStorage,
    g2_bases: Arc<Vec<G2Affine>>,

    wrapper_config: WrapperConfig,
    setup_is_ready: bool,
}

impl<MC: ManagerConfigs> WrapperProver<MC> {
    pub fn new(crs: &Crs<Bn256, CrsForMonomialForm>, wrapper_config: WrapperConfig) -> Self {
        println!("Start allocating new prover");
        let start = Instant::now();

        assert_eq!(
            MC::FULL_SLOT_SIZE_LOG,
            L1_VERIFIER_DOMAIN_SIZE_LOG,
            "slot size of manager is not correct"
        );

        let worker = Worker::new();
        let boojum_worker = BoojumWorker::new();

        let compact_crs = transform_crs(&crs, &worker);

        let device_ids: Vec<_> = (0..<MC as ManagerConfigs>::NUM_GPUS).collect();

        let manager = DeviceMemoryManager::<Fr, MC>::init(
            &device_ids,
            compact_crs.g1_bases.as_ref().as_ref(),
        )
        .unwrap();

        let setup_data = ProverSetupStorage::new(MC::FULL_SLOT_SIZE);
        let g2_bases = crs.g2_monomial_bases.clone();

        println!("Prover is allocated, took {:?}", start.elapsed());

        Self {
            manager,
            worker,
            boojum_worker,
            setup_data,
            g2_bases,
            wrapper_config,
            setup_is_ready: false,
        }
    }

    pub fn source(&self) -> &InMemoryDataSource {
        &self.setup_data.source
    }

    pub fn get_wrapper_proof(&self) -> SnarkProof<Bn256, ZkSyncSnarkWrapperCircuit> {
        let wrapper_type = self.wrapper_config.get_wrapper_type();
        self.setup_data
            .source
            .get_wrapper_proof(wrapper_type)
            .expect("wrapper proof is missing")
            .into_inner()
    }

    pub fn get_wrapper_vk(&self) -> SnarkVK<Bn256, ZkSyncSnarkWrapperCircuit> {
        let wrapper_type = self.wrapper_config.get_wrapper_type();
        self.setup_data
            .source
            .get_wrapper_vk(wrapper_type)
            .expect("wrapper vk is missing")
            .into_inner()
    }

    pub fn generate_setup_data(&mut self, scheduler_vk: ZkSyncRecursionVerificationKey) {
        println!("Start generating setup data");
        let start = Instant::now();

        if self.setup_is_ready {
            self.setup_data = ProverSetupStorage::new(MC::FULL_SLOT_SIZE);
        }

        self.setup_data
            .source
            .set_recursion_layer_vk(ZkSyncRecursionLayerStorage::SchedulerCircuit(scheduler_vk))
            .unwrap();

        self.compute_compression_setup_data();
        self.compute_compression_for_wrapper_setup_data();
        self.compute_wrapper_setup_and_vk();

        println!("Setup data is generated, took {:?}", start.elapsed());
        self.setup_is_ready = true;
    }

    fn compute_compression_setup_data(&mut self) {
        self.setup_data.compression_data.clear();

        for circuit_type in self.wrapper_config.get_compression_types() {
            println!("Start generating setup data for compression #{}", circuit_type);
            let start = Instant::now();

            let vk = get_vk_for_previous_circuit(&self.setup_data.source, circuit_type).expect(&format!(
                "VK of previous circuit should be present. Current circuit type: {}",
                circuit_type
            ));

            let circuit =
                ZkSyncCompressionLayerCircuit::from_witness_and_vk(None, vk, circuit_type);
            
            let proof_config = circuit.proof_config_for_compression_step();

            let (
                setup_base, 
                setup, 
                vk, 
                setup_tree, 
                vars_hint, 
                wits_hint, 
                finalization_hint
            ) = create_compression_layer_setup_data(
                    circuit,
                    &self.boojum_worker,
                    proof_config.fri_lde_factor,
                    proof_config.merkle_tree_cap_size,
                );

            self.setup_data.source
                .set_compression_vk(ZkSyncCompressionLayerStorage::from_inner(
                    circuit_type,
                    vk.clone(),
                )).unwrap();
            self.setup_data.source
                .set_compression_hint(ZkSyncCompressionLayerStorage::from_inner(
                    circuit_type,
                    finalization_hint.clone(),
                )).unwrap();

            self.setup_data.compression_data.push(
                CompressionData {
                    setup_base,
                    setup,
                    setup_tree,
                    vk,
                    vars_hint,
                    wits_hint,
                    finalization_hint,
                }
            );

            println!("Setup data for compression #{} is generated, took {:?}", circuit_type, start.elapsed());
        }
    }

    fn compute_compression_for_wrapper_setup_data(&mut self) {
        let circuit_type = self.wrapper_config.get_compression_for_wrapper_type();

        println!("Start generating setup data for compression for wrapper #{}", circuit_type);
        let start = Instant::now();

        let vk = get_vk_for_previous_circuit(&mut self.setup_data.source, circuit_type).expect(
            &format!(
                "VK of previous circuit should be present. Current circuit type: {}",
                circuit_type
            ),
        );

        let circuit =
            ZkSyncCompressionForWrapperCircuit::from_witness_and_vk(None, vk, circuit_type);
        let proof_config = circuit.proof_config_for_compression_step();

        let (
            setup_base, 
            setup, 
            vk, 
            setup_tree, 
            vars_hint, 
            wits_hint, 
            finalization_hint
        ) = create_compression_for_wrapper_setup_data(
                circuit,
                &self.boojum_worker,
                proof_config.fri_lde_factor,
                proof_config.merkle_tree_cap_size,
            );

        self.setup_data.source
            .set_compression_for_wrapper_vk(ZkSyncCompressionLayerStorage::from_inner(
                circuit_type,
                vk.clone(),
            )).unwrap();
        self.setup_data.source
            .set_compression_for_wrapper_hint(ZkSyncCompressionLayerStorage::from_inner(
                circuit_type,
                finalization_hint.clone(),
            )).unwrap();

        self.setup_data.wrapper_compression_data = Some(
            CompressionData {
                setup_base,
                setup,
                setup_tree,
                vk,
                vars_hint,
                wits_hint,
                finalization_hint,
            }
        );

        println!("Setup data for compression for wrapper #{} is generated, took {:?}", circuit_type, start.elapsed());
    }

    fn compute_wrapper_setup_and_vk(&mut self) {
        let wrapper_type = self.wrapper_config.get_wrapper_type();

        println!("Start generating setup data for wrapper #{}", wrapper_type);
        let start = Instant::now();

        let vk = self
            .setup_data
            .source
            .get_compression_for_wrapper_vk(wrapper_type)
            .unwrap()
            .into_inner();

        let mut assembly = DefaultAssembly::<SynthesisModeGenerateSetup>::new();

        let wrapper_function = ZkSyncCompressionWrapper::from_numeric_circuit_type(wrapper_type);
        let fixed_parameters = vk.fixed_parameters.clone();

        let wrapper_circuit = ZkSyncSnarkWrapperCircuit {
            witness: None,
            vk: vk,
            fixed_parameters,
            transcript_params: (),
            wrapper_function,
        };

        wrapper_circuit.synthesize(&mut assembly).unwrap();
        assembly.finalize_to_size_log_2(L1_VERIFIER_DOMAIN_SIZE_LOG);

        self.setup_data
            .wrapper_setup
            .generate_from_assembly(&self.worker, &assembly, &mut self.manager)
            .unwrap();

        let mut dummy_crs = Crs::<_, CrsForMonomialForm>::dummy_crs(1);
        dummy_crs.g2_monomial_bases = self.g2_bases.clone();

        let snark_vk = compute_vk_from_assembly::<
            _,
            _,
            PlonkCsWidth4WithNextStepAndCustomGatesParams,
            _,
        >(&mut self.manager, &assembly, &dummy_crs)
        .unwrap();
        self.manager.free_all_slots();

        let snark_vk = ZkSyncCompressionLayerStorage::from_inner(wrapper_type, snark_vk);
        self.setup_data.source.set_wrapper_vk(snark_vk).unwrap();

        println!("Setup data for wrapper #{} is generated, took {:?}", wrapper_type, start.elapsed());
    }

    pub fn generate_proofs(&mut self, scheduler_proof: ZkSyncRecursionProof) {
        println!("Start generating proofs");
        let start = Instant::now();

        assert!(self.setup_is_ready, "Need to generate vks before proving");

        self.setup_data
            .source
            .set_scheduler_proof(ZkSyncRecursionLayerStorage::SchedulerCircuit(
                scheduler_proof,
            )).unwrap();

        self.generate_compression_proofs();
        self.generate_compression_for_wrapper_proof();
        self.generate_wrapper_proof();

        println!("Proofs are generated, took {:?}", start.elapsed());
    }

    fn generate_compression_proofs(&mut self) {
        for (i, circuit_type) in self.wrapper_config.get_compression_types().into_iter().enumerate() {
            println!("Start generating proof for compression #{}", circuit_type);
            let start = Instant::now();

            let proof = get_proof_for_previous_circuit(&mut self.setup_data.source, circuit_type)
                .expect(&format!(
                    "Proof of previous circuit should be present. Current circuit type: {}",
                    circuit_type
                ));

            let vk = get_vk_for_previous_circuit(&mut self.setup_data.source, circuit_type).expect(
                &format!(
                    "VK of previous circuit should be present. Current circuit type: {}",
                    circuit_type
                ),
            );

            let circuit =
                ZkSyncCompressionLayerCircuit::from_witness_and_vk(Some(proof), vk, circuit_type);

            let proof_config = circuit.proof_config_for_compression_step();

            let CompressionData {
                setup_base,
                setup,
                setup_tree,
                vk,
                vars_hint,
                wits_hint,
                finalization_hint,
            } = &self.setup_data.compression_data[i];
    
            let setup_circuit = circuit.clone_without_witness();
    
            let proof = prove_compression_layer_circuit::<NoPow>(
                circuit,
                &self.boojum_worker,
                proof_config,
                &setup_base,
                &setup,
                &setup_tree,
                &vk,
                &vars_hint,
                &wits_hint,
                &finalization_hint,
            );

            let is_valid = verify_compression_layer_proof::<NoPow>(&setup_circuit, &proof, vk);
            assert!(is_valid);

            self.setup_data
                .source
                .set_compression_proof(ZkSyncCompressionLayerStorage::from_inner(
                    circuit_type,
                    proof,
                ))
                .unwrap();

            println!("Proof for compression #{} is generated, took {:?}", circuit_type, start.elapsed());
        }
    }

    fn generate_compression_for_wrapper_proof(&mut self) {
        let circuit_type = self.wrapper_config.get_compression_for_wrapper_type();

        println!("Start generating proof for compression for wrapper #{}", circuit_type);
        let start = Instant::now();

        let proof = get_proof_for_previous_circuit(&mut self.setup_data.source, circuit_type)
            .expect(&format!(
                "Proof of previous circuit should be present. Current circuit type: {}",
                circuit_type
            ));

        let vk = get_vk_for_previous_circuit(&mut self.setup_data.source, circuit_type).expect(
            &format!(
                "VK of previous circuit should be present. Current circuit type: {}",
                circuit_type
            ),
        );

        let circuit =
            ZkSyncCompressionForWrapperCircuit::from_witness_and_vk(Some(proof), vk, circuit_type);

        let proof_config = circuit.proof_config_for_compression_step();

        let CompressionData {
            setup_base,
            setup,
            setup_tree,
            vk,
            vars_hint,
            wits_hint,
            finalization_hint,
        } = &self.setup_data.wrapper_compression_data.as_ref().unwrap();

        let setup_circuit = circuit.clone_without_witness();

        let proof = prove_compression_for_wrapper_circuit::<NoPow>(
            circuit,
            &self.boojum_worker,
            proof_config,
            &setup_base,
            &setup,
            &setup_tree,
            &vk,
            &vars_hint,
            &wits_hint,
            &finalization_hint,
        );

        let is_valid = verify_compression_for_wrapper_proof::<NoPow>(&setup_circuit, &proof, &vk);
        assert!(is_valid);

        self.setup_data
            .source
            .set_compression_for_wrapper_proof(ZkSyncCompressionLayerStorage::from_inner(
                circuit_type,
                proof,
            ))
            .unwrap();

        println!("Proof for compression for wrapper #{} is generated, took {:?}", circuit_type, start.elapsed());
    }

    fn generate_wrapper_proof(&mut self) {
        let wrapper_type = self.wrapper_config.get_wrapper_type();

        println!("Start generating proof for wrapper #{}", wrapper_type);
        let start = Instant::now();

        let proof = self
            .setup_data
            .source
            .get_compression_for_wrapper_proof(wrapper_type)
            .unwrap()
            .into_inner();
        let vk = self
            .setup_data
            .source
            .get_compression_for_wrapper_vk(wrapper_type)
            .unwrap()
            .into_inner();

        let mut assembly = DefaultAssembly::<SynthesisModeProve>::new();

        let wrapper_function = ZkSyncCompressionWrapper::from_numeric_circuit_type(wrapper_type);
        let fixed_parameters = vk.fixed_parameters.clone();

        let wrapper_circuit = ZkSyncSnarkWrapperCircuit {
            witness: Some(proof),
            vk: vk,
            fixed_parameters,
            transcript_params: (),
            wrapper_function,
        };

        wrapper_circuit.synthesize(&mut assembly).unwrap();
        assembly.finalize_to_size_log_2(L1_VERIFIER_DOMAIN_SIZE_LOG);
        assert!(assembly.is_satisfied());

        let snark_proof = gpu_prover::create_proof::<
            _,
            ZkSyncSnarkWrapperCircuit,
            RollingKeccakTranscript<Fr>,
            _,
        >(
            &assembly,
            &mut self.manager,
            &self.worker,
            &mut self.setup_data.wrapper_setup,
            None,
        )
        .unwrap();
        self.manager.free_all_slots();

        let snark_vk = self.setup_data.source.get_wrapper_vk(wrapper_type).unwrap();
        let is_valid =
            verify::<_, _, RollingKeccakTranscript<Fr>>(&snark_vk.into_inner(), &snark_proof, None)
                .unwrap();
        assert!(is_valid);

        let snark_proof = ZkSyncCompressionLayerStorage::from_inner(wrapper_type, snark_proof);
        self.setup_data
            .source
            .set_wrapper_proof(snark_proof)
            .unwrap();

        println!("Proof for wrapper #{} is generated, took {:?}", wrapper_type, start.elapsed());
    }
}

fn transform_crs(
    crs: &Crs<Bn256, CrsForMonomialForm>,
    worker: &Worker,
) -> Crs<CompactBn256, CrsForMonomialForm> {
    use gpu_prover::bellman::CurveAffine;
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
