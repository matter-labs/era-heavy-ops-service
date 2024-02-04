use super::*;

use zkevm_test_harness::boojum::{
    algebraic_props::round_function::AbsorptionModeOverwrite,
    algebraic_props::sponge::GoldilocksPoseidon2Sponge,
    cs::implementations::{
        hints::DenseVariablesCopyHint, hints::DenseWitnessCopyHint,
        polynomial_storage::SetupBaseStorage, polynomial_storage::SetupStorage,
        setup::FinalizationHintsForProver, verifier::VerificationKey,
    },
    cs::oracle::merkle_tree::MerkleTreeWithCap,
    cs::oracle::TreeHasher,
    field::{goldilocks::GoldilocksField, traits::field::PrimeField, SmallField},
};

pub struct ProverSetupStorage {
    pub(crate) source: InMemoryDataSource,

    pub(crate) compression_data: Vec<CommonCompressionData>,
    pub(crate) wrapper_compression_data: Option<WrapperCompressionData>,
    pub(crate) wrapper_setup: AsyncSetup,
}

impl ProverSetupStorage {
    pub(crate) fn new(slot_size: usize) -> Self {
        let wrapper_setup = AsyncSetup::allocate(slot_size);

        Self {
            source: InMemoryDataSource::new(),
            compression_data: vec![],
            wrapper_compression_data: None,
            wrapper_setup,
        }
    }
}

pub(crate) struct CompressionData<F: PrimeField + SmallField, H: TreeHasher<F>> {
    pub(crate) setup_base: SetupBaseStorage<F>,
    pub(crate) setup: SetupStorage<F>,
    pub(crate) setup_tree: MerkleTreeWithCap<F, H>,
    pub(crate) vk: VerificationKey<F, H>,
    pub(crate) vars_hint: DenseVariablesCopyHint,
    pub(crate) wits_hint: DenseWitnessCopyHint,
    pub(crate) finalization_hint: FinalizationHintsForProver,
}

type CommonCompressionData =
    CompressionData<GoldilocksField, GoldilocksPoseidon2Sponge<AbsorptionModeOverwrite>>;
type WrapperCompressionData =
    CompressionData<GoldilocksField, zkevm_test_harness::prover_utils::TreeHasherForWrapper>;
