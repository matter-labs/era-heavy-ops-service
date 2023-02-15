use super::*;

#[derive(Debug)]
pub struct SetupWrapper {
    inner: Setup,
    is_busy: bool,
}

#[derive(Debug)]
#[repr(u8)]
pub enum ZkSyncSetup {
    Scheduler(SetupWrapper),
    NodeAggregation(SetupWrapper),
    LeafAggregation(SetupWrapper),
    MainVM(SetupWrapper),
    CodeDecommittmentsSorter(SetupWrapper),
    CodeDecommitter(SetupWrapper),
    LogDemuxer(SetupWrapper),
    KeccakRoundFunction(SetupWrapper),
    Sha256RoundFunction(SetupWrapper),
    ECRecover(SetupWrapper),
    RAMPermutation(SetupWrapper),
    StorageSorter(SetupWrapper),
    StorageApplication(SetupWrapper),
    InitialWritesPubdataHasher(SetupWrapper),
    RepeatedWritesPubdataHasher(SetupWrapper),
    EventsSorter(SetupWrapper),
    L1MessagesSorter(SetupWrapper),
    L1MessagesHasher(SetupWrapper),
    L1MessagesMerklier(SetupWrapper),
    EventsMerkelization(SetupWrapper),
    None(SetupWrapper),
}

impl ZkSyncSetup {
    pub fn empty(setup: Setup) -> Self {
        Self::None(SetupWrapper {
            inner: setup,
            is_busy: false,
        })
    }

    pub fn set_circuit_type(&mut self, circuit_id: u8) {
        let self_ptr = self as *mut ZkSyncSetup;
        let self_ptr = self_ptr as *mut u8;
        unsafe {
            // SAFETY: this is safe as long as order of variants in ZkSyncSetup and ZkSyncCircuit are the same
            std::ptr::write(self_ptr, circuit_id);
        }
    }

    pub fn is_busy(&self) -> bool {
        self.inner().is_busy
    }

    pub fn is_free(&self) -> bool {
        !self.is_busy()
    }

    pub fn free(&mut self) {
        self.inner_mut().is_busy = false;
    }

    #[cfg(not(feature = "legacy"))]
    pub fn reload(&mut self, encoding: Box<dyn Read>, circuit_id: u8) {
        assert!(self.is_free());
        let inner = self.as_setup_mut();
        inner.zeroize(); // clear contents of previous setup
        inner.read(encoding).unwrap();
        self.set_circuit_type(circuit_id);
        assert_eq!(self.numeric_circuit_type(), circuit_id);
        self.inner_mut().is_busy = true;
    }

    #[cfg(feature = "legacy")]
    pub fn reload(&mut self, encoding: Box<dyn Read>, circuit_id: u8) {
        assert!(self.is_free());
        self.inner_mut().inner = Setup::read(encoding).unwrap();
        self.set_circuit_type(circuit_id);
        self.inner_mut().is_busy = true;
    }

    pub fn numeric_circuit_type(&self) -> u8 {
        use zkevm_test_harness::sync_vm::scheduler::CircuitType;

        match &self {
            ZkSyncSetup::Scheduler(..) => CircuitType::Scheduler as u8,
            ZkSyncSetup::NodeAggregation(..) => CircuitType::IntermidiateNode as u8,
            ZkSyncSetup::LeafAggregation(..) => CircuitType::Leaf as u8,
            ZkSyncSetup::MainVM(..) => CircuitType::VM as u8,
            ZkSyncSetup::CodeDecommittmentsSorter(..) => CircuitType::DecommitmentsFilter as u8,
            ZkSyncSetup::CodeDecommitter(..) => CircuitType::Decommiter as u8,
            ZkSyncSetup::LogDemuxer(..) => CircuitType::LogDemultiplexer as u8,
            ZkSyncSetup::KeccakRoundFunction(..) => CircuitType::KeccakPrecompile as u8,
            ZkSyncSetup::Sha256RoundFunction(..) => CircuitType::Sha256Precompile as u8,
            ZkSyncSetup::ECRecover(..) => CircuitType::EcrecoverPrecompile as u8,
            ZkSyncSetup::RAMPermutation(..) => CircuitType::RamValidation as u8,
            ZkSyncSetup::StorageSorter(..) => CircuitType::StorageFilter as u8,
            ZkSyncSetup::StorageApplication(..) => CircuitType::StorageApplicator as u8,
            ZkSyncSetup::EventsSorter(..) => CircuitType::EventsRevertsFilter as u8,
            ZkSyncSetup::L1MessagesSorter(..) => CircuitType::L1MessagesRevertsFilter as u8,
            ZkSyncSetup::L1MessagesHasher(..) => CircuitType::L1MessagesHasher as u8,
            ZkSyncSetup::L1MessagesMerklier(..) => CircuitType::L1MessagesMerkelization as u8,
            ZkSyncSetup::InitialWritesPubdataHasher(..) => {
                CircuitType::StorageFreshWritesHasher as u8
            }
            ZkSyncSetup::RepeatedWritesPubdataHasher(..) => {
                CircuitType::StorageRepeatedWritesHasher as u8
            }
            ZkSyncSetup::EventsMerkelization(..) => CircuitType::EventsMerkelization as u8,
            ZkSyncSetup::L1MessagesHasher(..) => CircuitType::L1MessagesHasher as u8,
            ZkSyncSetup::None(..) => CircuitType::None as u8,
        }
    }

    pub fn short_description(&self) -> &'static str {
        match &self {
            ZkSyncSetup::None(..) => "None",
            ZkSyncSetup::Scheduler(..) => "Scheduler",
            ZkSyncSetup::LeafAggregation(..) => "Leaf aggregation",
            ZkSyncSetup::NodeAggregation(..) => "Node aggregation",
            ZkSyncSetup::MainVM(..) => "Main VM",
            ZkSyncSetup::CodeDecommittmentsSorter(..) => "Decommitts sorter",
            ZkSyncSetup::CodeDecommitter(..) => "Code decommitter",
            ZkSyncSetup::LogDemuxer(..) => "Log demuxer",
            ZkSyncSetup::KeccakRoundFunction(..) => "Keccak",
            ZkSyncSetup::Sha256RoundFunction(..) => "SHA256",
            ZkSyncSetup::ECRecover(..) => "ECRecover",
            ZkSyncSetup::RAMPermutation(..) => "RAM permutation",
            ZkSyncSetup::StorageSorter(..) => "Storage sorter",
            ZkSyncSetup::StorageApplication(..) => "Storage application",
            ZkSyncSetup::EventsSorter(..) => "Events sorter",
            ZkSyncSetup::L1MessagesSorter(..) => "L1 messages sorter",
            ZkSyncSetup::L1MessagesHasher(..) => "L1 messages hasher",
            ZkSyncSetup::L1MessagesMerklier(..) => "L1 messages merklizer",
            ZkSyncSetup::InitialWritesPubdataHasher(..) => "Initial writes pubdata rehasher",
            ZkSyncSetup::RepeatedWritesPubdataHasher(..) => "Repeated writes pubdata rehasher",
            ZkSyncSetup::EventsMerkelization(..) => "Events merklizer",
            ZkSyncSetup::L1MessagesHasher(..) => "L1 messages hasher",
        }
    }

    pub fn from_setup_and_numeric_type(numeric_type: u8, setup: Setup) -> Self {
        use zkevm_test_harness::sync_vm::scheduler::CircuitType;
        let setup = SetupWrapper {
            inner: setup,
            is_busy: false,
        };
        match numeric_type {
            a if a == CircuitType::Scheduler as u8 => ZkSyncSetup::Scheduler(setup),
            a if a == CircuitType::IntermidiateNode as u8 => ZkSyncSetup::NodeAggregation(setup),
            a if a == CircuitType::Leaf as u8 => ZkSyncSetup::LeafAggregation(setup),
            a if a == CircuitType::VM as u8 => ZkSyncSetup::MainVM(setup),
            a if a == CircuitType::DecommitmentsFilter as u8 => {
                ZkSyncSetup::CodeDecommittmentsSorter(setup)
            }
            a if a == CircuitType::Decommiter as u8 => ZkSyncSetup::CodeDecommitter(setup),
            a if a == CircuitType::LogDemultiplexer as u8 => ZkSyncSetup::LogDemuxer(setup),
            a if a == CircuitType::KeccakPrecompile as u8 => {
                ZkSyncSetup::KeccakRoundFunction(setup)
            }
            a if a == CircuitType::Sha256Precompile as u8 => {
                ZkSyncSetup::Sha256RoundFunction(setup)
            }
            a if a == CircuitType::EcrecoverPrecompile as u8 => ZkSyncSetup::ECRecover(setup),
            a if a == CircuitType::RamValidation as u8 => ZkSyncSetup::RAMPermutation(setup),
            a if a == CircuitType::StorageFilter as u8 => ZkSyncSetup::StorageSorter(setup),
            a if a == CircuitType::StorageApplicator as u8 => {
                ZkSyncSetup::StorageApplication(setup)
            }
            a if a == CircuitType::EventsRevertsFilter as u8 => ZkSyncSetup::EventsSorter(setup),
            a if a == CircuitType::L1MessagesRevertsFilter as u8 => {
                ZkSyncSetup::L1MessagesSorter(setup)
            }
            a if a == CircuitType::L1MessagesHasher as u8 => {
                ZkSyncSetup::L1MessagesHasher(setup)
            }
            a if a == CircuitType::L1MessagesMerkelization as u8 => {
                ZkSyncSetup::L1MessagesMerklier(setup)
            }
            a if a == CircuitType::StorageFreshWritesHasher as u8 => {
                ZkSyncSetup::InitialWritesPubdataHasher(setup)
            }
            a if a == CircuitType::StorageRepeatedWritesHasher as u8 => {
                ZkSyncSetup::RepeatedWritesPubdataHasher(setup)
            }
            a if a == CircuitType::EventsMerkelization as u8 => {
                ZkSyncSetup::EventsMerkelization(setup)
            }
            a if a == CircuitType::L1MessagesHasher as u8 => {
                ZkSyncSetup::L1MessagesHasher(setup)
            }
            a @ _ => panic!("unknown numeric type {}", a),
        }
    }

    pub fn into_setup(self) -> Setup {
        let inner = match self {
            ZkSyncSetup::None(inner) => inner,
            ZkSyncSetup::Scheduler(inner) => inner,
            ZkSyncSetup::LeafAggregation(inner) => inner,
            ZkSyncSetup::NodeAggregation(inner) => inner,
            ZkSyncSetup::MainVM(inner) => inner,
            ZkSyncSetup::CodeDecommittmentsSorter(inner) => inner,
            ZkSyncSetup::CodeDecommitter(inner) => inner,
            ZkSyncSetup::LogDemuxer(inner) => inner,
            ZkSyncSetup::KeccakRoundFunction(inner) => inner,
            ZkSyncSetup::Sha256RoundFunction(inner) => inner,
            ZkSyncSetup::ECRecover(inner) => inner,
            ZkSyncSetup::RAMPermutation(inner) => inner,
            ZkSyncSetup::StorageSorter(inner) => inner,
            ZkSyncSetup::StorageApplication(inner) => inner,
            ZkSyncSetup::EventsSorter(inner) => inner,
            ZkSyncSetup::L1MessagesSorter(inner) => inner,
            ZkSyncSetup::L1MessagesHasher(inner) => inner,
            ZkSyncSetup::L1MessagesMerklier(inner) => inner,
            ZkSyncSetup::InitialWritesPubdataHasher(inner) => inner,
            ZkSyncSetup::RepeatedWritesPubdataHasher(inner) => inner,
            ZkSyncSetup::EventsMerkelization(inner) => inner,
            ZkSyncSetup::L1MessagesHasher(inner) => {inner}
        };
        inner.inner
    }

    pub fn as_setup(&self) -> &Setup {
        let inner = match self {
            ZkSyncSetup::None(inner) => inner,
            ZkSyncSetup::Scheduler(inner) => inner,
            ZkSyncSetup::LeafAggregation(inner) => inner,
            ZkSyncSetup::NodeAggregation(inner) => inner,
            ZkSyncSetup::MainVM(inner) => inner,
            ZkSyncSetup::CodeDecommittmentsSorter(inner) => inner,
            ZkSyncSetup::CodeDecommitter(inner) => inner,
            ZkSyncSetup::LogDemuxer(inner) => inner,
            ZkSyncSetup::KeccakRoundFunction(inner) => inner,
            ZkSyncSetup::Sha256RoundFunction(inner) => inner,
            ZkSyncSetup::ECRecover(inner) => inner,
            ZkSyncSetup::RAMPermutation(inner) => inner,
            ZkSyncSetup::StorageSorter(inner) => inner,
            ZkSyncSetup::StorageApplication(inner) => inner,
            ZkSyncSetup::EventsSorter(inner) => inner,
            ZkSyncSetup::L1MessagesSorter(inner) => inner,
            ZkSyncSetup::L1MessagesHasher(inner) => inner,
            ZkSyncSetup::L1MessagesMerklier(inner) => inner,
            ZkSyncSetup::InitialWritesPubdataHasher(inner) => inner,
            ZkSyncSetup::RepeatedWritesPubdataHasher(inner) => inner,
            ZkSyncSetup::EventsMerkelization(inner) => inner,
            ZkSyncSetup::L1MessagesHasher(inner) => {inner}
        };
        &inner.inner
    }
    pub fn inner(&self) -> &SetupWrapper {
        let inner = match self {
            ZkSyncSetup::None(inner) => inner,
            ZkSyncSetup::Scheduler(inner) => inner,
            ZkSyncSetup::LeafAggregation(inner) => inner,
            ZkSyncSetup::NodeAggregation(inner) => inner,
            ZkSyncSetup::MainVM(inner) => inner,
            ZkSyncSetup::CodeDecommittmentsSorter(inner) => inner,
            ZkSyncSetup::CodeDecommitter(inner) => inner,
            ZkSyncSetup::LogDemuxer(inner) => inner,
            ZkSyncSetup::KeccakRoundFunction(inner) => inner,
            ZkSyncSetup::Sha256RoundFunction(inner) => inner,
            ZkSyncSetup::ECRecover(inner) => inner,
            ZkSyncSetup::RAMPermutation(inner) => inner,
            ZkSyncSetup::StorageSorter(inner) => inner,
            ZkSyncSetup::StorageApplication(inner) => inner,
            ZkSyncSetup::EventsSorter(inner) => inner,
            ZkSyncSetup::L1MessagesSorter(inner) => inner,
            ZkSyncSetup::L1MessagesHasher(inner) => inner,
            ZkSyncSetup::L1MessagesMerklier(inner) => inner,
            ZkSyncSetup::InitialWritesPubdataHasher(inner) => inner,
            ZkSyncSetup::RepeatedWritesPubdataHasher(inner) => inner,
            ZkSyncSetup::EventsMerkelization(inner) => inner,
            ZkSyncSetup::L1MessagesHasher(inner) => {inner}
        };
        inner
    }
    pub fn inner_mut(&mut self) -> &mut SetupWrapper {
        let inner = match self {
            ZkSyncSetup::None(inner) => inner,
            ZkSyncSetup::Scheduler(inner) => inner,
            ZkSyncSetup::LeafAggregation(inner) => inner,
            ZkSyncSetup::NodeAggregation(inner) => inner,
            ZkSyncSetup::MainVM(inner) => inner,
            ZkSyncSetup::CodeDecommittmentsSorter(inner) => inner,
            ZkSyncSetup::CodeDecommitter(inner) => inner,
            ZkSyncSetup::LogDemuxer(inner) => inner,
            ZkSyncSetup::KeccakRoundFunction(inner) => inner,
            ZkSyncSetup::Sha256RoundFunction(inner) => inner,
            ZkSyncSetup::ECRecover(inner) => inner,
            ZkSyncSetup::RAMPermutation(inner) => inner,
            ZkSyncSetup::StorageSorter(inner) => inner,
            ZkSyncSetup::StorageApplication(inner) => inner,
            ZkSyncSetup::EventsSorter(inner) => inner,
            ZkSyncSetup::L1MessagesSorter(inner) => inner,
            ZkSyncSetup::L1MessagesHasher(inner) => inner,
            ZkSyncSetup::L1MessagesMerklier(inner) => inner,
            ZkSyncSetup::InitialWritesPubdataHasher(inner) => inner,
            ZkSyncSetup::RepeatedWritesPubdataHasher(inner) => inner,
            ZkSyncSetup::EventsMerkelization(inner) => inner,
            ZkSyncSetup::L1MessagesHasher(inner) => {inner}
        };
        inner
    }

    pub fn as_setup_mut(&mut self) -> &mut Setup {
        let inner = match self {
            ZkSyncSetup::None(inner) => inner,
            ZkSyncSetup::Scheduler(inner) => inner,
            ZkSyncSetup::LeafAggregation(inner) => inner,
            ZkSyncSetup::NodeAggregation(inner) => inner,
            ZkSyncSetup::MainVM(inner) => inner,
            ZkSyncSetup::CodeDecommittmentsSorter(inner) => inner,
            ZkSyncSetup::CodeDecommitter(inner) => inner,
            ZkSyncSetup::LogDemuxer(inner) => inner,
            ZkSyncSetup::KeccakRoundFunction(inner) => inner,
            ZkSyncSetup::Sha256RoundFunction(inner) => inner,
            ZkSyncSetup::ECRecover(inner) => inner,
            ZkSyncSetup::RAMPermutation(inner) => inner,
            ZkSyncSetup::StorageSorter(inner) => inner,
            ZkSyncSetup::StorageApplication(inner) => inner,
            ZkSyncSetup::EventsSorter(inner) => inner,
            ZkSyncSetup::L1MessagesSorter(inner) => inner,
            ZkSyncSetup::L1MessagesHasher(inner) => inner,
            ZkSyncSetup::L1MessagesMerklier(inner) => inner,
            ZkSyncSetup::InitialWritesPubdataHasher(inner) => inner,
            ZkSyncSetup::RepeatedWritesPubdataHasher(inner) => inner,
            ZkSyncSetup::EventsMerkelization(inner) => inner,
            ZkSyncSetup::L1MessagesHasher(inner) => {inner}
        };
        &mut inner.inner
    }
}
