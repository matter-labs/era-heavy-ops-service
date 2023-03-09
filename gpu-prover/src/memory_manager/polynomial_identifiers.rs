use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotStatus {
    Free,
    Busy(PolyId, PolyForm),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolyForm {
    Monomial,
    Values,
    LDE(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolyId {
    Enumerated(usize),
    Custom(&'static str),
    Tmp,
    Tmp2,
    // Proof polynomials
    PI,
    A,
    B,
    C,
    D,
    DNext,
    F,
    T,
    S,
    TShifted,
    SShifted,
    SCol(usize),
    ZPermNum,
    ZPermDen,
    ZPerm,
    ZPermShifted,
    ZLookupNum,
    ZLookupDen,
    ZLookup,
    ZLookupShifted,
    TQuotient,
    TPart(usize),
    R,
    W,
    W1,
    // Setup polynomials
    X,
    QMab,
    QMac,
    QA,
    QB,
    QC,
    QD,
    QDNext,
    QConst,
    QMainSelector,
    QCustomSelector,
    Sigma(usize),
    QLookupSelector,
    QTableType,
    Col(usize),
    TableType,
    L0,
    Ln1,
}

pub trait ManagerConfigs {
    const NUM_GPUS_LOG: usize;
    const FULL_SLOT_SIZE_LOG: usize;
    const NUM_SLOTS: usize;
    const NUM_HOST_SLOTS: usize;

    const NUM_GPUS: usize = 1 << Self::NUM_GPUS_LOG;
    const FULL_SLOT_SIZE: usize = 1 << Self::FULL_SLOT_SIZE_LOG;
    const FULL_SLOT_BYTE_SIZE: usize = Self::FULL_SLOT_SIZE * FIELD_ELEMENT_LEN;
    const SLOT_SIZE_LOG: usize = Self::FULL_SLOT_SIZE_LOG - Self::NUM_GPUS_LOG;
    const SLOT_SIZE: usize = 1 << Self::SLOT_SIZE_LOG;
    const SLOT_BYTE_SIZE: usize = Self::SLOT_SIZE * FIELD_ELEMENT_LEN;
}

pub struct A100_80GB_Configs;

impl ManagerConfigs for A100_80GB_Configs {
    const NUM_GPUS_LOG: usize = 0;
    const FULL_SLOT_SIZE_LOG: usize = 26;
    const NUM_SLOTS: usize = 30;
    const NUM_HOST_SLOTS: usize = 2;
}

pub struct A100_40GB_2GPU_Configs;

impl ManagerConfigs for A100_40GB_2GPU_Configs {
    const NUM_GPUS_LOG: usize = 1;
    const FULL_SLOT_SIZE_LOG: usize = 26;
    const NUM_SLOTS: usize = 28;
    const NUM_HOST_SLOTS: usize = 2;
}

pub struct A100_40GB_2GPU_Test_Configs;

impl ManagerConfigs for A100_40GB_2GPU_Test_Configs {
    const NUM_GPUS_LOG: usize = 1;
    const FULL_SLOT_SIZE_LOG: usize = 20;
    const NUM_SLOTS: usize = 29;
    const NUM_HOST_SLOTS: usize = 2;
}

pub struct G5_5GB_Testing_Configs;

impl ManagerConfigs for G5_5GB_Testing_Configs {
    const NUM_GPUS_LOG: usize = 0;
    const FULL_SLOT_SIZE_LOG: usize = 18;
    const NUM_SLOTS: usize = 29;
    const NUM_HOST_SLOTS: usize = 2;
}
