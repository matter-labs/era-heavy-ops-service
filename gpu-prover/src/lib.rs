#![feature(get_mut_unchecked)]
#![feature(nonnull_slice_from_raw_parts)]
#![cfg_attr(feature = "allocator", feature(allocator_api))]
pub mod cuda_bindings;
// mod cuda_memory;
mod memory_manager;
mod proof;
mod rounds;
mod setup_precomputations;
mod utils;

// pub use cuda_bindings::*;
// pub use cuda_memory::*;
pub use franklin_crypto::bellman;
pub use memory_manager::*;
pub use proof::*;
pub use rounds::*;
pub use setup_precomputations::*;
pub use utils::*;

use bellman::{
    kate_commitment::{Crs, CrsForMonomialForm},
    plonk::better_better_cs::gates::main_gate_with_d_next::Width4MainGateWithDNext,
    plonk::better_better_cs::{
        cs::{Assembly, *},
        gates::selector_optimized_with_d_next::SelectorOptimizedWidth4MainGateWithDNext,
    },
    plonk::better_better_cs::{proof::Proof, setup::Setup},
    plonk::commitments::transcript::Transcript,
    worker::Worker,
    SynthesisError,
};
// pub use gpu_ffi::{GpuContext, GpuError, set_device};
use bellman::pairing::{
    // compact_bn256::{Bn256, Fr, FrRepr},
    bn256::{Bn256, Fr, FrRepr, G1Affine, G1},
    ff::{Field, PrimeField},
    CurveAffine,
    Engine,
};
type CompactG1Affine = bellman::compact_bn256::G1Affine;
use std::ffi::c_void;
use std::sync::Arc;

const FIELD_ELEMENT_LEN: usize = 32;
const LDE_FACTOR: usize = 4;
const STATE_WIDTH: usize = 4;
use cfg_if::*;

pub use crate::cuda_bindings::async_vec::AsyncVec;
cfg_if! {
    if #[cfg(feature = "allocator")]{
        use std::alloc::{Allocator, Global};
        use cuda_bindings::cuda_allocator::CudaAllocator;
        pub type DefaultAssembly<S> = Assembly<Bn256, PlonkCsWidth4WithNextStepAndCustomGatesParams, SelectorOptimizedWidth4MainGateWithDNext,S, cuda_bindings::CudaAllocator>;
        // pub type AsyncVec<T> = OriginalAsyncVec<T, cuda_bindings::CudaAllocator>;
    }else{
        pub type DefaultAssembly<S> = Assembly<Bn256, PlonkCsWidth4WithNextStepAndCustomGatesParams, SelectorOptimizedWidth4MainGateWithDNext,S>;
        // pub type AsyncVec<T> = OriginalAsyncVec<T>;
    }
}
