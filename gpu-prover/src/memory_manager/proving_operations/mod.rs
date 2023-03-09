mod columns_sorting;
mod compute_assigments_and_permutations;
mod compute_permutations;
mod compute_state_values;
mod create_selectors;

pub use columns_sorting::*;
pub use compute_assigments_and_permutations::*;
pub use compute_permutations::*;
pub use compute_state_values::*;
pub use create_selectors::*;

use super::*;
use crate::cuda_bindings::{set_device, Event, GpuError};

pub fn transform_variable(var: &Variable, num_inputs: usize) -> u32 {
    let index = match var.get_unchecked() {
        Index::Aux(0) => {
            // Dummy variables do not participate in the permutation actually
            // but we still keep them
            0
        }
        Index::Input(0) => {
            unreachable!("There must be no input with index 0");
        }
        Index::Input(index) => index,
        Index::Aux(index) => index + num_inputs,
    };
    index as u32
}
