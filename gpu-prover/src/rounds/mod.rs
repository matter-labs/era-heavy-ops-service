use super::*;

mod round1;
mod round15;
mod round2;
mod round3;
mod round4;
mod round5;
mod utils;

pub use round1::*;
pub use round15::*;
pub use round2::*;
pub use round3::*;
pub use round4::*;
pub use round5::*;

use bellman::plonk::commitments::transcript::Transcript;
use utils::*;

use super::*;

#[derive(Clone, Debug)]
pub struct ProverConstants<F: PrimeField> {
    pub omega: F,
    pub coset_omega: F,
    pub generator: F,
    pub non_residues: Vec<F>,

    pub eta: F,
    pub beta: F,
    pub gamma: F,
    pub beta_for_lookup: F,
    pub gamma_for_lookup: F,
    pub alpha: Vec<F>,
    pub z: F,
    pub v: Vec<F>,

    pub beta_plus_one_lookup: F,
    pub gamma_beta_lookup: F,
    pub expected: F,
}

impl<F: PrimeField> Default for ProverConstants<F> {
    fn default() -> Self {
        Self {
            omega: Default::default(),
            coset_omega: Default::default(),
            generator: Default::default(),
            non_residues: Default::default(),

            eta: Default::default(),
            beta: Default::default(),
            gamma: Default::default(),
            beta_for_lookup: Default::default(),
            gamma_for_lookup: Default::default(),
            alpha: Default::default(),
            z: Default::default(),
            v: Default::default(),

            beta_plus_one_lookup: Default::default(),
            gamma_beta_lookup: Default::default(),
            expected: Default::default(),
        }
    }
}
