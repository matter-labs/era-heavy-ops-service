use bellman::plonk::commitments::transcript::Transcript;

use super::*;

pub fn commit_point_as_xy<E: Engine, T: Transcript<E::Fr>>(
    transcript: &mut T,
    point: &E::G1Affine,
) {
    use bellman::pairing::ff::Field;
    use bellman::pairing::CurveAffine;

    // if point.is_zero() {
    if bellman::pairing::CurveAffine::is_zero(point) {
        transcript.commit_fe(&E::Fq::zero());
        transcript.commit_fe(&E::Fq::zero());
    } else {
        let (x, y) = bellman::pairing::CurveAffine::into_xy_unchecked(point.clone()); // TODO
        transcript.commit_fe(&x);
        transcript.commit_fe(&y);
    }
}

use bellman::plonk::better_better_cs::lookup_tables::KeyValueSet;
use itertools::*;
use std::time::Instant;
