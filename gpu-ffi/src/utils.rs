use std::ops::{Deref, DerefMut};

use futures::future::join_all;

use super::*;

pub fn log_2(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}


#[inline(always)]
pub fn bitreverse(n: usize, l: usize) -> usize {
    let mut r = n.reverse_bits();
    // now we need to only use the bits that originally were "last" l, so shift

    r >>= (std::mem::size_of::<usize>() * 8) - l;

    r
}


// pub fn decode_projective_point<E: Engine>(encoding: [Vec<u8>; 3]) -> E::G1 {
//     let [encoding_x, encoding_y, encoding_z] = encoding;
//     let mut repr = <<E::G1 as CurveProjective>::Base as PrimeField>::Repr::default();
//     repr.read_le(&encoding_x[..]).unwrap();
//     let x = <<E::G1 as CurveProjective>::Base as PrimeField>::from_raw_repr(repr).unwrap();
//     repr.read_le(&encoding_y[..]).unwrap();
//     let y = <<E::G1 as CurveProjective>::Base as PrimeField>::from_raw_repr(repr).unwrap();
//     repr.read_le(&encoding_z[..]).unwrap();
//     let z = <<E::G1 as CurveProjective>::Base as PrimeField>::from_raw_repr(repr).unwrap();

//     E::G1::from_xyz_unchecked(x, y, z)
// }

// pub async fn encode_bases<E: Engine>(
//     worker: Worker,
//     mut bases: Vec<E::G1Affine>,
// ) -> [Vec<u8>; 2] {
//     let degree = bases.len();

//     let final_encoding_x = SubVec::new(vec![0u8; degree * 32]);
//     let mut encoding_x = final_encoding_x.clone();
//     let final_encoding_y = SubVec::new(vec![0u8; degree * 32]);
//     let mut encoding_y = final_encoding_y.clone();

//     let mut num_cpus = worker.max_available_resources().cpu_cores;
//     if degree <= num_cpus {
//         num_cpus = 1
//     }

//     let chunk_size = degree / num_cpus;
//     let num_chunks = degree / chunk_size;
//     let mut handles = vec![];

//     for _ in 0..num_chunks {
//         let (current_bases, rest) = bases.split_at(chunk_size);
//         bases = rest;
        
//         let (mut current_encoding_x, rest) = encoding_x.split_at(chunk_size * 32);
//         encoding_x = rest;
//         let (mut current_encoding_y, rest) = encoding_y.split_at(chunk_size * 32);
//         encoding_y = rest;

//         let fut = async move {
//             for (base, (x_buf, y_buf)) in current_bases.deref().iter().zip(
//                 current_encoding_x
//                     .deref_mut()
//                     .chunks_exact_mut(32)
//                     .zip(current_encoding_y.deref_mut().chunks_exact_mut(32)),
//             ) {
//                 let (x, y) = base.as_xy();
//                 x.into_raw_repr().write_le(&mut x_buf[..]).unwrap();
//                 y.into_raw_repr().write_le(&mut y_buf[..]).unwrap();
//             }
//         };
//         let handle = worker.spawn_with_handle(fut).unwrap();
//         handles.push(handle);

//         if bases.is_empty() {
//             break;
//         }
//     }

//     let _ = join_all(handles).await;

//     [final_encoding_x, final_encoding_y]
// }

// pub fn decode_scalars<E: Engine>(encoding: &[u8]) -> Vec<E::Fr> {    
//     let len = encoding.len() / 32;
//     let mut result = vec![E::Fr::zero(); len];    
//     unsafe{std::ptr::copy(encoding.as_ptr() as *const E::Fr, result.as_mut_ptr(), len)};
//     result
// }

// pub fn decode_affine_points<E: Engine>(encoding_x: &[u8], encoding_y: &[u8]) -> Vec<E::G1Affine> {
//     let mut result = vec![];
//     let bytes_per_el = <<E::G1Affine as CurveAffine>::Base as PrimeField>::Repr::default()
//         .as_ref()
//         .len()
//         * 64
//         / 8;
//     assert_eq!(bytes_per_el, 32);
//     for (chunk_x, chunk_y) in encoding_x
//         .chunks_exact(bytes_per_el)
//         .zip(encoding_y.chunks_exact(bytes_per_el))
//     {
//         let mut repr = <<E::G1Affine as CurveAffine>::Base as PrimeField>::Repr::default();
//         repr.read_le(chunk_x).unwrap();
//         let el_x = <E::G1Affine as CurveAffine>::Base::from_raw_repr(repr).unwrap();
//         let mut repr = <<E::G1Affine as CurveAffine>::Base as PrimeField>::Repr::default();
//         repr.read_le(chunk_y).unwrap();
//         let el_y = <E::G1Affine as CurveAffine>::Base::from_raw_repr(repr).unwrap();
//         let p = E::G1Affine::from_xy_checked(el_x, el_y).expect("valid point");
//         result.push(p);
//     }

//     result
// }
