use super::*;
use bellman::pairing::{CurveAffine, GenericCurveProjective};
use rand::{thread_rng, Rng};
use core::ops::Range;

pub(crate) fn generate_scalars_to_buf<F: PrimeField>(worker: &Worker, buf: &mut [F]) {
    assert!(buf.len().is_power_of_two());
    worker.scope(buf.len(), |scope, chunk_size| {
        for chunk in buf.chunks_mut(chunk_size) {
            scope.spawn(|_| {
                let rng = &mut thread_rng();
                for el in chunk.iter_mut() {
                    *el = rng.gen();
                }
            });
        }
    });
}

pub(crate) fn generate_bases<E: Engine>(worker: &Worker, degree: usize) -> Vec<E::G1Affine> {
    assert!(degree.is_power_of_two());
    let mut bases = vec![E::G1Affine::zero(); degree];
    worker.scope(bases.len(), |scope, chunk_size| {
        for chunk in bases.chunks_mut(chunk_size) {
            scope.spawn(|_| {
                let rng = &mut thread_rng();
                for el in chunk.iter_mut() {
                    *el = rng.gen::<E::G1>().into_affine();
                }
            });
        }
    });

    bases
}

pub(crate) fn generate_dummy_bases<E: Engine>(worker: &Worker, degree: usize) -> Vec<E::G1Affine> {
    assert!(degree.is_power_of_two());
    let mut bases = vec![E::G1Affine::zero(); degree];
    worker.scope(bases.len(), |scope, chunk_size| {
        for chunk in bases.chunks_mut(chunk_size) {
            scope.spawn(|_| {
                for el in chunk.iter_mut() {
                    *el = E::G1Affine::one();
                }
            });
        }
    });

    bases
}

pub fn transmute_values<'a, U, V, const R: bool>(values: &'a [U]) -> &'a [V] {
    let ptr = values.as_ptr();
    let len = values.len();

    assert!(
        (ptr as usize) % std::mem::align_of::<V>() == 0,
        "trying to cast with mismatched layout"
    );

    let size = if R {
        assert!(len % std::mem::size_of::<V>() == 0);
        len / std::mem::size_of::<V>()
    } else {
        std::mem::size_of::<U>() * len
    };

    let out: &'a [V] = unsafe { std::slice::from_raw_parts(ptr as *const V, size) };

    out
}

pub fn transmute_values_mut<'a, U, V, const R: bool>(values: &'a mut [U]) -> &'a mut [V] {
    let ptr = values.as_mut_ptr();
    let len = values.len();

    assert!(
        (ptr as usize) % std::mem::align_of::<V>() == 0,
        "trying to cast with mismatched layout"
    );

    let size = if R {
        assert!(len % std::mem::size_of::<V>() == 0);
        len / std::mem::size_of::<V>()
    } else {
        std::mem::size_of::<U>() * len
    };

    let out: &'a mut [V] = unsafe { std::slice::from_raw_parts_mut(ptr as *mut V, size) };

    out
}

pub fn async_copy<T: Copy + Send + Sync>(worker: &Worker, dest: &mut [T], src: &[T]) {
    let length = dest.len();
    assert_eq!(length, src.len());
    
    worker.scope(length, |scope, chunk_size|{
        for (range, chunk) in chunks_mut_with_ranges!(chunk_size, dest) {
            let src_ref = &src;
            scope.spawn(move |_| {
                chunk.copy_from_slice(&src_ref[range]);
            });
        }
    });
}

pub fn fill_with_zeros<F: PrimeField>(worker: &Worker, dest: &mut [F]) {
    fill_with(worker, dest, F::zero());
}

pub fn fill_with_ones<F: PrimeField>(worker: &Worker, dest: &mut [F]) {
    fill_with(worker, dest, F::one());
}

pub fn fill_with<T: Copy + Send>(worker: &Worker, dest: &mut [T], value: T) {
    let length = dest.len();
    assert!(length > 0 );
    worker.scope(length, |scope, chunk_size|{
        for (_, chunk) in chunks_mut_with_ranges!(chunk_size, dest) {
            scope.spawn(move |_| {
                for el in chunk.iter_mut() {
                    *el = value;
                }
            });
        }
    });
}

#[cfg(feature = "allocator")]
pub fn empty_vec<T, A: Allocator + Default>(len: usize) -> std::vec::Vec<T, A> {
    let mut res = Vec::with_capacity_in(len, A::default());
    unsafe{
        res.set_len(len);
    }
    res
}
#[cfg(not(feature = "allocator"))]
pub fn empty_vec<T>(len: usize) -> Vec<T> {
    let mut res = Vec::with_capacity(len);
    unsafe{
        res.set_len(len);
    }
    res
}


#[macro_export]
macro_rules! chunks_mut_with_ranges{
    ($cl:expr, $first:expr) => {
        {
            let length = $first.len();
            itertools::izip!(
                ranges_from_length_and_chunk_size(length, $cl).into_iter(),
                $first.chunks_mut($cl)
            )
        }
    };
    ($cl:expr, $first:expr $(, $v:expr)+ ) => {
        {
            let length = $first.len();

            ranges_from_length_and_chunk_size(length, $cl).into_iter().zip(
                itertools::izip!(
                    $first.chunks_mut($cl)
                    $(
                    , $v.chunks_mut($cl)
                    )*
                )
            )
        }
    }
}

pub fn ranges_from_length_and_chunk_size(length: usize, chunk_size: usize) -> Vec<Range<usize>> {
    assert!(chunk_size > 0);
    if length == 0 {
        return vec![];
    }

    let mut result = vec![];
    let mut current_start = 0;

    while (current_start + chunk_size) < length {
        result.push(current_start..(current_start + chunk_size));
        current_start += chunk_size;
    }
    result.push(current_start..length);

    result
}

pub fn async_sort<T: Ord + Send + Copy>(worker: &Worker, vals: &mut Vec<T>) {
    let length = vals.len();
    let num_cpus = if let Ok(num_cpus) = std::env::var("NUM_CPUS"){
        num_cpus.parse().unwrap()
    } else {
        num_cpus::get()
    };

    let mut chunk_size = get_chunk_size(num_cpus, length);

    crossbeam::scope(
        |scope| {
            for chunk in vals.chunks_mut(chunk_size) {
                scope.spawn(move |_| {
                    chunk.sort();
                });
            }
        }
    ).expect("must run sorting chunks");

    while chunk_size < length {
        crossbeam::scope(
            |scope| {
                for chunks in vals.chunks_mut(2 * chunk_size) {
                    scope.spawn(move |_| {
                        if chunks.len() > chunk_size {
                            let mut chunks: Vec<&mut [T]> = chunks.chunks_mut(chunk_size).collect();
                            let chunk2 = chunks.pop().unwrap();
                            let chunk1 = chunks.pop().unwrap();
                            sorting_merge(chunk1, chunk2);
                        }
                    });
                }
            }
        ).expect("must run merging sorted chunks");

        chunk_size *= 2;
    }
}

fn sorting_merge<T: Ord + Copy>(vals1: &mut [T], vals2: &mut [T]) {
    let len1 = vals1.len();
    let len2 = vals2.len();
    let length = len1 + len2;
    let mut res = Vec::with_capacity(length);

    let (mut i, mut j) = (0, 0);

    for _ in 0..length {
        if i == len1 {
            res.push(vals2[j]);
            j += 1;
        } else if j == len2 {
            res.push(vals1[i]);
            i += 1;
        } else if vals1[i] > vals2[j] {
            res.push(vals2[j]);
            j += 1;
        } else {
            res.push(vals1[i]);
            i += 1;
        }
    }

    vals1.copy_from_slice(&res[..len1]);
    vals2.copy_from_slice(&res[len1..]);
}

pub fn get_chunk_size(num_cpus: usize, elements: usize) -> usize {
    let chunk_size = if elements <= num_cpus {
        1
    } else {
        assert!(
            elements >= num_cpus,
            "received {} elements to spawn {} threads",
            elements,
            num_cpus
        );
        if elements % num_cpus == 0 {
            elements / num_cpus
        } else {
            elements / num_cpus + 1
        }
    };

    chunk_size
}

use bit_vec::BitVec;
pub fn bitvec_to_field_buffer(
    worker: &Worker,
    bitvec: &BitVec,
    buffer: &mut [Fr],
) {
    worker.scope(bitvec.len(), |scope, chunk_len| {
        for (i, chunk) in buffer.chunks_mut(chunk_len).enumerate() {
            let bitvec_ref = &bitvec;
            scope.spawn(move |_| {
                for (bit, el) in bitvec_ref.iter().skip(i*chunk_len).zip(chunk.iter_mut()) {
                    if bit {
                        *el = Fr::one();
                    } else {
                        *el = Fr::zero();
                    }
                }
            });
        }
    });
}

pub fn vec_of_bits_to_field_buffer(
    worker: &Worker,
    bitvec: &Vec<bool>,
    buffer: &mut [Fr],
) {
    worker.scope(bitvec.len(), |scope, chunk_len| {
        for (i, chunk) in buffer.chunks_mut(chunk_len).enumerate() {
            let bitvec_ref = &bitvec;
            scope.spawn(move |_| {
                for (bit, el) in bitvec_ref.iter().skip(i*chunk_len).zip(chunk.iter_mut()) {
                    if *bit {
                        *el = Fr::one();
                    } else {
                        *el = Fr::zero();
                    }
                }
            });
        }
    });
}


#[cfg(not(feature="allocator"))]
macro_rules! new_vec_with_allocator {
    ($capacity:expr) => {
        Vec::with_capacity($capacity)
    }
}

#[cfg(feature="allocator")]
macro_rules! new_vec_with_allocator {
    ($capacity:expr) => {
        Vec::with_capacity_in($capacity, A::default())
    }
}
