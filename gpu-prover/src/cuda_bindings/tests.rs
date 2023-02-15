use super::*;
use crate::{Crs, CrsForMonomialForm, Worker};
use bellman::Field;

#[test]
fn test_bindings_allocating_and_copying() {
    let domain_size = 1 << 15;

    let worker = Worker::new();
    let crs_mons = Crs::<Bn256, CrsForMonomialForm>::crs_42(domain_size, &worker);

    let mut ctx = GpuContext::new_full(0, &crs_mons.g1_bases.as_ref()).unwrap();

    let mut host_vec_0 = AsyncVec::<Fr>::allocate_new(domain_size);
    let mut host_vec_1 = AsyncVec::<Fr>::allocate_new(domain_size);
    crate::generate_scalars_to_buf(&worker, host_vec_0.get_values_mut().unwrap());

    let mut device_vec_0 = DeviceBuf::<Fr>::async_alloc_in_exec(&ctx, domain_size).unwrap();
    let mut device_vec_1 = DeviceBuf::<Fr>::async_alloc_in_exec(&ctx, domain_size).unwrap();

    host_vec_0.async_copy_to_device(&mut ctx, &mut device_vec_0, 0..domain_size, 0..domain_size).unwrap();
    device_vec_0.async_copy_to_device(&mut ctx, &mut device_vec_1, 0..domain_size, 0..domain_size).unwrap();
    device_vec_1.async_copy_to_host(&mut ctx, &mut host_vec_1, 0..domain_size, 0..domain_size).unwrap();

    assert_eq!(host_vec_0.get_values().unwrap(), host_vec_1.get_values().unwrap());
}

#[test]
fn test_bindings_arithmetic() {
    let domain_size = 1 << 15;

    let worker = Worker::new();
    let crs_mons = Crs::<Bn256, CrsForMonomialForm>::crs_42(domain_size, &worker);

    let mut ctx = GpuContext::new_full(0, &crs_mons.g1_bases.as_ref()).unwrap();

    let mut host_vec_0 = AsyncVec::<Fr>::allocate_new(domain_size);
    crate::generate_scalars_to_buf(&worker, host_vec_0.get_values_mut().unwrap());

    let mut host_vec_1 = AsyncVec::<Fr>::allocate_new(domain_size);
    crate::generate_scalars_to_buf(&worker, host_vec_1.get_values_mut().unwrap());

    let mut device_vec_0 = DeviceBuf::<Fr>::async_alloc_in_exec(&ctx, domain_size).unwrap();
    let mut device_vec_1 = DeviceBuf::<Fr>::async_alloc_in_exec(&ctx, domain_size).unwrap();

    host_vec_0.async_copy_to_device(&mut ctx, &mut device_vec_0, 0..domain_size, 0..domain_size).unwrap();
    host_vec_1.async_copy_to_device(&mut ctx, &mut device_vec_1, 0..domain_size, 0..domain_size).unwrap();

    for (i, j) in host_vec_0
        .get_values_mut()
        .unwrap()
        .iter_mut()
        .zip(
            host_vec_1
                .get_values()
                .unwrap()
                .iter()
        ) 
    {
        i.sub_assign(j);
    }

    device_vec_0.async_exec_op(&mut ctx, Some(&mut device_vec_1), None, 0..domain_size, Operation::Sub).unwrap();

    let mut host_vec_2 = AsyncVec::<Fr>::allocate_new(domain_size);
    host_vec_2.async_copy_from_device(&mut ctx, &mut device_vec_0, 0..domain_size, 0..domain_size).unwrap();

    assert_eq!(host_vec_0.get_values().unwrap(), host_vec_2.get_values().unwrap());
}