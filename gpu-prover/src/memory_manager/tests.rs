use super::*;
use bellman::{pairing::CurveAffine, plonk::fft::cooley_tukey_ntt::bitreverse};
use franklin_crypto::bellman::CurveProjective;
use rand::thread_rng;
type TestConfigs = A100_40GB_2GPU_Test_Configs; // G5_5GB_Testing_Configs;

fn init_manager() -> DeviceMemoryManager<Fr, TestConfigs> {
    let degree = <TestConfigs as ManagerConfigs>::FULL_SLOT_SIZE;
    let bases = vec![CompactG1Affine::zero(); degree];
    init_manager_with_bases(&bases)
}

fn init_manager_with_bases(bases: &[CompactG1Affine]) -> DeviceMemoryManager<Fr, TestConfigs> {
    let device_ids = cuda_bindings::devices().unwrap();
    let memory_limit = 40;
    let mem_info = cuda_bindings::device_info(0).unwrap();
    let available_memory_in_gb = mem_info.total / 1024 / 1024 / 1024;
    // dbg!(memory_limit);
    // dbg!(available_memory_in_gb);
    let device_ids = if available_memory_in_gb >= memory_limit {
        vec![0]
    } else {
        vec![0, 2]
    };

    DeviceMemoryManager::<Fr, TestConfigs>::init(&device_ids, &bases).unwrap()
}

#[test]
fn test_all() {
    test_manager_fft();
    test_manager_coset_fft();
    test_manager_coset_ifft();
    test_manager_bitreversing();

    test_manager_msm();
    test_manager_multiple_msms();

    test_manager_add_constant();
    test_manager_sub_constant();
    test_manager_mul_constant();
    test_manager_add_assign();
    test_manager_sub_assign();
    test_manager_mul_assign();
    test_manager_add_assign_scaled();
    test_manager_sub_assign_scaled();

    test_manager_grand_product();
    test_manager_batch_inversion();
    test_manager_distribute_omega_powers();
    test_manager_x_values();
    test_manager_lagrange_poly_values();
}

fn test_manager_fft() {
    println!("fft");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let mut q_a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, q_a.get_values_mut().unwrap());

    manager
        .async_copy_to_device(&mut q_a, PolyId::QA, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .multigpu_ifft_to_free_slot(PolyId::QA, false)
        .unwrap();
    manager.free_slot(PolyId::QA, PolyForm::Values);
    manager
        .multigpu_fft_to_free_slot(PolyId::QA, false)
        .unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::QA, PolyForm::Values)
        .unwrap();

    assert_eq!(
        manager
            .get_host_slot_values(PolyId::QA, PolyForm::Values)
            .unwrap(),
        q_a.get_values().unwrap()
    );
}

fn test_manager_coset_fft() {
    println!("coset fft");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();
    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Monomial, 0..degree)
        .unwrap();
    for device_id in 0..<TestConfigs as ManagerConfigs>::NUM_GPUS {
        manager.ctx[device_id].sync().unwrap();
    }

    let coset_idx = 2;
    let coset_omega = domain_generator::<Fr>(4 * degree);
    let mut g = Fr::multiplicative_generator();
    for _ in 0..(bitreverse(coset_idx, 2)) {
        g.mul_assign(&coset_omega);
    }
    let mut u = Fr::one();
    for v in a.get_values_mut().unwrap().iter_mut() {
        v.mul_assign(&u);
        u.mul_assign(&g);
    }

    manager
        .async_copy_to_device(&mut a, PolyId::B, PolyForm::Monomial, 0..degree)
        .unwrap();

    manager.multigpu_coset_fft(PolyId::A, coset_idx).unwrap();
    manager
        .multigpu_bitreverse(PolyId::A, PolyForm::LDE(coset_idx))
        .unwrap();
    manager.multigpu_fft(PolyId::B, false).unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::A, PolyForm::LDE(coset_idx))
        .unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::B, PolyForm::Values)
        .unwrap();

    assert_eq!(
        manager
            .get_host_slot_values(PolyId::A, PolyForm::LDE(coset_idx))
            .unwrap(),
        manager
            .get_host_slot_values(PolyId::B, PolyForm::Values)
            .unwrap()
    );
}

fn test_manager_coset_ifft() {
    println!("coset ifft");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Monomial, 0..degree)
        .unwrap();
    for device_id in 0..<TestConfigs as ManagerConfigs>::NUM_GPUS {
        manager.ctx[device_id].sync().unwrap();
    }

    let coset_idx = 2;

    manager.multigpu_coset_fft(PolyId::A, coset_idx).unwrap();
    manager.multigpu_coset_ifft(PolyId::A, coset_idx).unwrap();

    manager
        .copy_from_device_to_host_pinned(PolyId::A, PolyForm::Monomial)
        .unwrap();

    assert_eq!(
        manager
            .get_host_slot_values(PolyId::A, PolyForm::Monomial)
            .unwrap(),
        a.get_values().unwrap()
    );
}

fn transform_bases(bases: &[CompactG1Affine]) -> Vec<G1Affine> {
    use bellman::CurveAffine;
    let mut compact_bases = vec![G1Affine::zero(); bases.len()];
    for (p, cp) in bases.iter().zip(compact_bases.iter_mut()) {
        let (x, y) = p.as_xy();
        let x = x.clone();
        let y = y.clone();
        let x = unsafe { std::mem::transmute(x) };
        let y = unsafe { std::mem::transmute(y) };
        *cp = G1Affine::from_xy_checked(x, y).unwrap();
    }
    compact_bases
}

type ComptactBn256 = bellman::compact_bn256::Bn256;
fn test_manager_msm() {
    println!("msm");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    if degree > 1 << 16 {
        return;
    }
    let compact_bases = generate_bases::<ComptactBn256>(&worker, degree);
    let bases = transform_bases(&compact_bases);
    let mut manager = init_manager_with_bases(&compact_bases);

    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Monomial, 0..degree)
        .unwrap();
    let handle = manager.msm(PolyId::A).unwrap();

    let expected = simple_msm::<Bn256>(&bases[..], a.get_values().unwrap());
    let actual = handle.get_result(&mut manager).unwrap();

    assert_eq!(expected, actual);
}

fn test_manager_multiple_msms() {
    println!("multiple msms");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    if degree > 1 << 16 {
        return;
    }
    let compact_bases = generate_bases::<ComptactBn256>(&worker, degree);
    let bases = transform_bases(&compact_bases);
    let mut manager = init_manager_with_bases(&compact_bases);

    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());
    let mut b = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, b.get_values_mut().unwrap());

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Monomial, 0..degree)
        .unwrap();
    let handle_a = manager.msm(PolyId::A).unwrap();
    manager
        .async_copy_to_device(&mut b, PolyId::B, PolyForm::Monomial, 0..degree)
        .unwrap();
    let handle_b = manager.msm(PolyId::B).unwrap();

    let expected_a = simple_msm::<Bn256>(&bases[..], a.get_values().unwrap());
    let actual_a = handle_a.get_result(&mut manager).unwrap();

    assert_eq!(expected_a, actual_a);

    let expected_b = simple_msm::<Bn256>(&bases[..], b.get_values().unwrap());
    let actual_b = handle_b.get_result(&mut manager).unwrap();

    assert_eq!(expected_b, actual_b);
}

fn simple_msm<E: Engine>(bases: &[E::G1Affine], scalars: &[E::Fr]) -> E::G1Affine {
    use bellman::CurveProjective;
    let mut result = <E::G1 as CurveProjective>::zero();
    for (base, scalar) in bases.iter().zip(scalars.iter()) {
        result.add_assign(&base.mul(*scalar));
    }
    result.into_affine()
}

fn test_manager_add_constant() {
    println!("add constant");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());

    use rand::Rng;
    let constant: Fr = thread_rng().gen();

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .add_constant(PolyId::A, PolyForm::Values, constant)
        .unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::A, PolyForm::Values)
        .unwrap();

    for (i, j) in a.get_values_mut().unwrap().iter_mut().zip(
        manager
            .get_host_slot_values(PolyId::A, PolyForm::Values)
            .unwrap()
            .iter(),
    ) {
        i.add_assign(&constant);
        assert_eq!(*i, *j);
    }
}

fn test_manager_sub_constant() {
    println!("sub constant");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());

    use rand::Rng;
    let constant: Fr = thread_rng().gen();

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .sub_constant(PolyId::A, PolyForm::Values, constant)
        .unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::A, PolyForm::Values)
        .unwrap();

    for (i, j) in a.get_values_mut().unwrap().iter_mut().zip(
        manager
            .get_host_slot_values(PolyId::A, PolyForm::Values)
            .unwrap()
            .iter(),
    ) {
        i.sub_assign(&constant);
        assert_eq!(*i, *j);
    }
}

fn test_manager_mul_constant() {
    println!("mul constant");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());

    use rand::Rng;
    let constant: Fr = thread_rng().gen();

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .mul_constant(PolyId::A, PolyForm::Values, constant)
        .unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::A, PolyForm::Values)
        .unwrap();

    for (i, j) in a.get_values_mut().unwrap().iter_mut().zip(
        manager
            .get_host_slot_values(PolyId::A, PolyForm::Values)
            .unwrap()
            .iter(),
    ) {
        i.mul_assign(&constant);
        assert_eq!(*i, *j);
    }
}

fn test_manager_add_assign() {
    println!("add assign");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());
    let mut b = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, b.get_values_mut().unwrap());

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .async_copy_to_device(&mut b, PolyId::B, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .add_assign(PolyId::A, PolyId::B, PolyForm::Values)
        .unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::A, PolyForm::Values)
        .unwrap();

    for (i, j) in a
        .get_values_mut()
        .unwrap()
        .iter_mut()
        .zip(b.get_values().unwrap().iter())
    {
        i.add_assign(j);
    }

    for (i, j) in a.get_values_mut().unwrap().iter_mut().zip(
        manager
            .get_host_slot_values(PolyId::A, PolyForm::Values)
            .unwrap()
            .iter(),
    ) {
        assert_eq!(*i, *j);
    }
}

fn test_manager_sub_assign() {
    println!("sub assign");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());
    let mut b = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, b.get_values_mut().unwrap());

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .async_copy_to_device(&mut b, PolyId::B, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .sub_assign(PolyId::A, PolyId::B, PolyForm::Values)
        .unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::A, PolyForm::Values)
        .unwrap();

    for (i, j) in a
        .get_values_mut()
        .unwrap()
        .iter_mut()
        .zip(b.get_values().unwrap().iter())
    {
        i.sub_assign(j);
    }

    for (i, j) in a.get_values_mut().unwrap().iter_mut().zip(
        manager
            .get_host_slot_values(PolyId::A, PolyForm::Values)
            .unwrap()
            .iter(),
    ) {
        assert_eq!(*i, *j);
    }
}

fn test_manager_mul_assign() {
    println!("mul assign");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());
    let mut b = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, b.get_values_mut().unwrap());

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .async_copy_to_device(&mut b, PolyId::B, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .mul_assign(PolyId::A, PolyId::B, PolyForm::Values)
        .unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::A, PolyForm::Values)
        .unwrap();

    for (i, j) in a
        .get_values_mut()
        .unwrap()
        .iter_mut()
        .zip(b.get_values().unwrap().iter())
    {
        i.mul_assign(j);
    }

    for (i, j) in a.get_values_mut().unwrap().iter_mut().zip(
        manager
            .get_host_slot_values(PolyId::A, PolyForm::Values)
            .unwrap()
            .iter(),
    ) {
        assert_eq!(*i, *j);
    }
}

fn test_manager_add_assign_scaled() {
    println!("add assign scaled");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());
    let mut b = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, b.get_values_mut().unwrap());

    use rand::Rng;
    let constant: Fr = thread_rng().gen();

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .async_copy_to_device(&mut b, PolyId::B, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .add_assign_scaled(PolyId::A, PolyId::B, PolyForm::Values, constant)
        .unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::A, PolyForm::Values)
        .unwrap();

    for (i, j) in a
        .get_values_mut()
        .unwrap()
        .iter_mut()
        .zip(b.get_values_mut().unwrap().iter_mut())
    {
        j.mul_assign(&constant);
        i.add_assign(j);
    }

    for (i, j) in a.get_values_mut().unwrap().iter_mut().zip(
        manager
            .get_host_slot_values(PolyId::A, PolyForm::Values)
            .unwrap()
            .iter(),
    ) {
        assert_eq!(*i, *j);
    }
}

fn test_manager_sub_assign_scaled() {
    println!("sub assign scaled");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());
    let mut b = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, b.get_values_mut().unwrap());

    use rand::Rng;
    let constant: Fr = thread_rng().gen();

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .async_copy_to_device(&mut b, PolyId::B, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .sub_assign_scaled(PolyId::A, PolyId::B, PolyForm::Values, constant)
        .unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::A, PolyForm::Values)
        .unwrap();

    for (i, j) in a
        .get_values_mut()
        .unwrap()
        .iter_mut()
        .zip(b.get_values_mut().unwrap().iter_mut())
    {
        j.mul_assign(&constant);
        i.sub_assign(j);
    }

    for (i, j) in a.get_values_mut().unwrap().iter_mut().zip(
        manager
            .get_host_slot_values(PolyId::A, PolyForm::Values)
            .unwrap()
            .iter(),
    ) {
        assert_eq!(*i, *j);
    }
}

fn test_manager_grand_product() {
    println!("grand product");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .shifted_grand_product_to_new_slot(PolyId::A, PolyId::B, PolyForm::Values)
        .unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::B, PolyForm::Values)
        .unwrap();

    shifted_grand_product(a.get_values_mut().unwrap());

    for (i, j) in a.get_values_mut().unwrap().iter_mut().zip(
        manager
            .get_host_slot_values(PolyId::B, PolyForm::Values)
            .unwrap()
            .iter(),
    ) {
        assert_eq!(*i, *j);
    }
}

fn shifted_grand_product(poly: &mut [Fr]) {
    let len = poly.len();
    let mut x = Fr::one();
    for i in 0..len {
        let y = poly[i];
        poly[i] = x;
        x.mul_assign(&y);
    }
}

fn test_manager_batch_inversion() {
    println!("batch inversion");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .batch_inversion(PolyId::A, PolyForm::Values)
        .unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::A, PolyForm::Values)
        .unwrap();

    for (i, j) in a.get_values_mut().unwrap().iter_mut().zip(
        manager
            .get_host_slot_values(PolyId::A, PolyForm::Values)
            .unwrap()
            .iter(),
    ) {
        i.mul_assign(j);
        assert_eq!(*i, Fr::one());
    }
}

fn test_manager_distribute_omega_powers() {
    println!("distribute powers");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Values, 0..degree)
        .unwrap();
    manager
        .distribute_omega_powers(
            PolyId::A,
            PolyForm::Values,
            <TestConfigs as ManagerConfigs>::FULL_SLOT_SIZE_LOG,
            0,
            1,
            false,
        )
        .unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::A, PolyForm::Values)
        .unwrap();

    let base: Fr = domain_generator::<Fr>(degree);
    distribute_powers(a.get_values_mut().unwrap(), base);

    for (i, j) in a.get_values().unwrap().iter().zip(
        manager
            .get_host_slot_values(PolyId::A, PolyForm::Values)
            .unwrap()
            .iter(),
    ) {
        assert_eq!(*i, *j);
    }
}

fn distribute_powers(poly: &mut [Fr], base: Fr) {
    let len = poly.len();
    let mut tmp = Fr::one();
    for i in 0..len {
        poly[i].mul_assign(&tmp);
        tmp.mul_assign(&base);
    }
}

fn test_manager_x_values() {
    println!("domain elements");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let omega = domain_generator::<Fr>(degree);
    let mut a = vec![];
    let mut tmp = Fr::one();
    for _ in 0..degree {
        a.push(tmp);
        tmp.mul_assign(&omega);
    }

    manager
        .create_x_poly_in_free_slot(PolyId::A, PolyForm::Values)
        .unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::A, PolyForm::Values)
        .unwrap();

    assert_eq!(
        manager
            .get_host_slot_values(PolyId::A, PolyForm::Values)
            .unwrap(),
        &a[..]
    );
}

fn test_manager_lagrange_poly_values() {
    println!("lagrange values");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let point = 3;

    manager
        .create_lagrange_poly_in_free_slot(PolyId::Tmp, PolyForm::Monomial, point)
        .unwrap();
    manager
        .copy_from_device_to_host_pinned(PolyId::Tmp, PolyForm::Monomial)
        .unwrap();

    let l_0 = calculate_lagrange_poly::<Fr>(degree, point).unwrap();

    assert_eq!(
        manager
            .get_host_slot_values(PolyId::Tmp, PolyForm::Monomial)
            .unwrap(),
        l_0.as_ref()
    );

    let omega = domain_generator::<Fr>(degree);
    let base = omega.pow([point as u64]);
    let handle = manager.evaluate_at(PolyId::Tmp, base).unwrap();
    assert_eq!(
        Fr::one(),
        handle.get_result::<TestConfigs>(&mut manager).unwrap()
    );
}

fn test_manager_bitreversing() {
    println!("bitreverse");
    let worker = Worker::new();
    let degree = TestConfigs::FULL_SLOT_SIZE;
    let mut manager = init_manager();

    let mut a = AsyncVec::<Fr>::allocate_new(degree);
    generate_scalars_to_buf(&worker, a.get_values_mut().unwrap());

    manager
        .async_copy_to_device(&mut a, PolyId::A, PolyForm::Values, 0..degree)
        .unwrap();
    // manager.bitreverse(PolyId::A, PolyForm::Values, 0).unwrap();
    manager
        .multigpu_bitreverse(PolyId::A, PolyForm::Values)
        .unwrap();

    manager
        .copy_from_device_to_host_pinned(PolyId::A, PolyForm::Values)
        .unwrap();

    for (i, el) in manager
        .get_host_slot_values(PolyId::A, PolyForm::Values)
        .unwrap()
        .iter()
        .enumerate()
    {
        let j = bitreverse(i, <TestConfigs as ManagerConfigs>::FULL_SLOT_SIZE_LOG);
        assert_eq!(a.get_values().unwrap()[j], *el);
    }
}

use bellman::plonk::polynomials::*;
fn calculate_lagrange_poly<F: PrimeField>(
    poly_size: usize,
    poly_number: usize,
) -> Result<Polynomial<F, Coefficients>, SynthesisError> {
    assert!(poly_size.is_power_of_two());
    assert!(poly_number < poly_size);

    let worker = bellman::worker::Worker::new();

    let mut poly = Polynomial::<F, Values>::from_values(vec![F::zero(); poly_size])?;
    poly.as_mut()[poly_number] = F::one();
    let poly = poly.ifft(&worker);

    Ok(poly)
}
