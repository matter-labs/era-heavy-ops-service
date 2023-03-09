use super::*;
use crate::cuda_bindings::GpuResult;
use bellman::plonk::better_better_cs::setup::VerificationKey;
use bellman::plonk::polynomials::Polynomial;
use bit_vec::BitVec;
// use franklin_crypto::plonk::circuit::custom_rescue_gate::Rescue5CustomGate;
use gpu_ffi::bc_event;

use std::io::{Read, Write};

pub const NUM_GATE_SETUP_POLYS: usize = 8;
pub const NUM_SELECTOR_POLYS: usize = 2;
pub const NUM_PERMUTATION_POLYS: usize = 4;
pub const NUM_LOOKUP_TABLE_POLYS: usize = 4;

pub const GATE_SETUP_LIST: [PolyId; 8] = [
    PolyId::QA,
    PolyId::QB,
    PolyId::QC,
    PolyId::QD,
    PolyId::QMab,
    PolyId::QMac,
    PolyId::QConst,
    PolyId::QDNext,
];

use cfg_if::*;

cfg_if! {
    if #[cfg(feature = "allocator")]{
        pub struct AsyncSetup<A: Allocator = CudaAllocator> {
            pub gate_setup_monomials: [AsyncVec<Fr, A>; NUM_GATE_SETUP_POLYS],
            pub gate_selectors_bitvecs: [BitVec; NUM_SELECTOR_POLYS],

            pub lookup_selector_bitvec: BitVec,
            pub lookup_tables_values: [AsyncVec<Fr, A>; NUM_LOOKUP_TABLE_POLYS], // Could have smaller size
            pub lookup_table_type_monomial: AsyncVec<Fr, A>,
        }

        unsafe impl<A: Allocator + Default> Send for AsyncSetup<A> {}
        unsafe impl<A: Allocator + Default> Sync for AsyncSetup<A> {}
    }else{
        pub struct AsyncSetup{
            pub gate_setup_monomials: [AsyncVec<Fr>; NUM_GATE_SETUP_POLYS],
            pub gate_selectors_bitvecs: [BitVec; NUM_SELECTOR_POLYS],

            pub lookup_selector_bitvec: BitVec,
            pub lookup_tables_values: [AsyncVec<Fr>; NUM_LOOKUP_TABLE_POLYS], // Could have smaller size
            pub lookup_table_type_monomial: AsyncVec<Fr>,
        }

        unsafe impl Send for AsyncSetup {}
        unsafe impl Sync for AsyncSetup {}
    }
}

macro_rules! impl_async_setup {
    (impl AsyncSetup $inherent:tt) => {
        #[cfg(feature = "allocator")]
        impl<A: Allocator + Default> AsyncSetup<A> $inherent

        #[cfg(not(feature = "allocator"))]
        impl AsyncSetup $inherent
    };
}

impl_async_setup! {
    impl AsyncSetup{
        pub fn empty() -> Self {
            Self::allocate(1)
        }



        pub fn allocate(length: usize) -> Self {
            Self::allocate_optimized(length, length-1)
        }

        pub fn allocate_optimized(length: usize, length_cols: usize) -> Self {
            assert!(length > length_cols);
            let gate_setup_monomials: Vec<_> = (0..NUM_GATE_SETUP_POLYS).map(|_| AsyncVec::allocate_new(length)).collect();
            let gate_selectors_bitvecs: Vec<_> = (0..NUM_SELECTOR_POLYS).map(|_| BitVec::with_capacity(length)).collect();
            let lookup_tables_values: Vec<_> = (0..NUM_LOOKUP_TABLE_POLYS).map(|_| AsyncVec::allocate_new(length_cols)).collect();

            Self {
                gate_setup_monomials: gate_setup_monomials.try_into().unwrap(),
                gate_selectors_bitvecs: gate_selectors_bitvecs.try_into().unwrap(),

                lookup_selector_bitvec: BitVec::with_capacity(length),
                lookup_tables_values: lookup_tables_values.try_into().unwrap(),
                lookup_table_type_monomial: AsyncVec::allocate_new(length),
            }
        }

        pub fn zeroize(&mut self){
            for poly in self.gate_setup_monomials.iter_mut() {
                poly.zeroize();
            }

            for poly in self.gate_selectors_bitvecs.iter_mut() {
                poly.set_all();
                poly.negate();
            }

            for poly in self.lookup_tables_values.iter_mut() {
                poly.zeroize()
            }

            self.lookup_selector_bitvec.set_all();
            self.lookup_selector_bitvec.negate();

            self.lookup_table_type_monomial.zeroize();
        }

        pub fn write<W: Write>(
            &self,
            mut writer: W,
        ) -> GpuResult<()> {
            for poly in self.gate_setup_monomials.iter() {
                poly.write(&mut writer)?;
            }

            for poly in self.gate_selectors_bitvecs.iter() {
                writer.write_all(&poly.to_bytes()[..]).expect("Can't write AsyncVec");
            }

            for poly in self.lookup_tables_values.iter() {
                poly.write(&mut writer)?;
            }

            writer.write_all(&self.lookup_selector_bitvec.to_bytes()[..]).expect("Can't write AsyncVec");
            self.lookup_table_type_monomial.write(&mut writer)?;

            Ok(())
        }

        pub fn read<R: Read>(
            &mut self,
            mut reader: R,
        ) -> GpuResult<()> {
            for poly in self.gate_setup_monomials.iter_mut() {
                poly.read(&mut reader)?;
            }

            for poly in self.gate_selectors_bitvecs.iter_mut() {
                let mut res_bytes: Vec<u8> = Vec::with_capacity(poly.capacity() >> 3);
                unsafe{ res_bytes.set_len(poly.capacity() >> 3); }
                reader.read_exact(&mut res_bytes).expect("Can't read AsyncVec");
                *poly = BitVec::from_bytes(&res_bytes);
            }

            for poly in self.lookup_tables_values.iter_mut() {
                poly.read(&mut reader)?;
            }

            let mut res_bytes: Vec<u8> = Vec::with_capacity(self.lookup_selector_bitvec.capacity() >> 3);
            unsafe{ res_bytes.set_len(self.lookup_selector_bitvec.capacity() >> 3); }
            reader.read_exact(&mut res_bytes).expect("Can't read AsyncVec");
            self.lookup_selector_bitvec = BitVec::from_bytes(&res_bytes);

            self.lookup_table_type_monomial.read(&mut reader)?;

            Ok(())
        }

        pub fn from_bytes(
            &mut self,
            src: &[u8],
        ) -> GpuResult<()> {
            let mut start = 0;
            let poly_byte_len = self.gate_setup_monomials[0].len() * FIELD_ELEMENT_LEN;
            let tables_poly_byte_len = self.lookup_tables_values[0].len() * FIELD_ELEMENT_LEN;

            for poly in self.gate_setup_monomials.iter_mut() {
                let end = start + poly_byte_len;
                poly.from_bytes(&src[start..end])?;
                start = end;
            }

            for poly in self.gate_selectors_bitvecs.iter_mut() {
                let end = start + poly_byte_len;
                *poly = BitVec::from_bytes(&src[start..end]);
                start = end;
            }

            for poly in self.lookup_tables_values.iter_mut() {
                let end = start + tables_poly_byte_len;
                poly.from_bytes(&src[start..end])?;
                start = end;
            }

            let end = start + poly_byte_len;
            self.lookup_selector_bitvec = BitVec::from_bytes(&src[start..end]);
            start = end;

            let end = start + poly_byte_len;
            self.lookup_table_type_monomial.from_bytes(&src[start..end])?;
            start = end;

            Ok(())
        }

        pub fn to_bytes(
            &mut self,
            dst: &mut [u8],
        ) -> GpuResult<()> {
            let mut start = 0;
            let poly_byte_len = self.gate_setup_monomials[0].len() * FIELD_ELEMENT_LEN;
            let tables_poly_byte_len = self.lookup_tables_values[0].len() * FIELD_ELEMENT_LEN;

            for poly in self.gate_setup_monomials.iter_mut() {
                let end = start + poly_byte_len;
                poly.to_bytes(&mut dst[start..end])?;
                start = end;
            }

            for poly in self.gate_selectors_bitvecs.iter_mut() {
                let end = start + poly_byte_len;
                let poly_bytes = poly.to_bytes();
                dst[start..end].copy_from_slice(&poly_bytes[..]);
                start = end;
            }

            for poly in self.lookup_tables_values.iter_mut() {
                let end = start + tables_poly_byte_len;
                poly.to_bytes(&mut dst[start..end])?;
                start = end;
            }

            let end = start + poly_byte_len;
            let poly_bytes = self.lookup_selector_bitvec.to_bytes();
            dst[start..end].copy_from_slice(&poly_bytes[..]);
            start = end;

            let end = start + poly_byte_len;
            self.lookup_table_type_monomial.to_bytes(&mut dst[start..end])?;
            start = end;

            Ok(())
        }

        pub fn generate_from_assembly<
            S: SynthesisMode,
            MC: ManagerConfigs,
        >(
            &mut self,
            worker: &Worker,
            assembly: &DefaultAssembly<S>,
            manager: &mut DeviceMemoryManager<Fr, MC>,
        ) -> Result<(), ProvingError> {
            assert!(assembly.is_finalized);
            assert!(S::PRODUCE_SETUP);

            let known_gates_list = &assembly.sorted_gates;

            // FIXME:
            // assert_eq!(known_gates_list, &vec![
            //     SelectorOptimizedWidth4MainGateWithDNext::default().into_internal(),
            //     Rescue5CustomGate::default().into_internal(),
            // ]);

            let mut setup_polys_values_map = assembly.make_setup_polynomials(true)?;
            for gate in known_gates_list.iter() {
                let setup_polys = gate.setup_polynomials();
                for (i, id) in setup_polys.into_iter().enumerate() {
                    let values = setup_polys_values_map.remove(&id).expect("must contain setup poly").clone_padded_to_domain()?;

                    manager.copy_to_device_with_host_slot(worker, values.as_ref(), PolyId::Enumerated(i), PolyForm::Values);
                    manager.multigpu_ifft(PolyId::Enumerated(i), false);
                    manager.copy_from_device_with_host_slot(worker, self.gate_setup_monomials[i].get_values_mut()?, PolyId::Enumerated(i), PolyForm::Monomial);

                    manager.free_host_slot(PolyId::Enumerated(i), PolyForm::Values);
                    manager.free_host_slot(PolyId::Enumerated(i), PolyForm::Monomial);
                    manager.free_slot(PolyId::Enumerated(i), PolyForm::Monomial);
                }
            }

            let num_input_gates = assembly.num_input_gates;
            let num_aux_gates = assembly.num_aux_gates;

            for idx in 0..2 {
                let id = &assembly.sorted_gates[idx];
                let mut bv = assembly.aux_gate_density.0.get(id).unwrap().clone();

                if num_input_gates > 0{
                    let bit_value = if idx == 0 {
                        true
                    } else {
                        false
                    };
                    self.gate_selectors_bitvecs[idx].grow(num_input_gates, bit_value);
                }

                let length = self.gate_selectors_bitvecs[idx].capacity() - num_input_gates - 1;
                assert_eq!(length, num_aux_gates);
                bv.truncate(length);
                self.gate_selectors_bitvecs[idx].append(&mut bv);
                self.gate_selectors_bitvecs[idx].grow(1, false);
                assert_eq!(self.gate_selectors_bitvecs[idx].len(), assembly.n() + 1);
            }

            let table_tails = assembly.calculate_t_polynomial_values_for_single_application_tables()?;
            assert_eq!(table_tails.len(), 4);

            let tails_len = table_tails[0].len();
            dbg!(tails_len);

            let size = self.lookup_tables_values[0].len();
            assert!(size >= tails_len);
            let copy_start = size - tails_len;
            let copy_end = copy_start + tails_len;

            for (i, tail) in table_tails.into_iter().enumerate() {
                let  values = self.lookup_tables_values[i].get_values_mut()?;
                fill_with_zeros(worker, &mut values[..]);
                async_copy(worker, &mut values[copy_start..copy_end], &tail[..]);
            }

            let length = self.lookup_selector_bitvec.capacity() - num_input_gates - 1;
            assert_eq!(length, num_aux_gates);
            let mut bv = BitVec::from_elem(length, false);

            for single_application in assembly.tables.iter() {
                let table_name = single_application.functional_name();
                let mut selector = assembly.table_selectors.get(&table_name).unwrap().clone();
                selector.truncate(length);
                bv.or(&selector);
            }

            if num_input_gates > 0{
                self.lookup_selector_bitvec.grow(num_input_gates, false);
            }

            self.lookup_selector_bitvec.append(&mut bv);
            self.lookup_selector_bitvec.grow(1, false);
            assert_eq!(self.lookup_selector_bitvec.len(), assembly.n() + 1);

            let table_type_values = assembly.calculate_table_type_values()?;
            let poly = Polynomial::from_values(table_type_values).unwrap();
            manager.copy_to_device_with_host_slot(worker, poly.as_ref(), PolyId::Enumerated(0), PolyForm::Values);
            manager.multigpu_ifft(PolyId::Enumerated(0), false);
            manager.copy_from_device_with_host_slot(worker, self.lookup_table_type_monomial.get_values_mut()?, PolyId::Enumerated(0), PolyForm::Monomial);

            manager.free_host_slot(PolyId::Enumerated(0), PolyForm::Values);
            manager.free_host_slot(PolyId::Enumerated(0), PolyForm::Monomial);
            manager.free_slot(PolyId::Enumerated(0), PolyForm::Monomial);

            Ok(())
        }
    }
}

#[cfg(feature = "allocator")]
impl<A: Allocator + Default> std::fmt::Debug for AsyncSetup<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Setup")
            .field("gate_setup_monomials", &self.gate_setup_monomials)
            .field("gate_selectors_monomials", &self.gate_selectors_bitvecs)
            .field("lookup_selector_monomial", &self.lookup_selector_bitvec)
            .field("lookup_tables_monomials", &self.lookup_tables_values)
            .field(
                "lookup_table_type_monomial",
                &self.lookup_table_type_monomial,
            )
            .finish()
    }
}
#[cfg(not(feature = "allocator"))]
impl std::fmt::Debug for AsyncSetup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Setup")
            .field("gate_setup_monomials", &self.gate_setup_monomials)
            .field("gate_selectors_monomials", &self.gate_selectors_bitvecs)
            .field("lookup_selector_monomial", &self.lookup_selector_bitvec)
            .field("lookup_tables_monomials", &self.lookup_tables_values)
            .field(
                "lookup_table_type_monomial",
                &self.lookup_table_type_monomial,
            )
            .finish()
    }
}

fn read_u64<R: Read>(reader: &mut R) -> std::io::Result<u64> {
    let mut bytes = vec![0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

use bellman::plonk::better_better_cs::gates::selector_optimized_with_d_next::SelectorOptimizedWidth4MainGateWithDNext;
use bellman::plonk::better_cs::generator::make_non_residues;

pub fn compute_vk_from_assembly<
    C: Circuit<Bn256>,
    MC: ManagerConfigs,
    P: PlonkConstraintSystemParams<Bn256>,
    S: SynthesisMode,
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    crs: &Crs<Bn256, CrsForMonomialForm>,
) -> Result<VerificationKey<Bn256, C>, ProvingError> {
    assert!(S::PRODUCE_SETUP);

    let mut vk = VerificationKey::empty();

    vk.n = assembly.n();
    vk.num_inputs = assembly.num_inputs;
    vk.state_width = P::STATE_WIDTH;
    vk.num_witness_polys = P::WITNESS_WIDTH;
    vk.total_lookup_entries_length = assembly.num_table_lookups;
    vk.non_residues = make_non_residues::<Fr>(vk.state_width - 1);
    vk.g2_elements = [crs.g2_monomial_bases[0], crs.g2_monomial_bases[1]];

    let worker = Worker::new();

    let mut msm_handles = vec![];

    compute_permutation_polynomials(manager, &assembly, &worker)?;
    for i in 0..4 {
        manager.multigpu_ifft_to_free_slot(PolyId::Sigma(i), false)?;
    }

    // GATE SETUP
    let mut tmp_handles = vec![];
    for i in 0..8 {
        copying_setup_poly(manager, &assembly, i)?;

        let poly_id = GATE_SETUP_LIST[i];
        manager.multigpu_ifft(poly_id, false)?;
        let handle = manager.msm(poly_id)?;
        tmp_handles.push(handle);
    }
    msm_handles.push(tmp_handles);

    // GATE SELECTORS
    let mut tmp_handles = vec![];
    for idx in 0..2 {
        let poly_id = if idx == 0 {
            PolyId::QMainSelector
        } else {
            PolyId::QCustomSelector
        };

        get_gate_selector_values_from_assembly(manager, &assembly, &worker, idx)?;
        manager.multigpu_ifft(poly_id, false)?;

        let handle = manager.msm(poly_id)?;
        tmp_handles.push(handle);

        // manager.free_host_slot(poly_id, PolyForm::Values);
    }
    msm_handles.push(tmp_handles);

    // PERMUTATION
    // copy_permutations_to_device_from_assembly(manager, &assembly, &worker)?;

    let mut tmp_handles = vec![];
    for i in 0..4 {
        let handle = manager.msm(PolyId::Sigma(i))?;
        tmp_handles.push(handle);
    }
    msm_handles.push(tmp_handles);

    // LOOKUP TABLES
    upload_t_poly_parts_from_assembly(manager, &assembly, &worker)?;
    let poly_id = [
        PolyId::Col(0),
        PolyId::Col(1),
        PolyId::Col(2),
        PolyId::TableType,
    ];
    let mut tmp_handles = vec![];
    for i in 0..4 {
        manager.multigpu_ifft(poly_id[i], false)?;
        let handle = manager.msm(poly_id[i])?;
        tmp_handles.push(handle);
    }
    msm_handles.push(tmp_handles);

    let mut optional_handles = [None, None];

    // LOOKUP SELECTOR
    get_lookup_selector_from_assembly(manager, &assembly, &worker)?;
    manager.multigpu_ifft(PolyId::QLookupSelector, false)?;
    let handle = manager.msm(PolyId::QLookupSelector)?;
    optional_handles[0] = Some(handle);

    // LOOKUP TABLE TYPE
    get_table_type_from_assembly(manager, &assembly)?;
    manager.multigpu_ifft(PolyId::QTableType, false)?;
    let handle = manager.msm(PolyId::QTableType)?;
    optional_handles[1] = Some(handle);

    for (c, handles) in vec![
        &mut vk.gate_setup_commitments,
        &mut vk.gate_selectors_commitments,
        &mut vk.permutation_commitments,
        &mut vk.lookup_tables_commitments,
    ]
    .into_iter()
    .zip(msm_handles.into_iter())
    {
        for handle in handles.into_iter() {
            c.push(handle.get_result(manager)?);
        }
    }

    let commitment = optional_handles[0].take().unwrap().get_result(manager)?;
    vk.lookup_selector_commitment = Some(commitment);

    let commitment = optional_handles[1].take().unwrap().get_result(manager)?;
    vk.lookup_table_type_commitment = Some(commitment);

    Ok(vk)
}

pub fn copy_permutations_to_device_from_assembly<
    P: PlonkConstraintSystemParams<Bn256>,
    MG: MainGate<Bn256>,
    S: SynthesisMode,
    MC: ManagerConfigs,
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    // assembly: &Assembly<Bn256, P, MG, S, CudaAllocator>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
) -> Result<(), ProvingError> {
    let permutation_polys = assembly.make_permutations(&worker).unwrap();

    for (i, poly) in permutation_polys.into_iter().enumerate() {
        manager.copy_to_device_with_host_slot(
            worker,
            &poly.into_coeffs(),
            PolyId::Sigma(i),
            PolyForm::Values,
        )?;

        if i > 0 {
            manager.free_host_slot(PolyId::Sigma(i - 1), PolyForm::Values);
        }
    }
    manager.free_host_slot(PolyId::Sigma(3), PolyForm::Values);

    for i in 0..4 {
        manager.multigpu_ifft_to_free_slot(PolyId::Sigma(i), false)?;
    }

    Ok(())
}

pub fn upload_t_poly_parts_from_assembly<S: SynthesisMode, MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    worker: &Worker,
) -> Result<(), ProvingError> {
    let poly_id = [
        PolyId::Col(0),
        PolyId::Col(1),
        PolyId::Col(2),
        PolyId::TableType,
    ];

    let t_poly_ends = assembly
        .calculate_t_polynomial_values_for_single_application_tables()
        .unwrap();

    for (i, t_poly) in t_poly_ends.into_iter().enumerate() {
        let copy_start = MC::FULL_SLOT_SIZE - t_poly.len() - 1;

        let mut t_col = manager.get_free_host_slot_values_mut(poly_id[i], PolyForm::Values)?;
        fill_with_zeros(worker, &mut t_col[..copy_start]);
        async_copy(
            worker,
            &mut t_col[copy_start..(MC::FULL_SLOT_SIZE - 1)],
            &t_poly,
        );

        t_col[MC::FULL_SLOT_SIZE - 1] = Fr::zero();
        manager.copy_from_host_pinned_to_device(poly_id[i], PolyForm::Values)?;

        if i > 0 {
            manager.free_host_slot(poly_id[i - 1], PolyForm::Values);
        }
    }
    manager.free_host_slot(PolyId::TableType, PolyForm::Values);

    Ok(())
}

fn u64_from_bytes(src: &[u8]) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(src);
    u64::from_le_bytes(bytes)
}
