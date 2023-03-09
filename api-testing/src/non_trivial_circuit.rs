use super::*;

use zkevm_test_harness::franklin_crypto::bellman::compact_bn256::{Bn256, Fr, G1Affine, G1};
use zkevm_test_harness::franklin_crypto::bellman::plonk::better_better_cs::cs::MainGate;
use zkevm_test_harness::franklin_crypto::bellman::plonk::better_better_cs::cs::*;
use zkevm_test_harness::franklin_crypto::{
    bellman::plonk::better_better_cs::{
        cs::{Circuit, GateInternal},
        gates::selector_optimized_with_d_next::SelectorOptimizedWidth4MainGateWithDNext,
    },
    plonk::circuit::{
        allocated_num::Num, boolean::Boolean, custom_rescue_gate::Rescue5CustomGate,
        linear_combination::LinearCombination,
    },
};

use bellman::{Engine, Field, PrimeField, SynthesisError};
use zkevm_test_harness::sync_vm::rescue_poseidon::{
    circuit_generic_hash, CustomGate, HashParams, RescueParams,
};
// use rand::{thread_rng, Rand, Rng};

#[derive(Debug)]
pub struct NonTrivialCircuit<E: Engine, MG: MainGate<E>> {
    log_degree: usize,
    hash_params: RescueParams<E, 2, 3>,
    m: std::marker::PhantomData<E>,
    main_gate: MG,
}

impl<E: Engine, MG: MainGate<E>> NonTrivialCircuit<E, MG> {
    pub fn new() -> Self {
        let log_degree = if let Ok(log_degree) = std::env::var("LOG_BASE") {
            log_degree.parse::<usize>().unwrap()
        } else {
            10
        };
        let mut hash_params = RescueParams::default();
        hash_params.use_custom_gate(CustomGate::QuinticWidth4);
        Self {
            log_degree,
            hash_params,
            main_gate: MG::default(),
            m: std::marker::PhantomData,
        }
    }
}

impl<E: Engine, MG: MainGate<E>> Circuit<E> for NonTrivialCircuit<E, MG> {
    type MainGate = MG;

    fn declare_used_gates() -> Result<Vec<Box<dyn GateInternal<E>>>, SynthesisError> {
        Ok(vec![
            MG::default().into_internal(),
            Rescue5CustomGate::default().into_internal(),
        ])
    }

    fn synthesize<CS: ConstraintSystem<E>>(&self, cs: &mut CS) -> Result<(), SynthesisError> {
        let columns = vec![
            PolyIdentifier::VariablesPolynomial(0),
            PolyIdentifier::VariablesPolynomial(1),
            PolyIdentifier::VariablesPolynomial(2),
        ];
        let bit_len = 4;
        let modulus = 1usize << bit_len;
        let and_table = LookupTableApplication::new_and_table(bit_len, columns.clone())?;
        let and_table_name = and_table.functional_name();
        cs.add_table(and_table)?;
        let table = cs.get_table(&and_table_name)?;

        let dummy = CS::get_dummy_variable();
        let num_keys_and_values = table.width();
        dbg!(table.size());

        let num_gates = (1 << self.log_degree) - 1;

        let cost_of_single_call = 99;
        assert!(
            table.size() + cost_of_single_call < num_gates,
            "there is no free room for main and custom gates"
        );

        let final_num_gates = num_gates - table.size();
        let mut num_loop = final_num_gates / cost_of_single_call;
        if final_num_gates % cost_of_single_call != 0 || num_loop == 0 {
            num_loop += 1;
        }

        // allocate at least 1 pub gate
        let p0 = cs.alloc_input(|| Ok(E::Fr::from_str("33").unwrap()))?;
        let p1 = cs.alloc_input(|| Ok(E::Fr::from_str("66").unwrap()))?;
        let p2 = cs.alloc_input(|| Ok(E::Fr::from_str("99").unwrap()))?;

        for loop_id in 1..=num_loop {
            let input = [Num::alloc(
                cs,
                Some(E::Fr::from_str(&format!("{}22", loop_id)).unwrap()),
            )?; 2];
            let _ = circuit_generic_hash(cs, &input, &self.hash_params, None)?;

            let binary_x_wit = loop_id % modulus;
            let binary_x_value = E::Fr::from_str(&binary_x_wit.to_string()).unwrap();

            let binary_y_wit = (loop_id + 1) % modulus;
            let binary_y_value = E::Fr::from_str(&binary_y_wit.to_string()).unwrap();

            let binary_x = cs.alloc(|| Ok(binary_x_value))?;

            let binary_y = cs.alloc(|| Ok(binary_y_value))?;

            let and_result_value = table.query(&[binary_x_value, binary_y_value])?[0];

            let binary_z = cs.alloc(|| Ok(and_result_value))?;

            cs.begin_gates_batch_for_step()?;

            let vars = [binary_x, binary_y, binary_z, dummy];
            cs.allocate_variables_without_gate(&vars, &[])?;

            cs.apply_single_lookup_gate(&vars[..num_keys_and_values], table.clone())?;

            cs.end_gates_batch_for_step()?;

            let a_wit = E::Fr::from_str(&format!("{}33", loop_id)).unwrap();
            let a = Num::alloc(cs, Some(a_wit))?;
            let b_wit = E::Fr::from_str(&format!("{}66", loop_id)).unwrap();
            let b = Num::alloc(cs, Some(b_wit))?;
            let c_wit = E::Fr::from_str(&format!("{}99", loop_id)).unwrap();
            let c = Num::alloc(cs, Some(c_wit))?;

            let one = E::Fr::one();
            let mut minus_one = one;
            minus_one.negate();

            let mut lc = LinearCombination::zero();
            lc.add_assign_number_with_coeff(&a, one);
            lc.add_assign_number_with_coeff(&a, minus_one);
            lc.add_assign_number_with_coeff(&b, one);
            lc.add_assign_number_with_coeff(&b, minus_one);
            lc.add_assign_number_with_coeff(&c, one);
            lc.add_assign_number_with_coeff(&c, minus_one);
            lc.enforce_zero(cs)?;

            let a_wit = E::Fr::from_str(&format!("{}44", loop_id)).unwrap();
            let a = Num::alloc(cs, Some(a_wit))?;
            let b_wit = E::Fr::from_str(&format!("{}55", loop_id)).unwrap();
            let b = Num::alloc(cs, Some(b_wit))?;
            let flag = Boolean::alloc(cs, Some(loop_id % 2 != 0))?;
            Num::conditionally_select(cs, &flag, &a, &b)?;
        }
        Ok(())
    }
}
