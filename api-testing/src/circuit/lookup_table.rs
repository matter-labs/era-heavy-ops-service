use std::marker::PhantomData;

use zkevm_test_harness::bellman::{
    plonk::better_better_cs::{
        cs::{
            ArithmeticTerm, Circuit, ConstraintSystem, Gate, GateInternal, MainGate, MainGateTerm,
            Width4MainGateWithDNext,
        },
        data_structures::PolyIdentifier,
        lookup_tables::LookupTableApplication,
    },
    Engine, Field, PrimeField, SynthesisError,
};

use super::*;
pub struct TestCircuit4WithLookups<E: Engine> {
    log_degree: usize,
    _marker: PhantomData<E>,
}

impl<E: Engine> TestCircuit4WithLookups<E> {
    pub fn new() -> Self {
        let log_degree = if let Ok(log_degree) = std::env::var("LOG_BASE") {
            log_degree.parse::<usize>().unwrap()
        } else {
            10
        };
        Self {
            log_degree,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<E: Engine> Circuit<E> for TestCircuit4WithLookups<E> {
    type MainGate = Width4MainGateWithDNext;

    fn declare_used_gates() -> Result<Vec<Box<dyn GateInternal<E>>>, SynthesisError> {
        Ok(vec![Width4MainGateWithDNext::default().into_internal()])
    }

    fn synthesize<CS: ConstraintSystem<E>>(&self, cs: &mut CS) -> Result<(), SynthesisError> {
        let columns = vec![
            PolyIdentifier::VariablesPolynomial(0),
            PolyIdentifier::VariablesPolynomial(1),
            PolyIdentifier::VariablesPolynomial(2),
        ];
        let range_table = LookupTableApplication::new_range_table_of_width_3(2, columns.clone())?;
        let range_table_name = range_table.functional_name();

        let xor_table = LookupTableApplication::new_xor_table(2, columns.clone())?;
        let xor_table_name = xor_table.functional_name();

        let and_table = LookupTableApplication::new_and_table(2, columns)?;
        let and_table_name = and_table.functional_name();

        cs.add_table(range_table)?;
        cs.add_table(xor_table)?;
        cs.add_table(and_table)?;

        let a = cs.alloc_input(|| Ok(E::Fr::from_str("10").unwrap()))?;

        let b = cs.alloc(|| Ok(E::Fr::from_str("20").unwrap()))?;

        let c = cs.alloc(|| Ok(E::Fr::from_str("200").unwrap()))?;

        let binary_x_value = E::Fr::from_str("3").unwrap();
        let binary_y_value = E::Fr::from_str("1").unwrap();

        let binary_x = cs.alloc(|| Ok(binary_x_value))?;

        let binary_y = cs.alloc(|| Ok(binary_y_value))?;

        let mut negative_one = E::Fr::one();
        negative_one.negate();

        let gates_per_op = 1 << (self.log_degree - 2);

        for _ in 0..gates_per_op {
            // c - a*b == 0
            let mut ab_term = ArithmeticTerm::from_variable(a).mul_by_variable(b);
            ab_term.scale(&negative_one);
            let c_term = ArithmeticTerm::from_variable(c);
            let mut term = MainGateTerm::new();
            term.add_assign(c_term);
            term.add_assign(ab_term);

            cs.allocate_main_gate(term)?;
        }

        let dummy = CS::get_dummy_variable();

        // and table
        for _ in 0..gates_per_op {
            let table = cs.get_table(&and_table_name)?;
            let num_keys_and_values = table.width();

            let and_result_value = table.query(&[binary_x_value, binary_y_value])?[0];

            let binary_z = cs.alloc(|| Ok(and_result_value))?;

            cs.begin_gates_batch_for_step()?;

            let vars = [binary_x, binary_y, binary_z, dummy];
            cs.allocate_variables_without_gate(&vars, &[])?;

            cs.apply_single_lookup_gate(&vars[..num_keys_and_values], table)?;

            cs.end_gates_batch_for_step()?;
        }

        let var_zero = cs.get_explicit_zero()?;

        // range table
        for _ in 0..gates_per_op {
            let table = cs.get_table(&range_table_name)?;
            let num_keys_and_values = table.width();

            cs.begin_gates_batch_for_step()?;

            let mut term = MainGateTerm::<E>::new();
            term.add_assign(ArithmeticTerm::from_variable_and_coeff(
                binary_y,
                E::Fr::zero(),
            ));
            term.add_assign(ArithmeticTerm::from_variable_and_coeff(
                var_zero,
                E::Fr::zero(),
            ));
            term.add_assign(ArithmeticTerm::from_variable_and_coeff(
                var_zero,
                E::Fr::zero(),
            ));

            let (vars, coeffs) = CS::MainGate::format_linear_term_with_duplicates(term, dummy)?;

            cs.new_gate_in_batch(&CS::MainGate::default(), &coeffs, &vars, &[])?;

            cs.apply_single_lookup_gate(&vars[..num_keys_and_values], table)?;

            cs.end_gates_batch_for_step()?;
        }

        // xor table
        for _ in 0..gates_per_op {
            let table = cs.get_table(&xor_table_name)?;
            let num_keys_and_values = table.width();

            let xor_result_value = table.query(&[binary_x_value, binary_y_value])?[0];

            let binary_z = cs.alloc(|| Ok(xor_result_value))?;

            cs.begin_gates_batch_for_step()?;

            let vars = [binary_x, binary_y, binary_z, dummy];
            cs.allocate_variables_without_gate(&vars, &[])?;

            cs.apply_single_lookup_gate(&vars[..num_keys_and_values], table)?;

            cs.end_gates_batch_for_step()?;
        }

        Ok(())
    }
}
