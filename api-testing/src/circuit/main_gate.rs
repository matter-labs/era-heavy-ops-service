use std::marker::PhantomData;

use zkevm_test_harness::bellman::{
    plonk::better_better_cs::{
        cs::{
            ArithmeticTerm, Circuit, ConstraintSystem, GateInternal, MainGate, MainGateTerm,
            Width4MainGateWithDNext,
        },
        data_structures::PolyIdentifier,
        lookup_tables::LookupTableApplication,
    },
    Engine, Field, PrimeField, SynthesisError,
};

use super::*;

pub struct TestMainGateOnly<E: Engine> {
    log_degree: usize,
    _marker: std::marker::PhantomData<E>,
}

impl<E: Engine> TestMainGateOnly<E> {
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

impl<E: Engine> Circuit<E> for TestMainGateOnly<E> {
    type MainGate = Width4MainGateWithDNext;

    fn synthesize<CS: ConstraintSystem<E>>(&self, cs: &mut CS) -> Result<(), SynthesisError> {
        let a = cs.alloc_input(|| Ok(E::Fr::from_str("10").unwrap()))?;

        println!("A = {:?}", a);

        let b = cs.alloc_input(|| Ok(E::Fr::from_str("20").unwrap()))?;

        println!("B = {:?}", b);

        let c = cs.alloc(|| Ok(E::Fr::from_str("200").unwrap()))?;

        println!("C = {:?}", c);

        let d = cs.alloc(|| Ok(E::Fr::from_str("100").unwrap()))?;

        println!("D = {:?}", d);

        let one = E::Fr::one();

        let mut two = one;
        two.double();

        let mut negative_one = one;
        negative_one.negate();

        // 2a - b = 0

        for _ in 0..(1 << (self.log_degree - 2)) {
            let two_a = ArithmeticTerm::from_variable_and_coeff(a, two);
            let minus_b = ArithmeticTerm::from_variable_and_coeff(b, negative_one);
            let mut term = MainGateTerm::new();
            term.add_assign(two_a);
            term.add_assign(minus_b);

            cs.allocate_main_gate(term)?;

            // c - a*b == 0

            let mut ab_term = ArithmeticTerm::from_variable(a).mul_by_variable(b);
            ab_term.scale(&negative_one);
            let c_term = ArithmeticTerm::from_variable(c);
            let mut term = MainGateTerm::new();
            term.add_assign(c_term);
            term.add_assign(ab_term);

            cs.allocate_main_gate(term)?;

            // d - 100 == 0

            let hundred = ArithmeticTerm::constant(E::Fr::from_str("100").unwrap());
            let d_term = ArithmeticTerm::from_variable(d);
            let mut term = MainGateTerm::new();
            term.add_assign(d_term);
            term.sub_assign(hundred);

            cs.allocate_main_gate(term)?;

            // let gamma = cs.alloc_input(|| {
            //     Ok(E::Fr::from_str("20").unwrap())
            // })?;

            let gamma = cs.alloc(|| Ok(E::Fr::from_str("20").unwrap()))?;

            // gamma - b == 0

            let gamma_term = ArithmeticTerm::from_variable(gamma);
            let b_term = ArithmeticTerm::from_variable(b);
            let mut term = MainGateTerm::new();
            term.add_assign(gamma_term);
            term.sub_assign(b_term);

            cs.allocate_main_gate(term)?;

            // 2a
            let mut term = MainGateTerm::<E>::new();
            term.add_assign(ArithmeticTerm::from_variable_and_coeff(a, two));

            let dummy = CS::get_dummy_variable();

            // 2a - d_next = 0

            let (vars, mut coeffs) = CS::MainGate::format_term(term, dummy)?;
            *coeffs.last_mut().unwrap() = negative_one;

            // here d is equal = 2a, so we need to place b there
            // and compensate it with -b somewhere before

            cs.new_single_gate_for_trace_step(&CS::MainGate::default(), &coeffs, &vars, &[])?;

            let mut term = MainGateTerm::<E>::new();
            term.add_assign(ArithmeticTerm::from_variable(b));

            // b + 0 + 0 - b = 0
            let (mut vars, mut coeffs) = CS::MainGate::format_term(term, dummy)?;
            coeffs[3] = negative_one;
            vars[3] = b;

            cs.new_single_gate_for_trace_step(&CS::MainGate::default(), &coeffs, &vars, &[])?;
        }

        Ok(())
    }
}
