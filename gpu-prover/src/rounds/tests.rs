use super::*;
use bellman::plonk::polynomials::*;
use bellman::worker::Worker;

use rand::{thread_rng, Rand};


// #[test]
// fn dummy_test_wrapper() {
//     dummy_test().unwrap();
// }

// #[test]
// fn dummy_test() -> Result<(), GpuError> {
//     let log_degree = if let Ok(log_degree) = std::env::var("LOG_BASE") {
//         log_degree.parse().unwrap()
//     } else {
//         4usize
//     };

//     let degree = 1 << log_degree;
//     let ctx = GpuContext::init_for_arithmetic(0).expect("init gpu");
//     let ctx = Arc::new(ctx);
//     let rng = &mut thread_rng();
//     let worker = Worker::new();

//     let constant = Fr::rand(rng);
//     let mut this: HVec<Fr> = HVec::empty_pinned(degree);
//     let mut other: HVec<Fr> = HVec::empty_pinned(degree);
//     generate_scalars_to_buf(this.as_mut());
//     generate_scalars_to_buf(other.as_mut());

//     let mut this_poly = Polynomial::from_values(this.as_ref().to_vec()).unwrap();
//     let other_poly = Polynomial::from_values(other.as_ref().to_vec()).unwrap();

//     print!("add constant::");
//     add_constant(ctx.clone(), &mut this, constant.clone())?;
//     this.d2h(ctx.clone())?;
//     this_poly.add_constant(&worker, &constant);
//     ctx.sync()?;
//     assert_eq!(this.as_ref()[..], this_poly.as_ref()[..]);
//     println!("ok");

//     Ok(())
// }
