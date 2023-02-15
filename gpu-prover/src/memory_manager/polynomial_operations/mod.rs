use super::*;
use bellman::GenericCurveProjective;

mod arithmetic;
mod heavy_operations;
mod other;

pub use arithmetic::*;
pub use heavy_operations::*;
pub use other::*;

pub struct MSMHandle {
    result: Vec<DeviceBuf<G1>>,
}

impl MSMHandle {
    pub fn from_buffers(buffers: Vec<DeviceBuf<G1>>) -> Self {
        for buf in buffers.iter() {
            assert_eq!(buf.len(), 254, "all buffers should have length 254");
        }

        Self{ result: buffers }
    }

    pub fn get_result<MC: ManagerConfigs>(mut self, manager: &mut DeviceMemoryManager<Fr, MC>) -> GpuResult<G1Affine> {
        assert_eq!(self.result.len(), MC::NUM_GPUS, "number of buffers should be equal to number of GPUs");

        let mut result = G1::zero();
        
        for device_id in 0..MC::NUM_GPUS {
            manager.host_buf_for_msm.async_copy_from_device(
                &mut manager.ctx[device_id],
                &mut self.result[device_id],
                0..NUM_MSM_RESULT_POINTS,
                0..NUM_MSM_RESULT_POINTS
            )?;

            let mut tmp_sum = G1::zero();

            for bucket in manager.host_buf_for_msm.get_values()?.iter().rev() {
                tmp_sum.double();
                tmp_sum.add_assign(bucket);
            }

            result.add_assign(&tmp_sum);
        }

        Ok(result.into_affine())
    }
}

pub struct EvaluationHandle {
    base_pow: Fr,
    result: Vec<DeviceBuf<Fr>>,
}

impl EvaluationHandle {
    pub fn from_buffers_and_base_pow(buffers: Vec<DeviceBuf<Fr>>, base_pow: Fr) -> Self {
        for buf in buffers.iter() {
            assert_eq!(buf.len(), 1, "all buffers should have length 1");
        }

        Self{ result: buffers, base_pow }
    }

    pub fn get_result<MC: ManagerConfigs>(mut self, manager: &mut DeviceMemoryManager<Fr, MC>) -> GpuResult<Fr> {
        assert_eq!(self.result.len(), MC::NUM_GPUS, "number of buffers should be equal to number of GPUs");

        let mut result = Fr::zero();

        for device_id in (0..MC::NUM_GPUS).rev() {
            manager.host_buf_for_poly_eval.async_copy_from_device(
                &mut manager.ctx[device_id],
                &mut self.result[device_id],
                0..NUM_POLY_EVAL_RESULT_ELEMS,
                0..NUM_POLY_EVAL_RESULT_ELEMS,
            )?;

            result.mul_assign(&self.base_pow);
            result.add_assign(&manager.host_buf_for_poly_eval.get_values()?[0]);
        }

        Ok(result)
    }    
}
