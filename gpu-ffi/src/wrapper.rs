use super::*;

#[derive(Clone)]
pub struct GpuContext {
    device_id: usize,
    mem_pool: bc_mem_pool,
    default_stream: bc_stream,
    bases: Option<*const c_void>, // addr of bases on gpu device
    h2d_stream: bc_stream,
    d2h_stream: bc_stream,
    exec_stream: bc_stream,
}
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

const POWERS_OF_OMEGA_COARSE_LOG_COUNT: u32 = 25;
const POWERS_OF_COSET_OMEGA_COARSE_LOG_COUNT: u32 = 14;

impl GpuContext {
    pub fn init(device_id: usize, bases: &[u8]) -> Result<Self, GpuError> {
        set_device(device_id);
        let mem_pool = bc_mem_pool::new(device_id)?;

        let devices = devices()?;
        for idx in 0..devices {
            if idx == device_id as i32 {
                continue;
            }
            mem_pool_enable_peer_access(mem_pool, idx).unwrap();
            device_enable_peer_access(idx).unwrap();
        }

        let len = bases.len() as u64;
        let mut d_bases_ptr: *mut c_void = std::ptr::null_mut();
        println!("allocating device memory for bases ");
        let result = unsafe { bc_malloc(std::ptr::addr_of_mut!(d_bases_ptr), len) };
        if result != 0 {
            return Err(GpuError::CreateContextErr);
        }

        println!("copying {} bytes bases from host to device", len);
        if unsafe { bc_memcpy(d_bases_ptr, bases.as_ptr() as *const c_void, len) } != 0 {
            return Err(GpuError::CreateContextErr);
        }

        println!("setup msm");
        if unsafe { msm_set_up() } != 0 {
            return Err(GpuError::CreateContextErr);
        }

        println!("setup arithmetic ");
        if unsafe {
            ff_set_up(
                POWERS_OF_OMEGA_COARSE_LOG_COUNT,
                POWERS_OF_COSET_OMEGA_COARSE_LOG_COUNT,
            )
        } != 0
        {
            return Err(GpuError::CreateContextErr);
        }

        println!("setup ntt");
        if unsafe { ntt_set_up() } != 0 {
            return Err(GpuError::CreateContextErr);
        }
        
        Ok(Self {
            device_id: device_id,
            mem_pool: mem_pool,
            bases: Some(d_bases_ptr),
            default_stream: bc_stream::new()?,
            h2d_stream: bc_stream::new()?,
            d2h_stream: bc_stream::new()?,
            exec_stream: bc_stream::new()?,
        })
    }
    pub fn init_for_msm(device_id: usize, bases: &[u8]) -> Result<Self, GpuError> {
        set_device(device_id);
        let mem_pool = bc_mem_pool::new(device_id)?;

        let devices = devices()?;
        for idx in 0..devices {
            if idx == device_id as i32 {
                continue;
            }
            mem_pool_enable_peer_access(mem_pool, idx).unwrap();
            device_enable_peer_access(idx).unwrap();
        }

        let len = bases.len() as u64;
        let mut d_bases_ptr = std::ptr::null_mut();
        println!("allocating device memory for bases ");
        let result = unsafe { bc_malloc(std::ptr::addr_of_mut!(d_bases_ptr), len) };
        if result != 0 {
            return Err(GpuError::CreateContextErr);
        }
        println!("copying {} bytes bases from host to device", len);
        if unsafe { bc_memcpy(d_bases_ptr, bases.as_ptr() as *const c_void, len) } != 0 {
            return Err(GpuError::CreateContextErr);
        }

        print!("setup msm ");
        if unsafe { msm_set_up() } != 0 {
            return Err(GpuError::CreateContextErr);
        }

        Ok(Self {
            device_id: device_id,
            mem_pool: mem_pool,
            bases: Some(d_bases_ptr),
            default_stream: bc_stream::new()?,
            h2d_stream: bc_stream::new()?,
            d2h_stream: bc_stream::new()?,
            exec_stream: bc_stream::new()?,
        })
    }

    pub fn init_for_ntt(device_id: usize) -> Result<Self, GpuError> {
        set_device(device_id);
        let mem_pool = bc_mem_pool::new(device_id)?;

        let devices = devices()?;
        for idx in 0..devices {
            if idx == device_id as i32 {
                continue;
            }
            mem_pool_enable_peer_access(mem_pool, idx).unwrap();
            device_enable_peer_access(idx).unwrap();
        }

        unsafe {
            println!("setup ntt");
            if ff_set_up(
                POWERS_OF_OMEGA_COARSE_LOG_COUNT,
                POWERS_OF_COSET_OMEGA_COARSE_LOG_COUNT,
            ) != 0
            {
                return Err(GpuError::CreateContextErr);
            }

            if ntt_set_up() != 0 {
                return Err(GpuError::CreateContextErr);
            }
        }
        Ok(Self {
            device_id: device_id,
            mem_pool: mem_pool,
            bases: None,
            default_stream: bc_stream::new()?,
            h2d_stream: bc_stream::new()?,
            d2h_stream: bc_stream::new()?,
            exec_stream: bc_stream::new()?,
        })
    }

    pub fn init_for_arithmetic(device_id: usize) -> Result<Self, GpuError> {
        set_device(device_id);
        let mem_pool = bc_mem_pool::new(device_id)?;

        let devices = devices()?;
        for idx in 0..devices {
            if idx == device_id as i32 {
                continue;
            }
            mem_pool_enable_peer_access(mem_pool, idx).unwrap();
            device_enable_peer_access(idx).unwrap();
        }

        let stream = bc_stream::new()?;
        unsafe {
            if ff_set_up(
                POWERS_OF_OMEGA_COARSE_LOG_COUNT,
                POWERS_OF_COSET_OMEGA_COARSE_LOG_COUNT,
            ) != 0
            {
                return Err(GpuError::CreateContextErr);
            }
        }
        Ok(Self {
            device_id: device_id,
            mem_pool: mem_pool,
            bases: None,
            default_stream: stream,
            h2d_stream: bc_stream::new()?,
            d2h_stream: bc_stream::new()?,
            exec_stream: bc_stream::new()?,
        })
    }

    pub fn get_mem_pool(&self) -> bc_mem_pool {
        self.mem_pool
    }

    pub fn get_stream(&self) -> bc_stream {
        self.default_stream
    }
    pub fn get_h2d_stream(&self) -> bc_stream {
        self.h2d_stream
    }

    pub fn get_d2h_stream(&self) -> bc_stream {
        self.d2h_stream
    }

    pub fn get_exec_stream(&self) -> bc_stream {
        self.exec_stream
    }

    pub fn get_bases_ptr_mut(&self) -> *mut c_void {
        self.bases.expect("device bases") as *mut c_void
    }

    pub fn wait_h2d(&self) -> Result<(), GpuError> {
        let h2d_finished = bc_event::new()?;
        h2d_finished.record(self.get_h2d_stream())?;
        self.get_exec_stream().wait(h2d_finished)?;

        Ok(())
    }

    pub fn wait_exec(&self) -> Result<(), GpuError> {
        let exec_finished = bc_event::new()?;
        exec_finished.record(self.get_exec_stream())?;
        self.get_d2h_stream().wait(exec_finished)?;
        Ok(())
    }

    pub fn destroy(&self) -> Result<(), GpuError> {
        unsafe {
            if bc_mem_pool_destroy(self.get_mem_pool()) != 0 {
                return Err(GpuError::DestroyContextErr);
            }
        }

        Ok(())
    }

    pub fn sync(&self) -> Result<(), GpuError> {
        self.get_h2d_stream().sync()?;
        self.get_exec_stream().sync()?;
        self.get_d2h_stream().sync()?;

        Ok(())
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

// impl Drop for GpuContext{
//     fn drop(&mut self) {
//         if unsafe { bc_mem_pool_destroy(self.get_mem_pool())} != 0{
//             panic!("couldn't destroy mempool");
//         }
//         println!("mempool destroyed");
//         if let Some(bases) = self.bases{
//             if unsafe { bc_free(bases as *mut c_void)} != 0{
//                 panic!("couldn't free bases");
//             }
//             println!("d bases freed");
//             if unsafe { msm_tear_down()} != 0{
//                 panic!("couldn't tear down msm");
//             }
//             println!("msm tear down");
//         }

//         if self.has_ntt{
//             if unsafe { ntt_tear_down()} != 0{
//                 panic!("couldn't tear down ntt");
//             }
//             println!("ntt tear down");
//         }
//     }
// }

pub fn devices() -> Result<i32, GpuError> {
    let mut count = 0;
    let success = unsafe { bc_get_device_count(std::ptr::addr_of_mut!(count)) } == 0;
    if success {
        Ok(count)
    } else {
        Err(GpuError::DeviceGetCountErr)
    }
}

pub fn device_info(device_id: i32) -> Result<DeviceMemoryInfo, GpuError> {
    let mut free = 0;
    let mut total = 0;
    let success = unsafe {
        let result = bc_set_device(device_id);
        assert_eq!(result, 0);
        bc_mem_get_info(std::ptr::addr_of_mut!(free), std::ptr::addr_of_mut!(total))
    } == 0;
    if success {
        Ok(DeviceMemoryInfo { free, total })
    } else {
        Err(GpuError::DeviceGetDeviceMemoryInfoErr)
    }
}

pub fn set_device(device_id: usize) -> Result<(), GpuError> {
    let success = unsafe { bc_set_device(device_id as i32) } == 0;
    if success {
        Ok(())
    } else {
        Err(GpuError::SetDeviceErr)
    }
}
