use super::*;
use crate::{Crs, CrsForMonomialForm};
use std::marker::PhantomData;

static DEVICES_IN_USE: Mutex<Vec<Option<()>>> = Mutex::new(vec![]);

pub struct GpuContext {
    device_id: usize,
    pub(crate) mem_pool: Option<bc_mem_pool>,
    pub(crate) bases: Option<*const c_void>, // addr of bases on gpu device
    pub(crate) bases_len: usize,

    pub(crate) ff: bool,
    pub(crate) ntt: bool,
    pub(crate) pn: bool,

    pub(crate) h2d_stream: Stream,
    pub(crate) d2h_stream: Stream,
    pub(crate) exec_stream: Stream,
    affinity: Vec<usize>,
}

const POWERS_OF_OMEGA_COARSE_LOG_COUNT: u32 = 25;
const POWERS_OF_COSET_OMEGA_COARSE_LOG_COUNT: u32 = 14;

impl GpuContext {
    pub fn new_full(device_id: usize, bases: &[CompactG1Affine]) -> GpuResult<Self> {
        let mut ctx = Self::new(device_id)?;

        ctx.set_up_ff()?;
        ctx.set_up_ntt()?;
        ctx.set_up_pn()?;
        ctx.set_up_msm(bases)?;
        ctx.set_up_mem_pool()?;

        Ok(ctx)
    }

    pub fn new(device_id: usize) -> GpuResult<Self> {
        assert!(device_id < devices()? as usize);
        set_device(device_id)?;
        let devices = devices()? as usize;

        let mut devices_in_use = DEVICES_IN_USE.lock().unwrap();
        if devices_in_use.len() == 0 {
            devices_in_use.resize(devices, None);
        }
        if devices_in_use[device_id].is_some() {
            return Err(GpuError::DeviceInUseErr(device_id));
        } else {
            devices_in_use[device_id] = Some(());
        }

        for idx in 0..devices {
            if idx == device_id {
                continue;
            }
            unsafe {
                let result = bc_device_enable_peer_access(idx as i32);
                if result != 0 {
                    return Err(GpuError::DevicePeerAccessErr(result));
                }
            }
        }

        let affinity = (0..devices as usize).collect();

        Ok(Self {
            device_id: device_id,
            mem_pool: None,
            bases: None,
            bases_len: 0,
            ff: false,
            ntt: false,
            pn: false,
            h2d_stream: Stream::new(device_id)?,
            d2h_stream: Stream::new(device_id)?,
            exec_stream: Stream::new(device_id)?,
            affinity,
        })
    }

    pub fn new_with_affinity(device_id: usize, affinity_devices: &[usize]) -> GpuResult<Self> {
        assert!(device_id < devices()? as usize);
        set_device(device_id)?;
        let devices = devices()? as usize;

        let mut devices_in_use = DEVICES_IN_USE.lock().unwrap();
        if devices_in_use.len() == 0 {
            devices_in_use.resize(devices, None);
        }
        if devices_in_use[device_id].is_some() {
            return Err(GpuError::DeviceInUseErr(device_id));
        } else {
            devices_in_use[device_id] = Some(());
        }

        for idx in affinity_devices.iter() {
            if *idx == device_id {
                continue;
            }
            unsafe {
                let result = bc_device_enable_peer_access(*idx as i32);
                if result != 0 {
                    return Err(GpuError::DevicePeerAccessErr(result));
                }
            }
        }

        Ok(Self {
            device_id: device_id,
            mem_pool: None,
            bases: None,
            bases_len: 0,
            ff: false,
            ntt: false,
            pn: false,
            h2d_stream: Stream::new(device_id)?,
            d2h_stream: Stream::new(device_id)?,
            exec_stream: Stream::new(device_id)?,
            affinity: affinity_devices.to_vec(),
        })
    }

    pub fn set_up_msm(&mut self, bases: &[CompactG1Affine]) -> GpuResult<()> {
        assert!(self.mem_pool.is_none(), "Can't set up msm with mem pool");
        set_device(self.device_id)?;

        self.bases_len = bases.len();

        let bases = transmute_values(bases.as_ref().as_ref());

        let len = bases.len() as u64;
        let mut d_bases_ptr = std::ptr::null_mut();

        unsafe {
            let result = unsafe { bc_malloc(std::ptr::addr_of_mut!(d_bases_ptr), len) };
            if result != 0 {
                return Err(GpuError::CreateContextErr(result));
            }
            let result = bc_memcpy(d_bases_ptr, bases.as_ptr() as *const c_void, len);
            if result != 0 {
                return Err(GpuError::CreateContextErr(result));
            }

            self.bases = Some(d_bases_ptr);
            let result = msm_set_up();
            if result != 0 {
                return Err(GpuError::CreateContextErr(result));
            }
        }

        Ok(())
    }

    pub fn set_up_ff(&mut self) -> GpuResult<()> {
        assert!(!self.ntt, "Can't set up ff with ntt");
        assert!(!self.pn, "Can't set up ff with pn");
        assert!(!self.ff, "ff is already set up");

        set_device(self.device_id)?;
        unsafe {
            let result = ff_set_up(
                POWERS_OF_OMEGA_COARSE_LOG_COUNT,
                POWERS_OF_COSET_OMEGA_COARSE_LOG_COUNT,
            );
            if result != 0 {
                return Err(GpuError::CreateContextErr(result));
            }
        }

        self.ff = true;

        Ok(())
    }

    pub fn set_up_pn(&mut self) -> GpuResult<()> {
        assert!(self.ff, "Can't set up permutations without ff");
        assert!(!self.pn, "pn is already set up");

        set_device(self.device_id)?;

        unsafe {
            let result = pn_set_up();
            if result != 0 {
                return Err(GpuError::PermutationSetupErr(result));
            }
        }

        self.pn = true;

        Ok(())
    }

    pub fn set_up_ntt(&mut self) -> GpuResult<()> {
        assert!(self.ff, "Can't set up ntt without ff");
        assert!(!self.ntt, "ntt is already set up");

        set_device(self.device_id)?;

        unsafe {
            let result = ntt_set_up();
            if result != 0 {
                return Err(GpuError::CreateContextErr(result));
            }
        }

        self.ntt = true;

        Ok(())
    }

    pub fn set_up_mem_pool(&mut self) -> GpuResult<()> {
        assert!(self.mem_pool.is_none(), "mem_pool is already set up");

        set_device(self.device_id)?;

        let mut mem_pool = bc_mem_pool {
            handle: std::ptr::null_mut() as *mut c_void,
        };

        unsafe {
            let result = bc_mem_pool_create(addr_of_mut!(mem_pool), self.device_id as i32);
            if result != 0 {
                return Err(GpuError::MemPoolCreateErr(result));
            }
        }

        let devices = devices()?;
        for idx in 0..devices {
            if idx == self.device_id as i32 {
                continue;
            }
            unsafe {
                let result = bc_mem_pool_enable_peer_access(mem_pool, idx);
                if result != 0 {
                    return Err(GpuError::MemPoolPeerAccessErr(result));
                }
            }
        }

        self.mem_pool = Some(mem_pool);

        Ok(())
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn h2d_stream(&self) -> &Stream {
        &self.h2d_stream
    }

    pub fn h2d_stream_mut(&mut self) -> &mut Stream {
        &mut self.h2d_stream
    }

    pub fn d2h_stream(&self) -> &Stream {
        &self.d2h_stream
    }

    pub fn d2h_stream_mut(&mut self) -> &mut Stream {
        &mut self.d2h_stream
    }

    pub fn exec_stream(&self) -> &Stream {
        &self.exec_stream
    }

    pub fn exec_stream_mut(&mut self) -> &mut Stream {
        &mut self.exec_stream
    }

    pub fn sync(&self) -> GpuResult<()> {
        self.h2d_stream().sync()?;
        self.exec_stream().sync()?;
        self.d2h_stream().sync()?;

        Ok(())
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        self.sync().unwrap();

        if let Some(mem_pool) = self.mem_pool {
            if unsafe { bc_mem_pool_destroy(mem_pool) } != 0 {
                panic!("couldn't destroy mempool");
            }
        }

        if let Some(bases) = self.bases {
            if unsafe { bc_free(bases as *mut c_void) } != 0 {
                panic!("couldn't free bases");
            }
            if unsafe { msm_tear_down() } != 0 {
                panic!("couldn't tear down msm");
            }
        }

        if self.pn {
            if unsafe { pn_tear_down() } != 0 {
                panic!("couldn't tear down permutations");
            }
        }

        if self.ntt {
            if unsafe { ntt_tear_down() } != 0 {
                panic!("couldn't tear down ntt");
            }
        }

        if self.ff {
            if unsafe { ff_tear_down() } != 0 {
                panic!("couldn't tear down ff");
            }
        }

        assert!(self.device_id < devices().unwrap() as usize);
        set_device(self.device_id).unwrap();

        for idx in self.affinity.iter() {
            if *idx == self.device_id {
                continue;
            }

            if unsafe { bc_device_disable_peer_access(*idx as i32) } != 0 {
                panic!("couldn't disable device peer access");
            }
        }

        let mut devices_in_use = DEVICES_IN_USE.lock().unwrap();
        assert!(
            devices_in_use[self.device_id].is_some(),
            "Device usage should be marked"
        );
        devices_in_use[self.device_id].take();
    }
}

pub fn transmute_values<'a, U>(values: &'a [U]) -> &'a [u8] {
    let ptr = values.as_ptr();
    let len = values.len();

    assert!(
        (ptr as usize) % std::mem::align_of::<u8>() == 0,
        "trying to cast with mismatched layout"
    );

    let size = std::mem::size_of::<U>() * len;

    let out: &'a [u8] = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };

    out
}
