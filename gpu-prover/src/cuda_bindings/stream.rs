use super::*;

pub struct Stream {
    pub(crate) inner: bc_stream,
    pub(crate) device_id: usize,
}

impl Stream {
    pub fn new(device_id: usize) -> GpuResult<Self> {
        set_device(device_id)?;

        let mut inner = bc_stream {
            handle: std::ptr::null_mut() as *mut c_void,
        };
        unsafe{
            let result = bc_stream_create(addr_of_mut!(inner), true);
            if result != 0 {
                return Err(GpuError::StremCreateErr(result));
            };
        }

        Ok(Self{inner, device_id})
    }

    pub fn wait(&mut self, event: &Event) -> GpuResult<()> {
        set_device(self.device_id)?;

        let mut sub_events = event.sub_events.lock().unwrap();

        for events in sub_events.iter() {
            for (_, event) in events.iter() {
                unsafe{
                    let result = bc_stream_wait_event(self.inner, *event.as_ref());
                    if result != 0 {
                        return Err(GpuError::StreamWaitEventErr(result));
                    }
                }
            }
        }

        Ok(())
    }

    pub fn sync(&self) -> GpuResult<()> {
        set_device(self.device_id)?;
        unsafe{
            let result = bc_stream_synchronize(self.inner);
            if result != 0 {
                return Err(GpuError::StreamSyncErr(result));
            }
        }

        Ok(())
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        set_device(self.device_id).expect("during Stream dropping");
        unsafe{
            let result = bc_stream_destroy(self.inner);
            if result != 0 {
                println!("StreamDestroyErr({})", result);
            }
        }
    }
}