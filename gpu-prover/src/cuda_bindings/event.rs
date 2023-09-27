use super::*;

// TODO check in device_ids needed

// #[derive(Clone)]
pub struct Event {
    pub(crate) sub_events: Mutex<Vec<Vec<(usize, Arc<bc_event>)>>>,
}

impl Clone for Event {
    fn clone(&self) -> Self {
        let inner = self.sub_events.lock().unwrap();
        Self {
            sub_events: Mutex::new(inner.clone()),
        }
    }
}

impl Event {
    pub fn new() -> Self {
        Self {
            sub_events: Mutex::new(vec![]),
        }
    }

    pub fn record(&mut self, stream: &Stream) -> GpuResult<()> {
        let device_id = stream.device_id();
        let stream_id = stream.inner.handle as usize;
        set_device(device_id)?;

        let mut sub_events = self.sub_events.lock().unwrap();

        if sub_events.len() < device_id + 1 {
            sub_events.resize(device_id + 1, vec![]);
        }

        let mut event = bc_event {
            handle: std::ptr::null_mut() as *mut c_void,
        };
        unsafe {
            let result = bc_event_create(addr_of_mut!(event), true, true);
            if result != 0 {
                return Err(GpuError::EventCreateErr(result));
            }
        }
        unsafe {
            let result = bc_event_record(event, stream.inner);
            if result != 0 {
                return Err(GpuError::EventRecordErr(result));
            }
        }

        let event = Arc::new(event);
        let inner = sub_events[device_id]
            .iter_mut()
            .filter(|(id, _)| *id == stream_id)
            .next();
        if let Some(inner) = inner {
            if Arc::strong_count(&inner.1) == 1 {
                unsafe {
                    let result = bc_event_destroy(*inner.1.as_ref());
                    if result != 0 {
                        panic!("EventDestroyErr({}) while droping Event", result);
                    }
                }
            }

            inner.1 = event;
        } else {
            sub_events[device_id].push((stream_id, event));
        }

        Ok(())
    }

    pub fn sync(&self) -> GpuResult<()> {
        let mut sub_events = self.sub_events.lock().unwrap();
        for events in sub_events.iter() {
            for (_, event) in events.iter() {
                unsafe {
                    let result = bc_event_synchronize(*event.as_ref());
                    if result != 0 {
                        return Err(GpuError::EventSyncErr(result));
                    }
                }
            }
        }

        Ok(())
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let mut sub_events = self.sub_events.lock().unwrap();
        for events in sub_events.iter() {
            for (_, event) in events.iter() {
                if Arc::strong_count(event) == 1 {
                    unsafe {
                        let result = bc_event_destroy(*event.as_ref());
                        if result != 0 {
                            panic!("EventDestroyErr({}) while droping Event", result);
                        }
                    }
                }
            }
        }
    }
}
