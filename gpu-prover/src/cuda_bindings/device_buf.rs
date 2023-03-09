use super::*;
use core::ops::Range;

pub struct DeviceBuf<T> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    pub(crate) device_id: usize,

    pub(crate) is_static_mem: bool,
    pub(crate) is_freed: bool,

    pub(crate) read_event: Event,
    pub(crate) write_event: Event,
}

impl<T> DeviceBuf<T> {
    pub fn alloc_static(ctx: &GpuContext, len: usize) -> GpuResult<Self> {
        set_device(ctx.device_id())?;
        assert!(ctx.mem_pool.is_none(), "mem pool is allocated");

        let byte_len = std::mem::size_of::<T>() * len;
        let mut ptr = std::ptr::null_mut();

        unsafe {
            let result = bc_malloc(addr_of_mut!(ptr), byte_len as size_t);
            if result != 0 {
                return Err(GpuError::MallocErr(result));
            }
        }
        Ok(Self {
            ptr: ptr as *mut T,
            len: len,
            device_id: ctx.device_id(),

            is_static_mem: true,
            is_freed: false,

            read_event: Event::new(),
            write_event: Event::new(),
        })
    }

    pub fn async_alloc_in_h2d(ctx: &GpuContext, len: usize) -> GpuResult<Self> {
        set_device(ctx.device_id())?;
        let mem_pool = ctx.mem_pool.expect("mem pool is not allocated");

        let byte_len = std::mem::size_of::<T>() * len;
        let mut ptr = std::ptr::null_mut();

        unsafe {
            let result = bc_malloc_from_pool_async(
                addr_of_mut!(ptr),
                byte_len as size_t,
                mem_pool,
                ctx.h2d_stream().inner,
            );
            if result != 0 {
                return Err(GpuError::AsyncPoolMallocErr(result));
            }
        }

        Ok(Self {
            ptr: ptr as *mut T,
            len: len,
            device_id: ctx.device_id(),

            is_static_mem: false,
            is_freed: false,

            read_event: Event::new(),
            write_event: Event::new(),
        })
    }

    pub fn async_alloc_in_exec(ctx: &GpuContext, len: usize) -> GpuResult<Self> {
        set_device(ctx.device_id())?;
        let mem_pool = ctx.mem_pool.expect("mem pool is not allocated");

        let byte_len = std::mem::size_of::<T>() * len;
        let mut ptr = std::ptr::null_mut();

        unsafe {
            let result = bc_malloc_from_pool_async(
                addr_of_mut!(ptr),
                byte_len as size_t,
                mem_pool,
                ctx.exec_stream().inner,
            );
            if result != 0 {
                return Err(GpuError::AsyncPoolMallocErr(result));
            }
        }

        Ok(Self {
            ptr: ptr as *mut T,
            len: len,
            device_id: ctx.device_id(),

            is_static_mem: false,
            is_freed: false,

            read_event: Event::new(),
            write_event: Event::new(),
        })
    }

    pub(crate) fn split(self, num: usize) -> Vec<Self> {
        assert!(!self.is_freed, "device buf is already freed");
        self.split_any_buf(num)
    }

    pub(crate) fn split_any_buf(mut self, num: usize) -> Vec<Self> {
        assert!(
            self.len % num == 0,
            "device buf of length {} could not be splited into {} parts",
            self.len,
            num
        );

        let mut res_chunks = vec![];
        for i in 0..num {
            let chunk_len = self.len / num;
            // SAFETY: Pointers are correct because of the length of original DeviceBuf
            let new_ptr = unsafe { self.ptr.add(i * chunk_len) };
            res_chunks.push(Self {
                ptr: new_ptr,
                len: chunk_len,
                device_id: self.device_id,

                is_static_mem: self.is_static_mem,
                is_freed: true,

                read_event: self.read_event.clone(),
                write_event: self.write_event.clone(),
            });
        }

        self.is_freed = true;
        res_chunks
    }

    pub fn async_copy_to_host(
        &mut self,
        ctx: &mut GpuContext,
        other: &mut AsyncVec<T>,
        this_range: Range<usize>,
        other_range: Range<usize>,
    ) -> GpuResult<()> {
        other.async_copy_from_device(ctx, self, other_range, this_range)
    }

    pub fn async_copy_from_host(
        &mut self,
        ctx: &mut GpuContext,
        other: &mut AsyncVec<T>,
        this_range: Range<usize>,
        other_range: Range<usize>,
    ) -> GpuResult<()> {
        other.async_copy_to_device(ctx, self, other_range, this_range)
    }

    pub fn async_copy_to_device(
        &mut self,
        ctx: &mut GpuContext,
        other: &mut DeviceBuf<T>,
        this_range: Range<usize>,
        other_range: Range<usize>,
    ) -> GpuResult<()> {
        assert_eq!(this_range.len(), other_range.len());
        let length = std::mem::size_of::<T>() * this_range.len();
        set_device(ctx.device_id())?;

        ctx.exec_stream.wait(self.write_event())?;
        ctx.exec_stream.wait(other.read_event())?;
        ctx.exec_stream.wait(other.write_event())?;

        let result = unsafe {
            bc_memcpy_async(
                other.as_mut_ptr(other_range) as *mut c_void,
                self.as_ptr(this_range) as *const c_void,
                length as u64,
                ctx.exec_stream().inner,
            )
        };

        if result != 0 {
            return Err(GpuError::AsyncH2DErr(result));
        }

        self.read_event.record(ctx.exec_stream())?;
        other.write_event.record(ctx.exec_stream())?;

        Ok(())
    }

    pub unsafe fn async_copy_from_pointer_and_len(
        &mut self,
        ctx: &mut GpuContext,
        other: *const T,
        this_range: Range<usize>,
        other_len: usize,
    ) -> GpuResult<()> {
        assert_eq!(this_range.len(), other_len);
        assert!(other_len > 0);
        let length = std::mem::size_of::<T>() * this_range.len();
        set_device(ctx.device_id())?;

        ctx.h2d_stream.wait(self.read_event())?;
        ctx.h2d_stream.wait(self.write_event())?;

        let result = bc_memcpy_async(
            self.as_mut_ptr(this_range) as *mut c_void,
            other as *const c_void,
            length as u64,
            ctx.h2d_stream().inner,
        );

        if result != 0 {
            return Err(GpuError::AsyncMemcopyErr(result));
        }

        self.write_event.record(ctx.h2d_stream())?;

        Ok(())
    }

    pub fn async_copy_from_device(
        &mut self,
        ctx: &mut GpuContext,
        other: &mut DeviceBuf<T>,
        this_range: Range<usize>,
        other_range: Range<usize>,
    ) -> GpuResult<()> {
        other.async_copy_to_device(ctx, self, other_range, this_range)
    }

    pub fn async_free(&mut self, stream: &mut Stream) -> GpuResult<()> {
        set_device(stream.device_id())?;
        assert!(
            !self.is_static_mem,
            "you can free static memory only with droping DeviceBuf"
        );
        assert!(!self.is_freed, "memory is already free");

        stream.wait(&self.read_event)?;
        stream.wait(&self.write_event)?;

        unsafe {
            let result = bc_free_async(self.ptr as *mut c_void, stream.inner);
            if result != 0 {
                return Err(GpuError::AsyncMemFreeErr(result));
            }
        }
        self.is_freed = true;

        Ok(())
    }

    pub fn read_event(&self) -> &Event {
        &self.read_event
    }

    pub fn write_event(&self) -> &Event {
        &self.write_event
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_ptr(&self, range: Range<usize>) -> *const T {
        assert!(
            range.end <= self.len,
            "the range is invalid for this device_buf"
        );

        unsafe { self.ptr.add(range.start) as *const T }
    }

    pub fn as_mut_ptr(&mut self, range: Range<usize>) -> *mut T {
        assert!(
            range.end <= self.len,
            "the range is invalid for this device_buf"
        );

        unsafe { self.ptr.add(range.start) }
    }
}

impl<T> Drop for DeviceBuf<T> {
    fn drop(&mut self) {
        if !self.is_freed {
            self.read_event.sync().unwrap();
            self.write_event.sync().unwrap();

            if unsafe { bc_free(self.ptr as *mut c_void) } != 0 {
                panic!("Can't free memory of DeviceBuf");
            }
        }
    }
}
