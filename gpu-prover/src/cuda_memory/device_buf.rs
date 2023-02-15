use super::*;

#[derive(Debug, Clone)]
pub struct DeviceBuf<T> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
}
unsafe impl<T> Send for DeviceBuf<T> {}
unsafe impl<T> Sync for DeviceBuf<T> {}

impl<T> DeviceBuf<T> {
    pub(crate) fn new(ctx: Arc<GpuContext>, count: usize) -> Result<Self, GpuError> {
        let len = std::mem::size_of::<T>() * count;
        let mut ptr = std::ptr::null_mut();
        malloc_from_pool_async(
            addr_of_mut!(ptr),
            len,
            ctx.get_mem_pool(),
            ctx.get_h2d_stream(),
        )?;

        Ok(Self {
            ptr: ptr as *mut T,
            len: count,
        })
    }

    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    pub(crate) fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }
    
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn h2d(&self, ctx: Arc<GpuContext>, h_ptr: *const T) -> Result<(), GpuError> {
        let len = std::mem::size_of::<T>() * self.len;
        memcpy_async(
            self.ptr as *mut c_void,
            h_ptr as *const c_void,
            len,
            ctx.get_h2d_stream(),
        )?;

        Ok(())
    }

    pub(crate) fn d2h(&self, ctx: Arc<GpuContext>, h_ptr: *mut T) -> Result<(), GpuError> {
        let len = std::mem::size_of::<T>() * self.len;
        memcpy_async(
            h_ptr as *mut c_void,
            self.ptr as *const c_void,
            len,
            ctx.get_d2h_stream(),
        )?;

        Ok(())
    }

    pub(crate) fn d2d(&self, ctx: Arc<GpuContext>, other: &mut Arc<DeviceBuf<T>>) -> Result<(), GpuError> {
        assert_eq!(self.len, other.len);

        let len = std::mem::size_of::<T>() * self.len;
        memcpy_async(
            other.ptr as *mut c_void,
            self.ptr as *const c_void,
            len,
            ctx.get_exec_stream(),
        )?;

        Ok(())
    }

    pub(crate) fn free(&self, ctx: Arc<GpuContext>) -> Result<(), GpuError> {
        free_async(self.as_mut_ptr() as *mut c_void, ctx.get_d2h_stream())?;
        Ok(())
    }
}
