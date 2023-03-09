use super::*;
use bellman::PrimeField;
use core::ops::Range;
use std::io::{Read, Write};

pub struct AsyncVec<T, #[cfg(feature = "allocator")] A: Allocator = CudaAllocator> {
    #[cfg(feature = "allocator")]
    pub values: Option<Vec<T, A>>,
    #[cfg(not(feature = "allocator"))]
    pub values: Option<Vec<T>>,
    pub(crate) read_event: Event,
    pub(crate) write_event: Event,
}

use std::fmt;

macro_rules! impl_async_vec {
    (impl AsyncVec $inherent:tt) => {
        #[cfg(feature = "allocator")]
        impl<T, A: Allocator + Default> AsyncVec<T, A> $inherent

        #[cfg(not(feature = "allocator"))]
        impl<T> AsyncVec<T> $inherent
    };
}

impl_async_vec! {
    impl AsyncVec{
        pub fn allocate_new(length: usize) -> Self {
            #[cfg(feature = "allocator")]
            let mut values = Vec::with_capacity_in(length, A::default());
            #[cfg(not(feature = "allocator"))]
            let mut values = Vec::with_capacity(length);
            unsafe {
                values.set_len(length);
            }

            Self {
                values: Some(values),
                read_event: Event::new(),
                write_event: Event::new(),
            }
        }

        pub fn get_values(&self) -> GpuResult<&[T]> {
            self.write_event.sync()?;
            Ok(self.values.as_ref().expect("async_vec inner is none"))
        }

        pub fn get_values_mut(&mut self) -> GpuResult<&mut [T]> {
            self.read_event.sync()?;
            self.write_event.sync()?;
            Ok(self.values.as_mut().expect("async_vec inner is none"))
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

            ctx.h2d_stream.wait(self.write_event())?;
            ctx.h2d_stream.wait(other.read_event())?;
            ctx.h2d_stream.wait(other.write_event())?;

            let result = unsafe {
                bc_memcpy_async(
                    other.as_mut_ptr(other_range) as *mut c_void,
                    self.as_ptr(this_range) as *const c_void,
                    length as u64,
                    ctx.h2d_stream().inner,
                )
            };

            if result != 0 {
                return Err(GpuError::AsyncH2DErr(result));
            }

            self.read_event.record(ctx.h2d_stream())?;
            other.write_event.record(ctx.h2d_stream())?;

            Ok(())
        }

        pub fn async_copy_from_device(
            &mut self,
            ctx: &mut GpuContext,
            other: &mut DeviceBuf<T>,
            this_range: Range<usize>,
            other_range: Range<usize>,
        ) -> GpuResult<()> {
            assert_eq!(this_range.len(), other_range.len());
            let length = std::mem::size_of::<T>() * this_range.len();
            set_device(ctx.device_id())?;

            ctx.d2h_stream.wait(self.write_event())?;
            ctx.d2h_stream.wait(self.read_event())?;
            ctx.d2h_stream.wait(other.write_event())?;

            let result = unsafe {
                bc_memcpy_async(
                    self.as_mut_ptr(this_range) as *mut c_void,
                    other.as_ptr(other_range) as *const c_void,
                    length as u64,
                    ctx.d2h_stream().inner,
                )
            };

            if result != 0 {
                return Err(GpuError::AsyncH2DErr(result));
            }

            self.write_event.record(ctx.d2h_stream())?;
            other.read_event.record(ctx.d2h_stream())?;

            Ok(())
        }

        pub fn len(&self) -> usize {
            self.values.as_ref().expect("async_vec inner is none").len()
        }
        #[cfg(feature = "allocator")]
        pub fn into_inner(mut self) -> GpuResult<std::vec::Vec<T, A>> {
            self.read_event.sync()?;
            self.write_event.sync()?;

            Ok(self.values.take().expect("async_vec inner is none"))
        }

        #[cfg(not(feature = "allocator"))]
        pub fn into_inner(mut self) -> GpuResult<std::vec::Vec<T>> {
            self.read_event.sync()?;
            self.write_event.sync()?;

            Ok(self.values.take().expect("async_vec inner is none"))
        }

        pub fn read_event(&self) -> &Event {
            &self.read_event
        }

        pub fn write_event(&self) -> &Event {
            &self.write_event
        }

        pub fn as_ptr(&self, range: Range<usize>) -> *const T {
            self.values.as_ref().expect("async_vec inner is none")[range].as_ptr()
        }

        pub fn as_mut_ptr(&mut self, range: Range<usize>) -> *mut T {
            self.values.as_mut().expect("async_vec inner is none")[range].as_mut_ptr()
        }

        pub fn zeroize(&mut self){
            let unit_len = std::mem::size_of::<T>();
            let total_len = unit_len * self.len();
            let dst = self.as_mut_ptr(0..self.len()) as *mut u8;
            unsafe{std::ptr::write_bytes(dst, 0, total_len)};
        }
    }
}

#[cfg(feature = "allocator")]
impl<T: fmt::Debug, A: Allocator + Default> fmt::Debug for AsyncVec<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncVec")
            .field("Values", &self.get_values().unwrap())
            .finish()
    }
}
#[cfg(not(feature = "allocator"))]
impl<T: fmt::Debug> fmt::Debug for AsyncVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncVec")
            .field("Values", &self.get_values().unwrap())
            .finish()
    }
}

#[cfg(feature = "allocator")]
impl<T, A: Allocator> From<Vec<T, A>> for AsyncVec<T, A> {
    fn from(values: Vec<T, A>) -> Self {
        Self {
            values: Some(values),
            read_event: Event::new(),
            write_event: Event::new(),
        }
    }
}
#[cfg(not(feature = "allocator"))]
impl<T> From<Vec<T>> for AsyncVec<T> {
    fn from(values: Vec<T>) -> Self {
        Self {
            values: Some(values),
            read_event: Event::new(),
            write_event: Event::new(),
        }
    }
}

#[cfg(feature = "allocator")]
impl<T, A: Allocator + Default> From<AsyncVec<T, A>> for Vec<T, A> {
    fn from(vector: AsyncVec<T, A>) -> Self {
        vector.into_inner().unwrap()
    }
}

#[cfg(not(feature = "allocator"))]
impl<T> From<AsyncVec<T>> for Vec<T> {
    fn from(vector: AsyncVec<T>) -> Self {
        vector.into_inner().unwrap()
    }
}

#[cfg(feature = "allocator")]
impl<T, A: Allocator> Drop for AsyncVec<T, A> {
    fn drop(&mut self) {
        self.read_event.sync().unwrap();
        self.write_event.sync().unwrap();
    }
}

#[cfg(not(feature = "allocator"))]
impl<T> Drop for AsyncVec<T> {
    fn drop(&mut self) {
        self.read_event.sync().unwrap();
        self.write_event.sync().unwrap();
    }
}

macro_rules! impl_async_vec_for_field {
    (impl AsyncVec $inherent:tt) => {
        #[cfg(feature = "allocator")]
        impl<F: PrimeField, A: Allocator + Default> AsyncVec<F, A> $inherent

        #[cfg(not(feature = "allocator"))]
        impl<F: PrimeField> AsyncVec<F> $inherent
    };
}

impl_async_vec_for_field! {
    impl AsyncVec{
        pub fn write<W: Write>(&self, writer: &mut W) -> GpuResult<()> {
            let length = self.len();
            let F_SIZE = F::zero().into_raw_repr().as_ref().len() * 8;

            let mut poly_bytes: Vec<u8> = Vec::with_capacity(F_SIZE * length);
            unsafe {
                poly_bytes.set_len(F_SIZE * length);
            }

            self.to_bytes(&mut poly_bytes[..])?;
            writer
                .write_all(&poly_bytes[..])
                .expect("Can't write AsyncVec");

            Ok(())
        }

        pub fn to_bytes(&self, dst: &mut [u8]) -> GpuResult<()> {
            let length = self.len();
            let F_SIZE = F::zero().into_raw_repr().as_ref().len() * 8;
            assert_eq!(length * F_SIZE, dst.len(), "Wrong destination length");
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.as_ptr(0..self.len()) as *const u8,
                    dst.as_mut_ptr(),
                    self.len() * FIELD_ELEMENT_LEN,
                )
            };

            Ok(())
        }

        pub fn read<R: Read>(&mut self, reader: &mut R) -> GpuResult<()> {
            let length = self.len();
            let F_SIZE = F::zero().into_raw_repr().as_ref().len() * 8;

            let mut res_bytes: Vec<u8> = Vec::with_capacity(F_SIZE * length);
            unsafe {
                res_bytes.set_len(F_SIZE * length);
            }

            reader
                .read_exact(&mut res_bytes)
                .expect("Can't read AsyncVec");

            self.from_bytes(&res_bytes[..])
        }

        pub fn from_bytes(&mut self, src: &[u8]) -> GpuResult<()> {
            let length = self.len();
            let F_SIZE = F::zero().into_raw_repr().as_ref().len() * 8;
            assert_eq!(length * F_SIZE, src.len(), "Wrong source length");

            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    self.as_ptr(0..self.len()) as *mut u8,
                    self.len() * FIELD_ELEMENT_LEN,
                )
            };

            Ok(())
        }
    }
}
