//! This file defines the array of curs and the structures
//! associated with the attributes that are members of the array.

use crate::compare::{
    impl_equal_float, impl_equal_int, impl_grater_equal_float, impl_grater_equal_int,
    impl_grater_float, impl_grater_int, impl_less_equal_float, impl_less_equal_int,
    impl_less_float, impl_less_int, impl_negative_equal_float, impl_negative_equal_int,
};
use crate::ffi;
use crate::CursState;
use crate::{dim::Dim, dtype::DataType, Num, Order};

use std::clone::Clone;

/// Multi-dimensional array on CUDA device.
/// # Parameters
/// * data_ptr <dr>
/// pointer to CUDA device
/// * dim <dr>
/// dimention
/// Dimention of Array
/// * Order <dr>
/// Array is Column priority or Row priority
/// * dtype <dr>
/// Data type of Array
#[derive(Debug)]
pub struct Array<'a, T: Num> {
    pub data_ptr: *mut T,
    pub dim: Dim,
    pub order: Order,
    pub dtype: DataType,
    pub state: &'a CursState,
}

impl<'a, T: Num> Drop for Array<'a, T> {
    fn drop(&mut self) {
        let cuda_error = ffi::free(self.data_ptr as *mut T);
        match cuda_error {
            Ok(_) => {}
            _ => {
                panic!("{:?} Can't Free CUDA Array", cuda_error);
            }
        };
    }
}

/// Definieren eines Arrays auf dem Grafikprozessor
fn malloc_array_on_device<'a, T: Num, D: AsRef<Dim>>(
    dim: &D,
    state: &'a CursState,
) -> ffi::Result<Array<'a, T>> {
    let dim = dim.as_ref();
    let size = dim.size();
    let n_bytes = size * std::mem::size_of::<T>();
    let order = Order::default();
    let dtype = T::dtype();

    let device_ptr = ffi::malloc(n_bytes)?;

    Ok(Array {
        data_ptr: device_ptr,
        dim: dim.clone(),
        order: order,
        dtype: dtype,
        state: state,
    })
}

/// Fill with the value specified for the device's pointer
pub fn fill<T: Num>(data_ptr: *mut T, num: T, size: usize) -> ffi::Result<()> {
    let mut vec: Vec<T> = Vec::with_capacity(size);
    for _ in 0..size {
        vec.push(num);
    }

    ffi::memcpy(
        data_ptr,
        vec.as_ptr(),
        size * std::mem::size_of::<T>(),
        cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
    )
}

impl<'a, T: Num> Array<'a, T> {
    /// Initialize an Array of the given size with zero fill.
    pub fn zeros<D: AsRef<Dim>>(dim: &D, state: &'a CursState) -> ffi::Result<Array<'a, T>> {
        let dim = dim.as_ref();
        let array = malloc_array_on_device(&dim, state)?;
        fill(array.data_ptr, T::zero(), dim.size())?;

        Ok(array)
    }

    /// Initializes an Array of the given size with 1 fill
    pub fn ones<D: AsRef<Dim>>(dim: &D, state: &'a CursState) -> ffi::Result<Array<'a, T>> {
        let dim = dim.as_ref();
        let array = malloc_array_on_device(&dim, state)?;
        fill(array.data_ptr, T::one(), dim.size())?;

        Ok(array)
    }

    /// Fill an Array with the given values
    pub fn fill(&self, num: T) -> ffi::Result<()> {
        fill(self.data_ptr, num, self.dim.size())?;
        Ok(())
    }

    /// Defines an Array filled with the specified values
    pub fn full<D: AsRef<Dim>>(dim: &D, num: T, state: &'a CursState) -> ffi::Result<Array<'a, T>> {
        let dim = dim.as_ref();
        let array = malloc_array_on_device(&dim, state)?;
        fill(array.data_ptr, num, array.dim.size())?;
        Ok(array)
    }

    /// Converting from Vec to Array
    pub fn from_vec<D: AsRef<Dim>>(
        vec: Vec<T>,
        dim: &D,
        state: &'a CursState,
    ) -> ffi::Result<Array<'a, T>> {
        let dim = dim.as_ref();
        if vec.len() != dim.size() {
            panic!("input vec shape and input dimention size is not same");
        }
        let array = malloc_array_on_device(&dim, state)?;
        ffi::memcpy(
            array.data_ptr as *const T as *mut T,
            vec.as_ptr() as *const T,
            array.size() * std::mem::size_of::<T>(),
            cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
        )?;
        Ok(array)
    }

    /// return shape (Dim)
    pub fn shape(&self) -> Dim {
        self.dim.clone()
    }

    /// return number of elements
    pub fn size(&self) -> usize {
        self.dim.size()
    }

    /// CUDA array to Slice
    pub fn as_slice(&self) -> ffi::Result<&[T]> {
        let vec: Vec<T> = self.as_vec()?;

        Ok(unsafe { &*(vec.as_slice() as *const [T]) })
    }

    /// CUDA array to Vec
    pub fn as_vec(&self) -> ffi::Result<Vec<T>> {
        let size = self.size();
        let mut vec: Vec<T> = Vec::with_capacity(size);
        for _ in 0..size {
            vec.push(T::zero());
        }

        ffi::memcpy(
            vec.as_ptr() as *mut T,
            self.data_ptr,
            size * std::mem::size_of::<T>(),
            cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        )?;

        Ok(vec)
    }

    /// Comparing Arrays operator like ==
    pub fn eq(&self, other: &Self) -> ffi::Result<Array<T>> {
        let res = match self.dtype {
            DataType::INT16 => impl_equal_int(self, other, self.state),
            DataType::FLOAT => impl_equal_float(self, other, self.state),
            _ => todo!(),
        }?;
        Ok(res)
    }

    /// Comparing Arrays operator like !=
    pub fn neq(&self, other: &Self) -> ffi::Result<Array<T>> {
        let res = match self.dtype {
            DataType::INT16 => impl_negative_equal_int(self, other, self.state),
            DataType::FLOAT => impl_negative_equal_float(self, other, self.state),
            _ => todo!(),
        }?;
        Ok(res)
    }

    /// Comparing Arrays operator like >
    pub fn greater(&self, other: &Self) -> ffi::Result<Array<T>> {
        let res = match self.dtype {
            DataType::INT16 => impl_grater_int(self, other, self.state),
            DataType::FLOAT => impl_grater_float(self, other, self.state),
            _ => todo!(),
        }?;
        Ok(res)
    }

    /// Comparing Arrays operator like >=
    pub fn greater_equal(&self, other: &Self) -> ffi::Result<Array<T>> {
        let res = match self.dtype {
            DataType::INT16 => impl_grater_equal_int(self, other, self.state),
            DataType::FLOAT => impl_grater_equal_float(self, other, self.state),
            _ => todo!(),
        }?;
        Ok(res)
    }

    /// Comparing Arrays operator like <
    pub fn less(&self, other: &Self) -> ffi::Result<Array<T>> {
        let res = match self.dtype {
            DataType::INT16 => impl_less_int(self, other, self.state),
            DataType::FLOAT => impl_less_float(self, other, self.state),
            _ => todo!(),
        }?;
        Ok(res)
    }

    /// Comparing Arrays operator like <=
    pub fn less_equal(&self, other: &Self) -> ffi::Result<Array<T>> {
        let res = match self.dtype {
            DataType::INT16 => impl_less_equal_int(self, other, self.state),
            DataType::FLOAT => impl_less_equal_float(self, other, self.state),
            _ => todo!(),
        }?;
        Ok(res)
    }
}

impl<'a, T: Num> Clone for Array<'a, T> {
    fn clone(&self) -> Self {
        let size = self.dim.size() * std::mem::size_of::<T>();
        let array = malloc_array_on_device(&self.dim, self.state).unwrap();
        ffi::memcpy(
            array.data_ptr,
            self.data_ptr,
            size,
            cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        )
        .unwrap();

        array
    }

    fn clone_from(&mut self, source: &Self) {
        *self = source.clone();
    }
}

#[test]
fn test_clone() {
    let status = CursState::new(0);

    let array_a = Array::<f32>::ones(&vec![10, 10], &status).unwrap();
    let array_b = array_a.clone();

    let comp_res = array_a.eq(&array_b).unwrap();

    let mut ans_vec = Vec::<f32>::with_capacity(100);
    for _ in 0..100 {
        ans_vec.push(1f32);
    }
    assert_eq!(comp_res.as_vec().unwrap(), ans_vec);
}
