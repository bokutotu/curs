//! This file defines the array of curs and the structures
//! associated with the attributes that are members of the array.
pub mod compare;
pub mod new_array;
pub mod shape;
pub mod transpose;

use crate::array::new_array::malloc_array_on_device;
use crate::cuda_runtime;
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
        let cuda_error = cuda_runtime::free(self.data_ptr as *mut T);
        match cuda_error {
            Ok(_) => {}
            _ => {
                panic!("{:?} Can't Free CUDA Array", cuda_error);
            }
        };
    }
}

impl<'a, T: Num> Array<'a, T> {
    /// CUDA ARRAY TO SLICE
    pub fn as_slice(&self) -> cuda_runtime::Result<&[T]> {
        let vec: Vec<T> = self.as_vec()?;

        Ok(unsafe { &*(vec.as_slice() as *const [T]) })
    }

    /// cuda array to vec
    pub fn as_vec(&self) -> cuda_runtime::Result<Vec<T>> {
        let size = self.size();
        let mut vec: Vec<T> = Vec::with_capacity(size);
        for _ in 0..size {
            vec.push(T::zero());
        }

        cuda_runtime::memcpy(
            vec.as_ptr() as *mut T,
            self.data_ptr,
            size * std::mem::size_of::<T>(),
            cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        )?;

        Ok(vec)
    }
}

impl<'a, T: Num> Clone for Array<'a, T> {
    fn clone(&self) -> Self {
        let size = self.dim.size() * std::mem::size_of::<T>();
        let array = malloc_array_on_device(&self.dim, self.state).unwrap();
        cuda_runtime::memcpy(
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

#[cfg(test)]
mod test {
    use crate::array::compare::Equal;
    use crate::array::Array;
    use crate::CursState;
    #[test]
    fn test_clone() {
        let status = CursState::new(0);

        let array_a = Array::<f32>::ones(&vec![10, 10], &status).unwrap();
        let array_b = array_a.clone();

        let comp_res = array_a.eq(&array_b);

        let mut ans_vec = Vec::<f32>::with_capacity(100);
        for _ in 0..100 {
            ans_vec.push(1f32);
        }
        assert_eq!(comp_res.as_vec().unwrap(), ans_vec);
    }
}
