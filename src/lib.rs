use num_traits;
use cublas_sys::{cublasContext, cublasHandle_t, cublasStatus_t, cublasCreate_v2, cublasDestroy_v2};
use cuda_runtime_sys::{ cudaDeviceReset, };

pub mod array;
pub mod compare;
pub mod dim;
pub mod dtype;
pub mod ffi;
pub mod operator;

/// Structure for checking the status of cublas and cuda
#[derive(Debug)]
pub struct CursState {
    cublas_handle: cublasHandle_t,
    cublas_state: cublasStatus_t,
}

impl CursState {
    pub fn new(dev_id: usize) -> Self {
        ffi::device_config(dev_id).unwrap();

        let handle:[u8; 0] = [];
        let mut handle: cublasHandle_t = handle.as_ptr() as *mut cublasContext as cublasHandle_t;

        let state = unsafe {cublasCreate_v2(&mut handle) };

        CursState {
            cublas_handle: handle,
            cublas_state: state,
        }
    }
}

impl Drop for CursState {
    fn drop(&mut self) {
        unsafe { cublasDestroy_v2(self.cublas_handle as cublasHandle_t) };
        unsafe { cudaDeviceReset() };
        println!("here from drop curs states")
    }
}

pub trait Num:
    num_traits::identities::Zero + num_traits::identities::One + num_traits::NumAssignOps + Copy
{
    fn dtype() -> dtype::DataType;
    fn bites() -> usize;
}

impl Num for f32 {
    fn dtype() -> dtype::DataType {
        dtype::DataType::FLOAT
    }

    fn bites() -> usize {
        4
    }
}

impl Num for i16 {
    fn dtype() -> dtype::DataType {
        dtype::DataType::INT16
    }

    fn bites() -> usize {
        2
    }
}

/// enum to determine whether an array is column priority or row priority.
#[derive(Debug, Clone, PartialEq)]
pub enum Order {
    /// column priority (C)
    C,
    /// row priority (Fortran array)
    F,
}

impl Default for Order {
    fn default() -> Self {
        Order::C
    }
}
