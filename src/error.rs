use cublas_sys::cublasStatus_t;
use cuda_runtime_sys::cudaError_t;

use std::ffi::CStr;
use std::fmt::{self, Debug, Display, Formatter, Result};
use std::result;

use thiserror::Error;

pub struct CublasError {
    pub raw: cublasStatus_t,
}

fn cublas_error_to_string(error: cublasStatus_t) -> String {
    let string = match error {
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => {
                "The cuBLAS library was not initialized. This is usually caused by the lack of a prior cublasCreate() call, an error in the CUDA Runtime API called by the cuBLAS routine, or an error in the hardware setup.
To correct: call cublasCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed."
            }
            cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED => {
                "Resource allocation failed inside the cuBLAS library. This is usually caused by a cudaMalloc() failure.
To correct: prior to the function call, deallocate previously allocated memory as much as possible."
            }
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => {
                "An unsupported value or parameter was passed to the function (a negative vector size, for example).

To correct: ensure that all the parameters being passed have valid values."
            }
            cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => {
                "The function requires a feature absent from the device architecture; usually caused by compute capability lower than 5.0.

To correct: compile and run the application on a device with appropriate compute capability."
            }
            cublasStatus_t::CUBLAS_STATUS_MAPPING_ERROR => {
                "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.
To correct: prior to the function call, unbind any previously bound textures."
            }
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => {
                "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.

To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed."
            }
            cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR => {
                "An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure.

To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion."
            }
            cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED => {
                "The functionality requested is not supported"
            }
            cublasStatus_t::CUBLAS_STATUS_LICENSE_ERROR => {
                "The functionality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly."
            }
            _ => unreachable!(),
        };
    string.to_string()
}

impl Debug for CublasError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let string = cublas_error_to_string(self.raw);
        f.debug_struct("cublasError")
            .field("error status", &self.raw)
            .field("content", &string)
            .finish()
    }
}

impl Display for CublasError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let string = cublas_error_to_string(self.raw);
        write!(f, "({})", string)
    }
}

pub struct CudaError {
    pub raw: cudaError_t,
}
impl Debug for CudaError {
    fn fmt(&self, f: &mut Formatter) -> result::Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            unsafe { CStr::from_ptr(cuda_runtime_sys::cudaGetErrorString(self.raw)) }
                .to_string_lossy()
        )
    }
}

impl Display for CudaError {
    fn fmt(&self, f: &mut Formatter) -> result::Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            unsafe { CStr::from_ptr(cuda_runtime_sys::cudaGetErrorString(self.raw)) }
                .to_string_lossy()
        )
    }
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("Cuda Error {0}")]
    Cuda(CudaError),
    #[error("cublas Error {0}")]
    Cublas(CublasError),
}
