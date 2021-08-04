/// error handle and enum operator
use cublas_sys::*;

pub fn init_cublas_status(
    handle: *mut cublas_sys::cublasHandle_t,
) -> Result<(), cublas_sys::cublasStatus_t> {
    match unsafe { cublas_sys::cublasCreate_v2(handle) } {
        cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
        err => Err(err),
    }
}

pub fn cublas_handle_destroy(handle: cublas_sys::cublasHandle_t) -> Result<(), cublasStatus_t> {
    match unsafe { cublas_sys::cublasDestroy_v2(handle) } {
        cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
        err => Err(err),
    }
}

pub fn get_cuda_operation_t(is_t: bool, is_h: bool) -> cublas_sys::cublasOperation_t {
    if is_t && is_h {
        unreachable!();
    } else if !is_t && !is_h {
        cublas_sys::cublasOperation_t::CUBLAS_OP_N
    } else if is_t {
        cublas_sys::cublasOperation_t::CUBLAS_OP_T
    } else {
        cublas_sys::cublasOperation_t::CUBLAS_OP_C
    }
}
