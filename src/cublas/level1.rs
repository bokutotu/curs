use crate::array::Array;
use crate::error::{CublasError, Error};
use cublas_sys::*;

pub fn saxpy<'a>(alpha: f32, x: &Array<'a, f32>, y: &Array<'a, f32>) -> Result<(), Error> {
    let result = unsafe {
        cublasSaxpy_v2(
            x.state.cublas_handle,
            x.dim.size() as i32,
            &alpha as *const f32,
            x.data_ptr as *const f32,
            1,
            y.data_ptr as *mut f32,
            1,
        )
    };
    if result != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(Error::Cublas(CublasError { raw: result }));
    }
    Ok(())
}

pub fn daxpy<'a>(alpha: f64, x: &Array<'a, f64>, y: &Array<'a, f64>) -> Result<(), Error> {
    let result = unsafe {
        cublasDaxpy_v2(
            x.state.cublas_handle,
            x.dim.size() as i32,
            &alpha as *const f64,
            x.data_ptr as *const f64,
            1,
            y.data_ptr as *mut f64,
            1,
        )
    };
    if result != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(Error::Cublas(CublasError { raw: result }));
    }
    Ok(())
}
