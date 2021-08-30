use crate::array::Array;
use crate::CursState;
use cublas_sys::*;

pub fn saxpy<'a>(handle: &'a CursState, alpha: f32, x: &Array<'a, f32>, y: &Array<'a, f32>) {
    let result = unsafe {
        cublasSaxpy_v2(
            handle.cublas_handle,
            x.dim.size() as i32,
            &alpha as *const f32,
            x.data_ptr as *const f32,
            1,
            y.data_ptr as *mut f32,
            1,
        )
    };
    if result != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        panic!("{:?}", result);
    }
}

pub fn daxpy<'a>(handle: &'a CursState, alpha: f64, x: &Array<'a, f64>, y: &Array<'a, f64>) {
    let result = unsafe {
        cublasDaxpy_v2(
            handle.cublas_handle,
            x.dim.size() as i32,
            &alpha as *const f64,
            x.data_ptr as *const f64,
            1,
            y.data_ptr as *mut f64,
            1,
        )
    };
    if result != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        panic!("{:?}", result);
    }
}
