//! Array addition
use std::mem::MaybeUninit;

use crate::array::Array;
use cublas_sys::*;

use crate::cublas::ffi::{cublas_handle_destroy, get_cuda_operation_t};

/// this function Add Array<f32> + Array<f32>
/// wrap of cublasSgeam
/// This function performs the matrix-matrix addition/transposition
/// matrix addition m x n
pub fn add_f32_matrix_matrix(
    a: &Array<f32>,
    b: &Array<f32>,
    is_transpose_a: bool,
    is_transpose_b: bool,
    is_h_a: bool,
    is_h_b: bool,
    m: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Result<Array<f32>, cublas_sys::cublasStatus_t> {
    let res_array = Array::zeros(&a.shape()).unwrap();

    let mut cublas_handle =
        [0u8; 0].as_mut_ptr() as *mut cublas_sys::cublasContext as cublas_sys::cublasHandle_t;

    let cublas_error =
        unsafe { cublas_sys::cublasCreate_v2(&mut cublas_handle as *mut cublasHandle_t) };

    if cublas_error != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(cublas_error);
    }

    let cublas_operation_a = get_cuda_operation_t(is_transpose_a, is_h_a);
    let cublas_operation_b = get_cuda_operation_t(is_transpose_b, is_h_b);

    let cublas_error = unsafe {
        cublas_sys::cublasSgeam(
            cublas_handle,
            cublas_operation_a,
            cublas_operation_b,
            m as ::libc::c_int,
            n as ::libc::c_int,
            &alpha as *const f32,
            a.data_ptr as *const f32,
            m as ::libc::c_int,
            &beta as *const f32,
            b.data_ptr as *const f32,
            m as ::libc::c_int,
            res_array.data_ptr,
            m as ::libc::c_int,
        )
    };

    if cublas_error != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(cublas_error);
    }

    cublas_handle_destroy(cublas_handle)?;

    Ok(res_array)
}

#[test]
fn test_add_f32_matrix_matrix() {
    let mut vec = vec![0., 1., 2., 3., 4., 5.];
    vec.reverse();
    let a = Array::from_vec(vec![0., 1., 2., 3., 4., 5.], &vec![3, 2]).unwrap();
    let b = Array::from_vec(vec, &vec![3, 2]).unwrap();

    let c = add_f32_matrix_matrix(&a, &b, false, false, false, false, 3, 3, 1., 1.).unwrap();
    assert_eq!([5f32, 5., 5., 5., 5., 5.], c.as_vec().unwrap().as_slice());
}
