//! Implementing an operator for an Array
use std::ops;

use cublas_sys::{cublasSaxpy_v2, cublasStatus_t};

use super::array::Array;
use super::element_wise_operator::{element_wise_devide, element_wise_product};

macro_rules! array_operator_axpy {
    ($impl_name: ident , $func_name: ident, $type: ty, $alpha: literal, $cublas_func: ident) => {
        impl<'a> ops::$impl_name<Array<'a, $type>> for Array<'a, $type> {
            type Output = Array<'a, $type>;
            fn $func_name(self, other: Array<'a, $type>) -> Self::Output {
                if self.dim != other.dim {
                    panic!("add operation dimension mismatch");
                }
                if self.order != other.order {
                    todo!();
                }
                if self.dtype != other.dtype {
                    panic!("data type is not same ");
                }

                let self_clone = self.clone();

                let cublas_status = unsafe {
                    $cublas_func(
                        self.state.cublas_handle,
                        self.dim.size() as i32,
                        &mut $alpha as *const $type,
                        other.data_ptr as *const $type,
                        1,
                        self_clone.data_ptr as *mut $type,
                        1,
                    )
                };

                if cublas_status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    panic!("{:?}", cublas_status);
                }
                self_clone
            }
        }
    };
}

array_operator_axpy!(Add, add, f32, 1f32, cublasSaxpy_v2);
array_operator_axpy!(Sub, sub, f32, -1f32, cublasSaxpy_v2);

impl<'a> ops::Mul<Array<'a, f32>> for Array<'a, f32> {
    type Output = Array<'a, f32>;
    fn mul(self, other: Array<'a, f32>) -> Self::Output {
        if self.dim != other.dim {
            panic!("add operation dimension mismatch");
        }
        if self.order != other.order {
            todo!();
        }
        if self.dtype != other.dtype {
            panic!("data type is not same ");
        }

        element_wise_product(self, other)
    }
}

impl<'a> ops::Div<Array<'a, f32>> for Array<'a, f32> {
    type Output = Array<'a, f32>;
    fn div(self, other: Array<'a, f32>) -> Self::Output {
        if self.dim != other.dim {
            panic!("add operation dimension mismatch");
        }
        if self.order != other.order {
            todo!();
        }
        if self.dtype != other.dtype {
            panic!("data type is not same ");
        }

        element_wise_devide(self, other)
    }
}

// TODO
// impl Mul and Div Array<'a, f64> Array<'a, f64>
