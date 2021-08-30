//! Implementing an operator for an Array
use std::ops;

use super::array::Array;
use super::cublas::level1::{daxpy, saxpy};
use super::element_wise_operator::{element_wise_devide, element_wise_product};

macro_rules! add_sub_impl {
    ($impl_name: ident, $func_name: ident, $type: ty, $alpha: literal, $cublas_func: ident) => {
        impl<'a> ops::$impl_name<Array<'a, $type>> for Array<'a, $type> {
            type Output = Array<'a, $type>;
            fn $func_name(self, other: Array<'a, $type>) -> Self::Output {
                if self.dim != other.dim {
                    panic!("dimension mismatch");
                }
                if self.order != other.order {
                    todo!();
                }
                let self_clone = self.clone();
                $cublas_func(self.state, $alpha, &other, &self_clone);
                self_clone
            }
        }
    };
}

add_sub_impl!(Add, add, f32, 1f32, saxpy);
add_sub_impl!(Add, add, f64, 1f64, daxpy);
add_sub_impl!(Sub, sub, f32, -1f32, saxpy);
add_sub_impl!(Sub, sub, f64, -1f64, daxpy);

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
