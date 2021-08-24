//! Implementing an operator for an Array
use std::ops;

use cublas_sys::{cublasSaxpy_v2, cublasStatus_t};

use super::array::Array;

macro_rules! array_operator {
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

array_operator!(Add, add, f32, 1f32, cublasSaxpy_v2);
array_operator!(Sub, sub, f32, -1f32, cublasSaxpy_v2);

#[cfg(test)]
mod test_operators {
    use crate::array::Array;
    use crate::CursState;

    #[test]
    fn test_add() {
        let status = CursState::new(0);

        let mut source_vec_a = Vec::<f32>::with_capacity(1000);
        let mut source_vec_b = Vec::<f32>::with_capacity(1000);
        let mut ans = Vec::<f32>::with_capacity(1000);

        for i in 0..1000 {
            source_vec_a.push(i as f32);
            source_vec_b.push(i as f32 * 2.);
            ans.push(i as f32 * 3.);
        }
        let source_array_a = Array::from_vec(source_vec_a, &vec![10, 100], &status).unwrap();
        let source_array_b = Array::from_vec(source_vec_b, &vec![10, 100], &status).unwrap();

        let res = source_array_a + source_array_b;

        assert_eq!(res.as_vec().unwrap(), ans);
    }

    #[test]
    fn test_sub() {
        let status = CursState::new(0);

        let mut source_vec_a = Vec::<f32>::with_capacity(1000);
        let mut source_vec_b = Vec::<f32>::with_capacity(1000);
        let mut ans = Vec::<f32>::with_capacity(1000);

        for i in 0..1000 {
            source_vec_a.push(i as f32);
            source_vec_b.push(i as f32 * 2.);
            ans.push(i as f32);
        }
        let source_array_a = Array::from_vec(source_vec_a, &vec![10, 100], &status).unwrap();

        let source_array_b = Array::from_vec(source_vec_b, &vec![10, 100], &status).unwrap();

        let res = source_array_b - source_array_a;

        assert_eq!(res.as_vec().unwrap(), ans);
    }
}
