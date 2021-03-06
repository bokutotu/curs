use kernel::{
    equalFloat, greaterEqualFloat, greaterFloat, lessEqualFloat, lessFloat, negativeEqualFloat,
};

use crate::array::Array;
use crate::cuda_runtime;
use crate::CursState;
use crate::Num;

macro_rules! impl_compare_fn {
    ($func_name: ident, $kernel_func: ident) => {
        pub fn $func_name<'a, T: Num>(
            array_a: &Array<T>,
            array_b: &Array<T>,
            state: &'a CursState,
        ) -> cuda_runtime::Result<Array<'a, T>> {
            if array_a.dim != array_b.dim {
                panic!(
                    "Array dim is not same, Array Dimention are {:?}, {:?}",
                    array_a.dim, array_b.dim
                );
            }

            if array_a.dtype != array_b.dtype {
                panic!(
                    "Data Type is not Same. DataType {:?} {:?} cant compare",
                    array_a.dtype, array_b.dtype
                );
            }

            if array_a.order != array_b.order {
                todo!();
            }

            let size = array_a.size() as ::libc::c_int;

            let res: Array<T> = Array::zeros(&array_a.shape(), state)?;

            unsafe {
                $kernel_func(
                    array_a.data_ptr as *mut f32,
                    array_b.data_ptr as *mut f32,
                    res.data_ptr as *mut f32,
                    size,
                );
            }

            Ok(res)
        }
    };
}

impl_compare_fn!(impl_equal_float, equalFloat);
impl_compare_fn!(impl_negative_equal_float, negativeEqualFloat);
impl_compare_fn!(impl_grater_float, greaterFloat);
impl_compare_fn!(impl_grater_equal_float, greaterEqualFloat);
impl_compare_fn!(impl_less_float, lessFloat);
impl_compare_fn!(impl_less_equal_float, lessEqualFloat);

#[cfg(test)]
mod tests {
    use cuda_runtime_sys;
    use kernel::equalFloat;
    use std::ptr::null_mut;

    #[test]
    fn kenel_is_work_check() {
        let mut d_a: *mut f32 = null_mut();
        let mut d_b: *mut f32 = null_mut();
        let mut res: *mut f32 = null_mut();

        let arraysize = 1024;

        let _cuda_error = unsafe {
            cuda_runtime_sys::cudaMalloc(
                &mut d_a as *mut *mut f32 as *mut *mut ::libc::c_void,
                arraysize,
            )
        };
        let _cuda_error = unsafe {
            cuda_runtime_sys::cudaMalloc(
                &mut d_b as *mut *mut f32 as *mut *mut ::libc::c_void,
                arraysize,
            )
        };
        let _cuda_error = unsafe {
            cuda_runtime_sys::cudaMalloc(
                &mut res as *mut *mut f32 as *mut *mut ::libc::c_void,
                arraysize,
            )
        };

        unsafe { equalFloat(d_a as *mut f32, d_b as *mut f32, res as *mut f32, 1024) };
    }
}
