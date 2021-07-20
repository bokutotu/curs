use kernel::{
    equal, negativeEqual, greater, 
    greaterEqual, less, lessEqual};

use ffi;

use std::raw;

macro_rules! impl_compare_fn {
    ($func_name: ident, $kernel_func: ident) => {
        pub fn $func_name<T>(
            compareArrayA: *mut T,
            compareArrayB: *mut T,
            resArray: *mut T,
            size: i32) -> Result<()> {
            let threads_per_block = 256usize;
            let block_dim = ffi::usize_to_dim3(threads_per_block, 1usize, 1usize);
            let grid_dim = ffi::usize_to_dim3(
                (size + threads_per_block - 1) / threads_per_block, 1usize, 1usize);
            let shared_mem = 0usize;
            let size = size as raw::c_int;
            
            ffi::launch(
                $kernel_func as *const raw::c_void,
                grid_dim,
                block_dim,
                &mut [
                    &mut compareArrayA as *mut *mut T as *mut raw::c_void,
                    &mut compareArrayB as *mut *mut T as *mut raw::c_void,
                    &mut resArray as *mut *mut bool as *mut raw::c_void,
                    &size as *const c_int as *mut c_int as *mut c_void,
                ],
                shared_mem
            ).unwrap()
        }
    }
}

impl_compare_fn!(impl_equal, equal);
impl_compare_fn!(impl_negative_equal, negativeEqual);
impl_compare_fn!(impl_grater, greater);
impl_compare_fn!(impl_grater_equal, greaterEqual);
impl_compare_fn!(impl_less, less);
impl_compare_fn!(impl_less_equal, lessEqual);
