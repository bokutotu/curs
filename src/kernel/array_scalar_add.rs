use kernel::{doubleArrayScalarAdd, floatArrayScalarAdd};

use crate::array::Array;

pub fn float_array_add_scalar<'a>(x: Array<'a, f32>, y: f32) -> Array<'a, f32> {
    let size = x.size() as ::libc::c_int;
    let res = Array::zeros(&x.shape(), x.state).unwrap();
    unsafe {
        floatArrayScalarAdd(
            x.data_ptr as *mut ::libc::c_void,
            res.data_ptr as *mut ::libc::c_void,
            y as ::libc::c_float,
            size,
        )
    };
    res
}

pub fn double_array_add_scalar<'a>(x: Array<'a, f64>, y: f64) -> Array<'a, f64> {
    let size = x.size() as ::libc::c_int;
    let res = Array::zeros(&x.shape(), x.state).unwrap();
    unsafe {
        doubleArrayScalarAdd(
            x.data_ptr as *mut ::libc::c_void,
            res.data_ptr as *mut ::libc::c_void,
            y as ::libc::c_double,
            size,
        )
    };
    res
}
