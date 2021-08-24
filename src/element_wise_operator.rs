use kernel::{float_element_wise_devide, float_element_wise_product};

use crate::array::Array;
use crate::Num;

pub fn element_wise_product<'a, T: Num>(
    array_a: Array<'a, T>,
    array_b: Array<'a, T>,
) -> Array<'a, T> {
    let res_array = Array::zeros(&array_a.dim, array_a.state).unwrap();
    if array_a.dim != array_b.dim {
        panic!("input array's dimention is not same");
    }
    if array_a.dtype != array_b.dtype {
        panic!("input array dtype is not same");
    }
    unsafe {
        float_element_wise_product(
            array_a.data_ptr as *mut ::libc::c_void,
            array_b.data_ptr as *mut ::libc::c_void,
            res_array.data_ptr as *mut ::libc::c_void,
            array_a.dim.size() as ::libc::c_int,
        )
    };
    res_array
}

pub fn element_wise_devide<'a, T: Num>(
    array_a: Array<'a, T>,
    array_b: Array<'a, T>,
) -> Array<'a, T> {
    let res_array = Array::zeros(&array_a.dim, array_a.state).unwrap();
    if array_a.dim != array_b.dim {
        panic!("input array's dimention is not same");
    }
    if array_a.dtype != array_b.dtype {
        panic!("input array dtype is not same");
    }
    unsafe {
        float_element_wise_devide(
            array_a.data_ptr as *mut ::libc::c_void,
            array_b.data_ptr as *mut ::libc::c_void,
            res_array.data_ptr as *mut ::libc::c_void,
            array_a.dim.size() as ::libc::c_int,
        )
    };
    res_array
}

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::element_wise_operator::*;
    use crate::CursState;

    #[test]
    fn test_product() {
        let state = CursState::new(0);
        let array_a =
            Array::<f32>::from_vec(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3], &state).unwrap();
        let array_b =
            Array::<f32>::from_vec(vec![7., 8., 9., 10., 11., 12.], &vec![2, 3], &state).unwrap();
        let res = element_wise_product(array_a, array_b);
        let res_vec = res.as_vec().unwrap();
        let ans: Vec<f32> = vec![7., 16., 27., 40., 55., 72.];
        assert_eq!(ans, res_vec);
    }
}
