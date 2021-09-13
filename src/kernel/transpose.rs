use kernel::{doubleTranspose, floatTranspose};

use crate::array::{malloc_array_on_device, Array};

fn _idx_to_dim(idx: usize, stride: &Vec<usize>) -> Vec<usize> {
    let mut res = Vec::with_capacity(stride.len());
    let mut idx = idx;

    for _stride in stride {
        res.push(idx / _stride);
        idx = idx % _stride;
    }

    res
}

fn _dim_to_idx(dim: &Vec<usize>, stride: &Vec<usize>) -> usize {
    dim.iter()
        .zip(stride.iter())
        .fold(0, |x, (dim, stride)| x + (dim * stride))
}

fn _get_swap_axis_stride(x_stride: &Vec<usize>, swap_axis_idx: &Vec<usize>) -> Vec<usize> {
    let mut res = Vec::with_capacity(x_stride.len() as usize);
    for i in 0..x_stride.len() {
        res.push(x_stride[swap_axis_idx[i] as usize]);
    }
    res
}

fn _get_new_dim(dim: Vec<usize>, axis: &Vec<usize>) -> Vec<usize> {
    let mut res = Vec::with_capacity(dim.len());
    for i in axis.iter() {
        res.push(dim[*i]);
    }
    res
}

fn _transpose_index(x_stride: Vec<usize>, swap_axis_idx: &Vec<usize>) -> Vec<i32> {
    let size = (&x_stride).iter().fold(1, |x, y| x * y);
    let mut res = Vec::with_capacity(size as usize);
    let transpose_stride = _get_swap_axis_stride(&x_stride, &swap_axis_idx);

    for idx in 0..size {
        let x_dim = _idx_to_dim(idx as usize, &x_stride);
        let transpose_index = _dim_to_idx(&x_dim, &transpose_stride);
        res.push(transpose_index as i32);
    }

    res
}

macro_rules! impl_transpose {
    ($t: ty, $func: ident, $func_name: ident) => {
        pub fn $func_name<'a>(x: &Array<'a, $t>, axis: Vec<usize>) -> Array<'a, $t> {
            let size = x.dim.size();

            let mut x_stride = x
                .dim
                .get_raw()
                .iter()
                .rev()
                .scan(1, |prev, &x| {
                    let val = *prev;
                    *prev = *prev * x;
                    Some(val)
                })
                .collect::<Vec<usize>>();
            x_stride.reverse();

            let index = _transpose_index(x_stride, &axis);
            let y_dim = _get_new_dim(x.dim.get_raw(), &axis);
            let y = malloc_array_on_device(&y_dim, x.state).unwrap();

            unsafe {
                $func(
                    x.data_ptr as *mut $t,
                    y.data_ptr as *mut $t,
                    size as ::libc::c_int,
                    index.as_ptr(),
                )
            };

            y
        }
    };
}

impl_transpose!(f32, floatTranspose, float_transpose);
impl_transpose!(f64, doubleTranspose, double_transpose);

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::kernel::transpose::float_transpose;
    use crate::CursState;
    // use kernel::toy_func;

    // #[test]
    // fn test_toy_func_is_work() {
    //     let vec = vec![1, 10, 100, 1000];
    //     unsafe {
    //         toy_func(vec.as_ptr(), vec.len() as i32);
    //     }
    //     assert_eq!(1, 10);
    // }

    #[test]
    fn test_transpose() {
        let state = CursState::new(0);
        let mut x = Vec::with_capacity(27);
        for i in 0..27 {
            x.push(i as f32);
        }
        let x = Array::from_vec(x, &vec![3, 3, 3], &state).unwrap();
        let y = float_transpose(&x, vec![1, 0, 2]);
        let y = y.as_vec().unwrap();
        assert_eq!(
            y,
            vec![
                0f32, 1., 2., 9., 10., 11., 18., 19., 20., 3., 4., 5., 12., 13., 14., 21., 22.,
                23., 6., 7., 8., 15., 16., 17., 24., 25., 26.
            ]
        );
    }
}
