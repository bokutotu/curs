use crate::array::Array;
use crate::kernel::transpose::{double_transpose, float_transpose};

pub trait Transpose {
    type Output;
    fn transpose(self, axis_index: Vec<usize>) -> Self::Output;
}

impl<'a> Transpose for Array<'a, f32> {
    type Output = Array<'a, f32>;
    #[inline]
    fn transpose(self: Array<'a, f32>, axis_index: Vec<usize>) -> Self::Output {
        if *axis_index.iter().max().unwrap() >= self.dim.len() {
            panic!("index must be smaller than dim nuber");
        }
        if axis_index.len() > self.dim.len() {
            panic!("axis_index length is longer than array's dimentions");
        }
        let x = float_transpose(&self, axis_index);
        x
    }
}

impl<'a> Transpose for Array<'a, f64> {
    type Output = Array<'a, f64>;
    #[inline]
    fn transpose(self: Array<'a, f64>, axis_index: Vec<usize>) -> Self::Output {
        if *axis_index.iter().max().unwrap() >= self.dim.len() {
            panic!("index must be smaller than dim nuber");
        }
        if axis_index.len() > self.dim.len() {
            panic!("axis_index length is longer than array's dimentions");
        }
        let x = double_transpose(&self, axis_index);
        x
    }
}

#[cfg(test)]
mod tests {
    use crate::array::transpose::Transpose;
    use crate::array::Array;
    use crate::CursState;

    #[test]
    fn test_transpose() {
        let state = CursState::new(0);
        let mut x = Vec::with_capacity(27);
        for i in 0..27 {
            x.push(i as f32);
        }
        let x = Array::from_vec(x, &vec![3, 3, 3], &state).unwrap();
        let y = x.transpose(vec![1, 0, 2]);
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
