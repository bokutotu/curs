use crate::array::Array;
use crate::{dim::Dim, Num};

impl<'a, T: Num> Array<'a, T> {
    /// return shape (Dim)
    pub fn shape(&self) -> Dim {
        self.dim.clone()
    }

    /// return number of elements
    pub fn size(&self) -> usize {
        self.dim.size()
    }

    /// flatten Array
    pub fn flatten(self) -> Array<'a, T> {
        let order = self.order.clone();
        let ptr = self.data_ptr;
        let dtype = self.dtype.clone();
        let state = self.state;
        let size = self.dim.size();

        std::mem::forget(self);

        Array {
            data_ptr: ptr,
            dim: Dim::new(&vec![size]).clone(),
            order: order,
            dtype: dtype,
            state: state,
        }
    }

    /// reshape Array
    pub fn reshape<D: AsRef<Dim>>(self, new_dim: &D) -> Array<'a, T> {
        let new_dim = new_dim.as_ref();
        if new_dim.size() != self.dim.size() {
            panic!("Number of elements is not same");
        }

        let order = self.order.clone();
        let ptr = self.data_ptr;
        let dtype = self.dtype.clone();
        let state = self.state;

        std::mem::forget(self);

        Array {
            data_ptr: ptr,
            dim: new_dim.clone(),
            order: order.clone(),
            dtype: dtype.clone(),
            state: state,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::CursState;

    #[test]
    fn test_flatten() {
        let status = CursState::new(0);
        let x = Array::<f32>::zeros(&vec![100, 100], &status).unwrap();
        let x = x.flatten();
        println!("{:?}", x);
    }

    #[test]
    fn test_reshape() {
        let status = CursState::new(0);
        let x = Array::<f32>::zeros(&vec![100, 100], &status).unwrap();
        let x = x.reshape(&vec![10, 10, 100]);
        println!("{:?}", x);
    }
}
