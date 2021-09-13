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

        std::mem::forget(self.data_ptr);

        Array {
            data_ptr: ptr,
            dim: Dim::new(&vec![self.dim.size()]).clone(),
            order: order,
            dtype: dtype,
            state: self.state,
        }
    }

    /// reshape Array
    pub fn reshape<D: AsRef<Dim>>(self, new_dim: &D) -> Array<'a, T> {
        let new_dim = new_dim.as_ref();
        if new_dim.size() != self.dim.size() {
            panic!("Number of elements is not same");
        }

        std::mem::forget(self.data_ptr);

        Array {
            data_ptr: self.data_ptr,
            dim: new_dim.clone(),
            order: self.order.clone(),
            dtype: self.dtype.clone(),
            state: self.state,
        }
    }
}
