/// A structure representing the dimension of an array
/// Representing dimensions with slices of type usize
#[derive(Clone, Debug, PartialEq)]
pub struct Dim {
    dimention: Vec<usize>,
}

impl Dim {
    /// Define the new dimensions
    fn new<V: AsRef<Vec<usize>> + ?Sized>(dim: &V) -> &Dim {
        unsafe { &*(dim.as_ref() as *const Vec<usize> as *const Dim) }
    }

    /// Calculate the number of elements when Dim is given.
    pub fn size(&self) -> usize {
        self.dimention.iter().fold(1, |x, y| x * y) as usize
    }

    /// return how many dimention
    pub fn len(&self) -> usize {
        self.dimention.len()
    }

    pub fn get_raw(&self) -> Vec<usize> {
        self.dimention.clone()
    }
}

impl AsRef<Dim> for Dim {
    fn as_ref(&self) -> &Dim {
        self
    }
}

impl AsRef<Dim> for Vec<usize> {
    fn as_ref(&self) -> &Dim {
        Dim::new(self)
    }
}

impl AsRef<Dim> for [usize] {
    fn as_ref(&self) -> &Dim {
        unsafe { &*(self as *const [usize] as *const Vec<usize> as *const Dim) }
    }
}
