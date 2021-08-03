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
        let mut size = 1;
        for item in &self.dimention { 
            size = size * item;
        }
        size
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
        unsafe {
            &*(self as *const [usize]
                as *const Vec<usize> as *const Dim)
        }
    }
}