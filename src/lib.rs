use num_traits;

pub mod array;
pub mod ffi;
pub mod compare;

/// data type. it must be an argument of numpy.dtype.
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    INT16,
    FLOAT,
    BOOL,
}

#[test]
fn test_data_type_eq() {
    assert!(DataType::INT16 == DataType::INT16);
    assert!(DataType::INT16 != DataType::FLOAT);
}

pub trait Num:
    num_traits::identities::Zero
    + num_traits::identities::One
    + num_traits::NumAssignOps
    + Copy
    
{
    fn dtype() -> DataType;
    fn bites() -> usize;
}

impl Num for f32 {
    fn dtype() -> DataType {
        DataType::FLOAT
    }

    fn bites() -> usize {
        4
    }
}

impl Num for i16 {
    fn dtype() -> DataType {
        DataType::INT16
    }

    fn bites() -> usize {
        2
    }
}

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

/// enum to determine whether an array is column priority or row priority.
#[derive(Debug, Clone, PartialEq)]
pub enum Order {
    /// column priority (C)
    C,
    /// row priority (Fortran array)
    F
}

impl Default for Order {
    fn default() -> Self { 
        Order::C 
    }
}
