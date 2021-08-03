use num_traits;

pub mod array;
pub mod compare;
pub mod dtype;
pub mod dim;
pub mod ffi;

pub trait Num:
    num_traits::identities::Zero
    + num_traits::identities::One
    + num_traits::NumAssignOps
    + Copy
    
{
    fn dtype() -> dtype::DataType;
    fn bites() -> usize;
}

impl Num for f32 {
    fn dtype() -> dtype::DataType {
        dtype::DataType::FLOAT
    }

    fn bites() -> usize {
        4
    }
}

impl Num for i16 {
    fn dtype() -> dtype::DataType {
        dtype::DataType::INT16
    }

    fn bites() -> usize {
        2
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
