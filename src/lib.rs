use std::cmp;

pub mod array;
pub mod ffi;

/// data type. it must be an argument of numpy.dtype.
pub enum DataType {
    INT16,
    INT32,
    FLOAT,
}

impl cmp::PartialEq for DataType {
    #[inline]
    fn eq(&self, other: &DataType) -> bool {
        match (self, other) {
            (DataType::INT16, DataType::INT16) => true,
            (DataType::INT32, DataType::INT32) => true,
            (DataType::FLOAT, DataType::FLOAT) => true,
            _ => false,
        }
    }
}



pub trait Num {
    fn dtype() -> DataType;
}

impl Num for f32 {
    fn dtype() -> DataType {
        DataType::FLOAT
    }
}
impl Num for i16 {
    fn dtype() -> DataType {
        DataType::INT16
    }
}
impl Num for i32 {
    fn dtype() -> DataType {
        DataType::INT32
    }
}


