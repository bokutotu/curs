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
