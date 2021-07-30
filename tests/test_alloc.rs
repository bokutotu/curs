use curs;
use curs::array::Array;

#[test]
fn test_alloc_array_on_cuda() {
    curs::ffi::device_config(0 as usize).unwrap();

    let array = Array::<f32>::zeros(&vec![10,10]).unwrap();
    array.fill(10f32).unwrap();
    let _host_array = array.as_vec().unwrap();
    let _host_slice = array.as_slice().unwrap();
}
