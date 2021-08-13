use curs;
use curs::array::Array;

#[test]
fn test_alloc_array_on_cuda() {

    let state = curs::CursState::new(0usize);
    let array = Array::<f32>::zeros(&vec![10,10], &state).unwrap();
    array.fill(10f32).unwrap();
    let _host_array = array.as_vec().unwrap();
}