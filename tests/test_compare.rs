use curs::array::compare::Equal;
use curs::array::Array;
use curs::CursState;

#[test]
fn test_eq() {
    let status = CursState::new(0usize);

    let device_array_a: Array<f32> = Array::ones(&vec![256usize], &status).unwrap();
    let device_array_b: Array<f32> = Array::ones(&vec![256usize], &status).unwrap();

    let _res_array = device_array_a.eq(&device_array_b);
}
