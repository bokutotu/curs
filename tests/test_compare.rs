use curs;
use curs::array::Array;

#[test]
fn test_eq() {
    curs::ffi::device_config(0 as usize).unwrap();
    // let host_vec_a: Vec<f32> = vec![1f32, 2., 3., 4., 5., 6., 7., 8., 9.];
    // let mut host_vec_b = host_vec_a.clone(); host_vec_b.reverse();

    let device_array_a: Array<f32> = Array::ones(&vec![256usize]).unwrap();
    let device_array_b: Array<f32> = Array::ones(&vec![256usize]).unwrap();

    println!("device array a {:?}", device_array_a.as_vec());
    println!("device array b {:?}", device_array_b.as_vec());

    let res_array = device_array_a.eq(&device_array_b).unwrap();

    println!("device array a {:?}", device_array_a.as_vec());
    println!("device array b {:?}", device_array_b.as_vec());
    println!("Result array {:?}", res_array.as_vec());

    println!("line 19");

}
