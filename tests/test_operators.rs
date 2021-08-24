use curs::array::Array;
use curs::CursState;

#[test]
fn test_add() {
    let status = CursState::new(0);

    let mut source_vec_a = Vec::<f32>::with_capacity(1000);
    let mut source_vec_b = Vec::<f32>::with_capacity(1000);
    let mut ans = Vec::<f32>::with_capacity(1000);

    for i in 0..1000 {
        source_vec_a.push(i as f32);
        source_vec_b.push(i as f32 * 2.);
        ans.push(i as f32 * 3.);
    }
    let source_array_a = Array::from_vec(source_vec_a, &vec![10, 100], &status).unwrap();
    let source_array_b = Array::from_vec(source_vec_b, &vec![10, 100], &status).unwrap();

    let res = source_array_a + source_array_b;

    assert_eq!(res.as_vec().unwrap(), ans);
}

#[test]
fn test_sub() {
    let status = CursState::new(0);

    let mut source_vec_a = Vec::<f32>::with_capacity(100);
    let mut source_vec_b = Vec::<f32>::with_capacity(100);
    let mut ans = Vec::<f32>::with_capacity(100);

    for i in 0..100 {
        source_vec_a.push(i as f32);
        source_vec_b.push(i as f32 * 2.);
        ans.push(i as f32);
    }
    let source_array_a = Array::from_vec(source_vec_a, &vec![10, 10], &status).unwrap();

    let source_array_b = Array::from_vec(source_vec_b, &vec![10, 10], &status).unwrap();

    let res = source_array_b - source_array_a;

    assert_eq!(res.as_vec().unwrap(), ans);
}
