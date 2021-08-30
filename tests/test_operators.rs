use curs::array::Array;
use curs::CursState;

#[test]
fn test_add_array_array() {
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

#[test]
fn test_mul_array_array() {
    let status = CursState::new(0);

    let mut source_vec_a = Vec::<f32>::with_capacity(100);
    let mut source_vec_b = Vec::<f32>::with_capacity(100);
    let mut ans = Vec::<f32>::with_capacity(100);

    for i in 0..100 {
        source_vec_a.push(i as f32);
        source_vec_b.push(i as f32);
        ans.push((i * i) as f32);
    }
    let source_array_a = Array::from_vec(source_vec_a, &vec![10, 10], &status).unwrap();

    let source_array_b = Array::from_vec(source_vec_b, &vec![10, 10], &status).unwrap();

    let res = source_array_b * source_array_a;

    assert_eq!(res.as_vec().unwrap(), ans);
}

#[test]
fn test_mul_array_scalar() {
    let status = CursState::new(0);

    let mut x = Vec::<f32>::with_capacity(100);
    let y = 10f32;
    let mut ans = Vec::<f32>::with_capacity(100);
    for i in 0..100 {
        x.push(i as f32);
        ans.push((i * 10) as f32);
    }
    let x = Array::from_vec(x, &vec![10, 10], &status).unwrap();
    let res = x * y;
    assert_eq!(res.as_vec().unwrap(), ans)
}

#[test]
fn test_div_array_array() {
    let status = CursState::new(0);

    let mut source_vec_a = Vec::<f32>::with_capacity(100);
    let mut source_vec_b = Vec::<f32>::with_capacity(100);
    let mut ans = Vec::<f32>::with_capacity(100);

    for i in 1..101 {
        source_vec_a.push((i * 2) as f32);
        source_vec_b.push(i as f32);
        ans.push(0.5 as f32);
    }
    let source_array_a = Array::from_vec(source_vec_a, &vec![10, 10], &status).unwrap();

    let source_array_b = Array::from_vec(source_vec_b, &vec![10, 10], &status).unwrap();

    let res = source_array_b / source_array_a;

    assert_eq!(res.as_vec().unwrap(), ans);
}

#[test]
fn test_div_array_scalar() {
    let status = CursState::new(0);
    let mut x = Vec::<f32>::with_capacity(100);
    let mut ans = Vec::<f32>::with_capacity(100);

    for i in 0..100 {
        x.push(((i + 1) * 10) as f32);
        ans.push((i + 1) as f32);
    }

    let x = Array::from_vec(x, &vec![10, 10], &status).unwrap();
    let res = x / 10f32;
    assert_eq!(res.as_vec().unwrap(), ans);
}
