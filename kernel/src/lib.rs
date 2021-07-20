#[allow(dead_code)]

#[link(name="compare", kind="static")]
extern "C" {
    fn equal(
        compareArrayA: *mut ::libc::c_void, 
        compareArrayB: *mut ::libc::c_void, 
        resArray: bool, size: i32);
    fn negativeEqual(
        compareArrayA: *mut ::libc::c_void, 
        compareArrayB: *mut ::libc::c_void, 
        resArray: bool, size: i32);
    fn greater(
        compareArrayA: *mut ::libc::c_void, 
        compareArrayB: *mut ::libc::c_void, 
        resArray: bool, size: i32);
    fn less(
        compareArrayA: *mut ::libc::c_void, 
        compareArrayB: *mut ::libc::c_void, 
        resArray: bool, size: i32);
    fn greaterEqual(
        compareArrayA: *mut ::libc::c_void, 
        compareArrayB: *mut ::libc::c_void, 
        resArray: bool, size: i32);
    fn lessEqual(
        compareArrayA: *mut ::libc::c_void, 
        compareArrayB: *mut ::libc::c_void, 
        resArray: bool, size: i32);
}



#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
