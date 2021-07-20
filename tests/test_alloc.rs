use cuda_runtime_sys::*;
use std::mem;

#[test]
fn test_malloc_cuda() {
    let host_slice = [10f32;100];
    unsafe {
        let device_slice: *mut f32 = ::libc::malloc(mem::size_of::<f32>()) as *mut f32;
        cudaMalloc(device_slice as *mut *mut ::libc::c_void, (host_slice.len() * mem::size_of::<f32>()) as u64);
    }
}