#[allow(dead_code)]

#[link(name="compare", kind="static")]
extern "C" {
    pub fn equalFloat(
        compareArrayA:  *mut ::libc::c_void, 
        compareArrayB:  *mut ::libc::c_void, 
        resArray:       *mut ::libc::c_void, 
        size: ::libc::c_int,
    );

    pub fn negativeEqualFloat(
        compareArrayA:  *mut ::libc::c_void, 
        compareArrayB:  *mut ::libc::c_void, 
        resArray:       *mut ::libc::c_void, 
        size: ::libc::c_int,
    );

    pub fn greaterFloat(
        compareArrayA:  *mut ::libc::c_void, 
        compareArrayB:  *mut ::libc::c_void, 
        resArray:       *mut ::libc::c_void, 
        size: ::libc::c_int,
    );

    pub fn greaterEqualFloat(
        compareArrayA:  *mut ::libc::c_void, 
        compareArrayB:  *mut ::libc::c_void, 
        resArray:       *mut ::libc::c_void, 
        size: ::libc::c_int,
    );

    pub fn lessFloat(
        compareArrayA:  *mut ::libc::c_void, 
        compareArrayB:  *mut ::libc::c_void, 
        resArray:       *mut ::libc::c_void, 
        size: ::libc::c_int,
    );

    pub fn lessEqualFloat(
        compareArrayA:  *mut ::libc::c_void, 
        compareArrayB:  *mut ::libc::c_void, 
        resArray:       *mut ::libc::c_void, 
        size: ::libc::c_int,
    );

    pub fn equalInt(
        compareArrayA:  *mut ::libc::c_void, 
        compareArrayB:  *mut ::libc::c_void, 
        resArray:       *mut ::libc::c_void, 
        size: ::libc::c_int,
    );

    pub fn negativeEqualInt(
        compareArrayA:  *mut ::libc::c_void, 
        compareArrayB:  *mut ::libc::c_void, 
        resArray:       *mut ::libc::c_void, 
        size: ::libc::c_int,
    );

    pub fn greaterInt(
        compareArrayA:  *mut ::libc::c_void, 
        compareArrayB:  *mut ::libc::c_void, 
        resArray:       *mut ::libc::c_void, 
        size: ::libc::c_int,
    );

    pub fn greaterEqualInt(
        compareArrayA:  *mut ::libc::c_void, 
        compareArrayB:  *mut ::libc::c_void, 
        resArray:       *mut ::libc::c_void, 
        size: ::libc::c_int,
    );

    pub fn lessInt(
        compareArrayA:  *mut ::libc::c_void, 
        compareArrayB:  *mut ::libc::c_void, 
        resArray:       *mut ::libc::c_void, 
        size: ::libc::c_int,
    );

    pub fn lessEqualInt(
        compareArrayA:  *mut ::libc::c_void, 
        compareArrayB:  *mut ::libc::c_void, 
        resArray:       *mut ::libc::c_void, 
        size: ::libc::c_int,
    );

}

#[cfg(test)]
mod tests {
    use cuda_runtime_sys;
    use std::ptr::null_mut;
    use super::*;
    #[test]
    pub fn it_works() {
        assert_eq!(2 + 2, 4);
        let mut d_a: *mut f32 = null_mut();
        let mut d_b: *mut f32 = null_mut();
        let mut res: *mut f32 = null_mut();
        
        let _cuda_error = unsafe {
            cuda_runtime_sys::cudaMalloc(
                &mut d_a as *mut *mut f32 as *mut *mut ::libc::c_void,
                1024,
            )
        };

        let _cuda_error = unsafe {
            cuda_runtime_sys::cudaMalloc(
                &mut d_b as *mut *mut f32 as *mut *mut ::libc::c_void,
                1024,
            )
        };

        let _cuda_error = unsafe {
            cuda_runtime_sys::cudaMalloc(
                &mut res as *mut *mut f32 as *mut *mut ::libc::c_void,
                1024,
            )
        };

        unsafe { 
            equalFloat(
                d_a as *mut f32 as *mut ::libc::c_void, 
                d_b as *mut f32 as *mut ::libc::c_void, 
                res as *mut f32 as *mut ::libc::c_void, 1024 
            )
        };
    }
}
