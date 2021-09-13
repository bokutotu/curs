use crate::array::Array;
use crate::cuda_runtime;
use crate::CursState;
use crate::{dim::Dim, Num, Order};

/// Definieren eines Arrays auf dem Grafikprozessor
pub fn malloc_array_on_device<'a, T: Num, D: AsRef<Dim>>(
    dim: &D,
    state: &'a CursState,
) -> cuda_runtime::Result<Array<'a, T>> {
    let dim = dim.as_ref();
    let size = dim.size();
    let n_bytes = size * std::mem::size_of::<T>();
    let order = Order::default();
    let dtype = T::dtype();

    let device_ptr = cuda_runtime::malloc(n_bytes)?;

    Ok(Array {
        data_ptr: device_ptr,
        dim: dim.clone(),
        order: order,
        dtype: dtype,
        state: state,
    })
}

/// Fill with the value specified for the device's pointer
pub fn fill<T: Num>(data_ptr: *mut T, num: T, size: usize) -> cuda_runtime::Result<()> {
    let mut vec: Vec<T> = Vec::with_capacity(size);
    for _ in 0..size {
        vec.push(num);
    }

    cuda_runtime::memcpy(
        data_ptr,
        vec.as_ptr(),
        size * std::mem::size_of::<T>(),
        cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
    )
}

impl<'a, T: Num> Array<'a, T> {
    /// Initialize an Array of the given size with zero fill.
    pub fn zeros<D: AsRef<Dim>>(
        dim: &D,
        state: &'a CursState,
    ) -> cuda_runtime::Result<Array<'a, T>> {
        let dim = dim.as_ref();
        let array = malloc_array_on_device(&dim, state)?;
        fill(array.data_ptr, T::zero(), dim.size())?;

        Ok(array)
    }

    /// Initializes an Array of the given size with 1 fill
    pub fn ones<D: AsRef<Dim>>(
        dim: &D,
        state: &'a CursState,
    ) -> cuda_runtime::Result<Array<'a, T>> {
        let dim = dim.as_ref();
        let array = malloc_array_on_device(&dim, state)?;
        fill(array.data_ptr, T::one(), dim.size())?;

        Ok(array)
    }

    /// Fill an Array with the given values
    pub fn fill(&self, num: T) -> cuda_runtime::Result<()> {
        fill(self.data_ptr, num, self.dim.size())?;
        Ok(())
    }

    /// Defines an Array filled with the specified values
    pub fn full<D: AsRef<Dim>>(
        dim: &D,
        num: T,
        state: &'a CursState,
    ) -> cuda_runtime::Result<Array<'a, T>> {
        let dim = dim.as_ref();
        let array = malloc_array_on_device(&dim, state)?;
        fill(array.data_ptr, num, array.dim.size())?;
        Ok(array)
    }

    /// Converting from Vec to Array
    pub fn from_vec<D: AsRef<Dim>>(
        vec: Vec<T>,
        dim: &D,
        state: &'a CursState,
    ) -> cuda_runtime::Result<Array<'a, T>> {
        let dim = dim.as_ref();
        if vec.len() != dim.size() {
            panic!("input vec shape and input dimention size is not same");
        }
        let array = malloc_array_on_device(&dim, state)?;
        cuda_runtime::memcpy(
            array.data_ptr as *const T as *mut T,
            vec.as_ptr() as *const T,
            array.size() * std::mem::size_of::<T>(),
            cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
        )?;
        Ok(array)
    }
}
