//! This file defines the array of curs and the structures 
//! associated with the attributes that are members of the array. 
use std::cmp;

use crate::{DataType, Num};
use crate::ffi;

/// A structure representing the dimension of an array
/// Representing dimensions with slices of type usize
pub struct Dim {
    dimention: Vec<usize>,
}

impl Dim {
    /// Define the new dimensions
    fn new(dim: Vec<usize>) -> Self {
        Self {
            dimention: dim
        }
    }

    /// Calculate the number of elements when Dim is given.
    pub fn size(&self) -> usize {
        let mut size = 1;
        for item in self.dimention { 
            size = size * item;
        }
        size
    }
}

impl AsRef<Dim> for Dim {
    fn as_ref(&self) -> &Dim {
        self
    }
}

impl AsRef<Dim> for Vec<usize> {
    fn as_ref(&self) -> &Dim {
        &Dim::new(*self)
    }
}

macro_rules! impl_AsRef_slice {
    ($number_of_element: expr) => {
        impl AsRef<Dim> for [usize;$number_of_element] {
            fn as_ref(&self) -> &Dim {
                let vec = Vec::from(*self);
                &Dim::new(vec)
            }
        }
    }
}

impl_AsRef_slice!(1);
impl_AsRef_slice!(2);
impl_AsRef_slice!(3);
impl_AsRef_slice!(4);
impl_AsRef_slice!(5);
impl_AsRef_slice!(6);

impl cmp::PartialEq for Dim {
    #[inline]
    fn eq(&self, other: &Dim) -> bool {
        if other.dimention.len() != self.dimention.len() {
            false
        } else {
            self.dimention == other.dimention
        }
    }
}

/// enum to determine whether an array is column priority or row priority.
enum Order {
    /// column priority
    C,
    /// row priority
    F
}

impl Default for Order {
    fn default() -> Self { Order::C }
}

impl cmp::PartialEq for Order {
    #[inline]
    fn eq(&self, other: &Order) -> bool {
        match (self, other) {
            (Order::C, Order::C) => true,
            (Order::F, Order::F) => true,
            _ => false,
        }
    }
}

/// Multi-dimensional array on CUDA device.
/// # Parameters
/// * data_ptr <dr>
/// pointer to CUDA device
/// * dim <dr>
/// dimention 
/// Dimention of Array
/// * Order <dr>
/// Array is Column priority or Row priority
/// * dtype <dr>
/// Data type of Array
pub struct Array<T> {
    data_ptr: *mut T,
    dim: Dim,
    order: Order,
    dtype: DataType,
}

impl<T: Num> Array<T> {

    pub fn zeros<D: AsRef<Dim>>(dim: D) -> Array<T> {
        let dim = dim.as_ref();
        let size = dim.size();
        let order = Order::default();
        let dtype = Num::dtype();

        let device_ptr = ffi::malloc(size).unwrap();
    }

}
