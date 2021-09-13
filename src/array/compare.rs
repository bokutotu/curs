use crate::array::Array;
use crate::kernel::compare::{
    impl_equal_float, impl_grater_equal_float, impl_grater_float, impl_less_equal_float,
    impl_less_float, impl_negative_equal_float,
};
// use crate::{dim::Dim, dtype::DataType, Num, Order};

macro_rules! impl_comaare {
    ($trait_name: ident, $trait_func: ident, $float_function: ident) => {
        pub trait $trait_name {
            type Array;
            fn $trait_func(&self, other: &Self::Array) -> Self::Array;
        }

        impl<'a> $trait_name for Array<'a, f32> {
            type Array = Array<'a, f32>;
            fn $trait_func(&self, other: &Self::Array) -> Self::Array {
                if self.dim != other.dim {
                    panic!("dimention is not same");
                }
                if self.order != other.order {
                    todo!();
                }
                $float_function(self, other, self.state).unwrap()
            }
        }
    };
}

impl_comaare!(Equal, eq, impl_equal_float);
impl_comaare!(NegativeEq, neq, impl_negative_equal_float);
impl_comaare!(Greater, greater, impl_grater_float);
impl_comaare!(GreaterEq, greater_eq, impl_grater_equal_float);
impl_comaare!(Less, less, impl_less_float);
impl_comaare!(LessEq, less_eq, impl_less_equal_float);
