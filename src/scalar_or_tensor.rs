use crate::Tensor;

pub trait GraphScalar: Copy {}
impl GraphScalar for f64 {}
impl GraphScalar for i64 {}

#[derive(Debug, Clone, Copy)]
pub enum ScalarOrTensor<'a, T: GraphScalar> {
    Scalar(T),
    Tensor(&'a Tensor),
}
