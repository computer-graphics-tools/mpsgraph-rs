use crate::Tensor;

pub enum ShapeOrTensor<'a> {
    Shape(&'a [isize]),
    Tensor(&'a Tensor),
}
