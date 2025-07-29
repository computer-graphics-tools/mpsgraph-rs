/// Allows specifying a list of axes either as a Rust slice or as a graph
/// [`Tensor`] input.
///
/// Many shape-manipulation ops accept their *axes* argument in two forms:
/// a static slice known at compile time (`Axes`) or a tensor that will be
/// filled/known only at graph-build time (`Tensor`).
///
use crate::Tensor;

pub enum AxesOrTensor<'a> {
    Axes(&'a [isize]),
    Tensor(&'a Tensor),
}
