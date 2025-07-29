use crate::Tensor;

/// Defines how per-split sizes are provided to [`split_tensor`].
///
/// The caller can supply a compile-time slice of lengths (`Sizes`) or a graph
/// [`Tensor`] that contains those lengths computed at runtime (`Tensor`).
///
pub enum SizesOrTensor<'a> {
    Sizes(&'a [isize]),
    Tensor(&'a Tensor),
}
