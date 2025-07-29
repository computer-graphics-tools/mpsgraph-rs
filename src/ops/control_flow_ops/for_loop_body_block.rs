use crate::Tensor;
use block2::{Block, RcBlock};
use objc2::rc::Retained;
use objc2_foundation::NSArray;
use std::{ops::Deref, ptr::NonNull};

/// A block for the body of the `for` loop.
///
/// # Arguments
///
/// * `index` - The loop index for this iteration; it is a scalar [`Tensor`].
/// * `iteration_arguments` - Arguments for this iteration, with the same count and corresponding element types as `initial_iteration_arguments` and the return types of the `for` loop.
///
/// # Returns
///
/// A valid [`Tensor`] slice with the same count and corresponding element types as `initial_iteration_arguments` and the return types of the `for` loop.
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphforloopbodyblock?language=objc)
pub struct ForLoopBodyBlock(
    RcBlock<dyn Fn(NonNull<Tensor>, NonNull<NSArray<Tensor>>) -> NonNull<NSArray<Tensor>>>,
);

impl ForLoopBodyBlock {
    pub fn new<F>(while_before_ops: F) -> Self
    where
        F: Fn(&Tensor, &[&Tensor]) -> Box<[Retained<Tensor>]> + 'static,
    {
        Self(RcBlock::new(
            move |index: NonNull<Tensor>, iteration_arguments: NonNull<NSArray<Tensor>>| {
                let index = unsafe { index.as_ref() };
                let iteration_arguments =
                    unsafe { iteration_arguments.as_ref().to_vec_unchecked() }.into_boxed_slice();
                let result = while_before_ops(index, &iteration_arguments);
                let result_array = NSArray::from_retained_slice(&result);
                let raw = Retained::autorelease_return(result_array);
                unsafe { NonNull::new_unchecked(raw) }
            },
        ))
    }
}

impl Deref for ForLoopBodyBlock {
    type Target =
        Block<dyn Fn(NonNull<Tensor>, NonNull<NSArray<Tensor>>) -> NonNull<NSArray<Tensor>>>;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}
