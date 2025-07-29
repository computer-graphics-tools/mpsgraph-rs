use crate::Tensor;
use block2::{Block, IntoBlock, RcBlock};
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSMutableArray};
use std::{ops::Deref, ptr::NonNull};

/// The block that executes before the condition evaluates for each iteration.
///
/// # Arguments
///
/// * `input_tensors` - Input tensors to the `whileConditionBlock`. For the first iteration, these are the same as the `initial_inputs` passed to the while loop.
/// * `result_tensors` - A valid [`Tensor`] array with results forwarded to the after block or returned from the while loop, depending on the predicate tensor. It will be empty and the caller block should fill it before returning.
///
/// # Returns
///
/// A [`Tensor`] that MUST be set and have a single scalar value, used to decide between executing the body block or returning from the while loop.
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphwhilebeforeblock?language=objc)
pub struct WhileBeforeBlock(
    RcBlock<dyn Fn(NonNull<NSArray<Tensor>>, NonNull<NSMutableArray<Tensor>>) -> NonNull<Tensor>>,
);

impl WhileBeforeBlock {
    pub fn new<F>(while_before_ops: F) -> Self
    where
        F: Fn(&[&Tensor], &mut [&Tensor]) -> Retained<Tensor> + 'static,
    {
        Self(RcBlock::new(
            move |input_tensors: NonNull<NSArray<Tensor>>,
                  result_tensors: NonNull<NSMutableArray<Tensor>>| {
                let inputs = unsafe { input_tensors.as_ref().to_vec_unchecked() };
                let mut results = unsafe { result_tensors.as_ref().to_vec_unchecked() };
                let tensor = while_before_ops(&inputs, &mut results);
                let raw = Retained::autorelease_return(tensor);
                unsafe { NonNull::new_unchecked(raw) }
            },
        ))
    }
}

impl Deref for WhileBeforeBlock {
    type Target =
        Block<dyn Fn(NonNull<NSArray<Tensor>>, NonNull<NSMutableArray<Tensor>>) -> NonNull<Tensor>>;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}
