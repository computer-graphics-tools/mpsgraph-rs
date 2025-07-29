use crate::Tensor;
use block2::{Block, IntoBlock, RcBlock};
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSMutableArray};
use std::{ops::Deref, ptr::NonNull};

/// The block that executes after the condition evaluates for each iteration.
///
/// # Arguments
///
/// * `body_block_arguments` - Inputs to the body of the while loop returned by the condition block.
///   These tensors must have the same element types as the return value of the while loop.
///
/// # Returns
///
/// A valid [`Tensor`] array with results forwarded to the condition block.
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphwhileafterblock?language=objc)
pub struct WhileAfterBlock(RcBlock<dyn Fn(NonNull<NSArray<Tensor>>) -> NonNull<NSArray<Tensor>>>);

impl WhileAfterBlock {
    pub fn new<F>(while_after_ops: F) -> Self
    where
        F: Fn(&[&Tensor]) -> Box<[Retained<Tensor>]> + 'static,
    {
        Self(RcBlock::new(
            move |body_block_arguments: NonNull<NSArray<Tensor>>| {
                let body_block_arguments =
                    unsafe { body_block_arguments.as_ref().to_vec_unchecked() };
                let results = while_after_ops(&body_block_arguments);
                let results_array = NSArray::from_retained_slice(&results);
                let raw = Retained::autorelease_return(results_array);
                unsafe { NonNull::new_unchecked(raw) }
            },
        ))
    }
}

impl Deref for WhileAfterBlock {
    type Target = Block<dyn Fn(NonNull<NSArray<Tensor>>) -> NonNull<NSArray<Tensor>>>;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}
