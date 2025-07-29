use crate::Tensor;
use block2::{Block, RcBlock};
use objc2::rc::Retained;
use objc2_foundation::NSArray;
use std::{ops::Deref, ptr::NonNull};

/// A block of operations executed under either the `if` or `else` condition.
///
/// # Returns
///
/// Tensors returned by the user. If not empty, the user must define both the then and else blocks;
/// both should have the same number of arguments, and each corresponding argument should have the same element types.
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphifthenelseblock?language=objc)
pub struct IfThenElseBlock(RcBlock<dyn Fn() -> NonNull<NSArray<Tensor>>>);

impl IfThenElseBlock {
    pub fn new<F>(if_then_else_ops: F) -> Self
    where
        F: Fn() -> Box<[Retained<Tensor>]> + 'static,
    {
        Self(RcBlock::new(move || {
            let tensors = if_then_else_ops();
            let arr = NSArray::from_retained_slice(&tensors);
            let raw = Retained::autorelease_return(arr);
            unsafe { NonNull::new_unchecked(raw) }
        }))
    }
}

impl Deref for IfThenElseBlock {
    type Target = Block<dyn Fn() -> NonNull<NSArray<Tensor>>>;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}
