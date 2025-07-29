use crate::{Graph, Tensor};
use block2::{Block, RcBlock};
use objc2::rc::Retained;
use objc2_foundation::NSArray;
use std::{ops::Deref, ptr::NonNull};

/// The scope where all the operations defined in this block get control-dependency operations.
///
/// # Returns
///
/// A valid [`Tensor`] with the results forwarded to the return of [`Graph::control_dependency`].
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphcontrolflowdependencyblock?language=objc)
pub struct ControlFlowDependencyBlock(RcBlock<dyn Fn() -> NonNull<NSArray<Tensor>>>);

impl ControlFlowDependencyBlock {
    pub fn new<F>(control_flow_dependency_ops: F) -> Self
    where
        F: Fn() -> Box<[Retained<Tensor>]> + 'static,
    {
        Self(RcBlock::new(move || {
            let tensors = control_flow_dependency_ops();
            let arr = NSArray::from_retained_slice(&tensors);
            let raw = Retained::autorelease_return(arr);
            unsafe { NonNull::new_unchecked(raw) }
        }))
    }
}

impl Deref for ControlFlowDependencyBlock {
    type Target = Block<dyn Fn() -> NonNull<NSArray<Tensor>>>;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}
