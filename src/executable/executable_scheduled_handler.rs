use crate::TensorData;
use block2::{Block, DynBlock, RcBlock};
use objc2_foundation::{NSArray, NSError};
use std::{ops::Deref, ptr::NonNull};

/// A notification when graph executable execution schedules.
///
/// - Parameters:
/// - results: If no error, the results produced by the graph operation.
/// - error: If an error occurs, more information might be found here.
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphexecutablescheduledhandler?language=objc)
pub struct ExecutableScheduledHandler(RcBlock<dyn Fn(NonNull<NSArray<TensorData>>, *mut NSError)>);

impl ExecutableScheduledHandler {
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn(Box<[&TensorData]>) -> Result<(), NSError> + 'static,
    {
        Self(RcBlock::new(
            move |tensors_array: NonNull<NSArray<TensorData>>, error: *mut NSError| {
                let tensors_array = unsafe { tensors_array.as_ref() };
                let tensors_boxed_slice =
                    unsafe { tensors_array.to_vec_unchecked() }.into_boxed_slice();
                if let Err(e) = handler(tensors_boxed_slice) {
                    unsafe {
                        *error = e;
                    }
                }
            },
        ))
    }

    pub fn copy(ptr: *mut DynBlock<dyn Fn(NonNull<NSArray<TensorData>>, *mut NSError)>) -> Self {
        Self(unsafe { RcBlock::copy(ptr) }.unwrap())
    }
}

impl Deref for ExecutableScheduledHandler {
    type Target = Block<dyn Fn(NonNull<NSArray<TensorData>>, *mut NSError)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
