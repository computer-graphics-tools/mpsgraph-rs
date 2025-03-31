use crate::core::AsRawObject;
use metal::foreign_types::ForeignType;
use metal::{BlitCommandEncoder, CommandBuffer, CommandQueue, ComputeCommandEncoder, Device};
use objc2::msg_send;
use objc2::runtime::AnyObject;
use objc2_foundation::{NSError, NSString};
use std::fmt;

/// Command buffer status
#[repr(u64)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MTLCommandBufferStatus {
    /// The command buffer has not been enqueued yet
    NotEnqueued = 0,
    /// The command buffer is enqueued, but not committed
    Enqueued = 1,
    /// Committed to the command queue, but not yet scheduled for execution
    Committed = 2,
    /// All dependencies have been resolved and scheduled for execution
    Scheduled = 3,
    /// The command buffer has finished executing successfully
    Completed = 4,
    /// Execution was aborted due to an error
    Error = 5,
}

/// A wrapper for MPSCommandBuffer objects which is a subclass of MTLCommandBuffer
pub struct MPSCommandBuffer(pub(crate) *mut AnyObject);

// Implement Send + Sync for the wrapper type
unsafe impl Send for MPSCommandBuffer {}
unsafe impl Sync for MPSCommandBuffer {}

impl MPSCommandBuffer {
    /// Creates a new MPSCommandBuffer from a Metal CommandBuffer
    pub fn from_command_buffer(command_buffer: &CommandBuffer) -> Self {
        unsafe {
            let class_name = c"MPSCommandBuffer";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();

            // Get the command buffer pointer
            let buffer_ptr = command_buffer.as_ptr() as *mut AnyObject;

            // Create MPSCommandBuffer with the Metal command buffer
            let obj: *mut AnyObject = msg_send![cls, alloc];
            let mps_command_buffer: *mut AnyObject =
                msg_send![obj, initWithCommandBuffer: buffer_ptr];

            MPSCommandBuffer(mps_command_buffer)
        }
    }

    /// Creates a new MPSCommandBuffer from a command queue
    pub fn from_command_queue(command_queue: &CommandQueue) -> Self {
        unsafe {
            let class_name = c"MPSCommandBuffer";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();

            // Get the command queue pointer
            let queue_ptr = command_queue.as_ptr() as *mut AnyObject;

            // Create MPSCommandBuffer from the command queue
            let mps_command_buffer: *mut AnyObject =
                msg_send![cls, commandBufferFromCommandQueue: queue_ptr];

            MPSCommandBuffer(mps_command_buffer)
        }
    }

    /// Returns the underlying Metal CommandBuffer
    pub fn command_buffer(&self) -> CommandBuffer {
        unsafe {
            let cmd_buf_ptr: *mut AnyObject = msg_send![self.0, commandBuffer];
            CommandBuffer::from_ptr(cmd_buf_ptr as *mut metal::MTLCommandBuffer)
        }
    }

    /// Returns the root Metal CommandBuffer
    pub fn root_command_buffer(&self) -> CommandBuffer {
        unsafe {
            let cmd_buf_ptr: *mut AnyObject = msg_send![self.0, rootCommandBuffer];
            CommandBuffer::from_ptr(cmd_buf_ptr as *mut metal::MTLCommandBuffer)
        }
    }

    /// Commits the current command buffer and continues with a new one
    pub fn commit_and_continue(&self) {
        unsafe {
            let _: () = msg_send![self.0, commitAndContinue];
        }
    }

    /// Prefetches heap for workload size
    pub fn prefetch_heap_for_workload_size(&self, size: usize) {
        unsafe {
            let _: () = msg_send![self.0, prefetchHeapForWorkloadSize: size];
        }
    }

    // MTLCommandBuffer methods

    /// Returns the device this command buffer was created against
    pub fn device(&self) -> Device {
        unsafe {
            let device_ptr: *mut AnyObject = msg_send![self.0, device];
            Device::from_ptr(device_ptr as *mut metal::MTLDevice)
        }
    }

    /// Returns the command queue this command buffer was created from
    pub fn command_queue(&self) -> CommandQueue {
        unsafe {
            let queue_ptr: *mut AnyObject = msg_send![self.0, commandQueue];
            CommandQueue::from_ptr(queue_ptr as *mut metal::MTLCommandQueue)
        }
    }

    /// Returns whether this command buffer holds strong references
    pub fn retained_references(&self) -> bool {
        unsafe { msg_send![self.0, retainedReferences] }
    }

    /// Sets a label to help identify this object
    pub fn set_label(&self, label: &str) {
        unsafe {
            // Get the command buffer first - MPSCommandBuffer might not implement setLabel directly
            let cmd_buf_ptr: *mut AnyObject = msg_send![self.0, commandBuffer];
            if !cmd_buf_ptr.is_null() {
                let label_str = NSString::from_str(label);
                let _: () = msg_send![cmd_buf_ptr, setLabel:label_str.as_raw_object()];
            }
        }
    }

    /// Returns the current label
    pub fn label(&self) -> String {
        unsafe {
            // Get the command buffer first - MPSCommandBuffer might not implement label directly
            let cmd_buf_ptr: *mut AnyObject = msg_send![self.0, commandBuffer];
            if cmd_buf_ptr.is_null() {
                return String::new();
            }

            let label_ptr: *mut AnyObject = msg_send![cmd_buf_ptr, label];
            if label_ptr.is_null() {
                return String::new();
            }

            let nsstring: &NSString = &*(label_ptr as *const NSString);
            nsstring.to_string()
        }
    }

    /// Returns the kernel start time
    pub fn kernel_start_time(&self) -> f64 {
        unsafe {
            let cmd_buf_ptr: *mut AnyObject = msg_send![self.0, commandBuffer];
            if cmd_buf_ptr.is_null() {
                return 0.0;
            }
            msg_send![cmd_buf_ptr, kernelStartTime]
        }
    }

    /// Returns the kernel end time
    pub fn kernel_end_time(&self) -> f64 {
        unsafe {
            let cmd_buf_ptr: *mut AnyObject = msg_send![self.0, commandBuffer];
            if cmd_buf_ptr.is_null() {
                return 0.0;
            }
            msg_send![cmd_buf_ptr, kernelEndTime]
        }
    }

    /// Returns the GPU start time
    pub fn gpu_start_time(&self) -> f64 {
        unsafe {
            let cmd_buf_ptr: *mut AnyObject = msg_send![self.0, commandBuffer];
            if cmd_buf_ptr.is_null() {
                return 0.0;
            }
            msg_send![cmd_buf_ptr, GPUStartTime]
        }
    }

    /// Returns the GPU end time
    pub fn gpu_end_time(&self) -> f64 {
        unsafe {
            let cmd_buf_ptr: *mut AnyObject = msg_send![self.0, commandBuffer];
            if cmd_buf_ptr.is_null() {
                return 0.0;
            }
            msg_send![cmd_buf_ptr, GPUEndTime]
        }
    }

    /// Appends this command buffer to the end of its MTLCommandQueue
    pub fn enqueue(&self) {
        unsafe {
            let cmd_buf_ptr: *mut AnyObject = msg_send![self.0, commandBuffer];
            if !cmd_buf_ptr.is_null() {
                let _: () = msg_send![cmd_buf_ptr, enqueue];
            }
        }
    }

    /// Commits the command buffer for execution as soon as possible
    pub fn commit(&self) {
        unsafe {
            let cmd_buf_ptr: *mut AnyObject = msg_send![self.0, commandBuffer];
            if !cmd_buf_ptr.is_null() {
                let _: () = msg_send![cmd_buf_ptr, commit];
            }
        }
    }

    /// Waits until this command buffer has been scheduled
    pub fn wait_until_scheduled(&self) {
        unsafe {
            let cmd_buf_ptr: *mut AnyObject = msg_send![self.0, commandBuffer];
            if !cmd_buf_ptr.is_null() {
                let _: () = msg_send![cmd_buf_ptr, waitUntilScheduled];
            }
        }
    }

    /// Waits until this command buffer has completed execution
    pub fn wait_until_completed(&self) {
        unsafe {
            // Get the underlying Metal command buffer
            let cmd_buf_ptr: *mut AnyObject = msg_send![self.0, commandBuffer];
            if !cmd_buf_ptr.is_null() {
                let _: () = msg_send![cmd_buf_ptr, waitUntilCompleted];
            }
        }
    }

    /// Returns the current status of the command buffer
    pub fn status(&self) -> MTLCommandBufferStatus {
        unsafe {
            let cmd_buf_ptr: *mut AnyObject = msg_send![self.0, commandBuffer];
            if cmd_buf_ptr.is_null() {
                return MTLCommandBufferStatus::NotEnqueued;
            }

            let status: u64 = msg_send![cmd_buf_ptr, status];
            match status {
                0 => MTLCommandBufferStatus::NotEnqueued,
                1 => MTLCommandBufferStatus::Enqueued,
                2 => MTLCommandBufferStatus::Committed,
                3 => MTLCommandBufferStatus::Scheduled,
                4 => MTLCommandBufferStatus::Completed,
                5 => MTLCommandBufferStatus::Error,
                _ => MTLCommandBufferStatus::NotEnqueued,
            }
        }
    }

    /// Returns the error if an error occurred during execution
    pub fn error(&self) -> Option<String> {
        unsafe {
            let cmd_buf_ptr: *mut AnyObject = msg_send![self.0, commandBuffer];
            if cmd_buf_ptr.is_null() {
                return None;
            }

            let error_ptr: *mut AnyObject = msg_send![cmd_buf_ptr, error];
            if error_ptr.is_null() {
                None
            } else {
                let error = &*(error_ptr as *const NSError);
                let description = error.localizedDescription().to_string();
                Some(description)
            }
        }
    }

    /// Creates a compute command encoder to encode into this command buffer
    pub fn new_compute_command_encoder(&self) -> Option<ComputeCommandEncoder> {
        unsafe {
            let encoder_ptr: *mut AnyObject = msg_send![self.0, computeCommandEncoder];
            if encoder_ptr.is_null() {
                None
            } else {
                Some(ComputeCommandEncoder::from_ptr(
                    encoder_ptr as *mut metal::MTLComputeCommandEncoder,
                ))
            }
        }
    }

    /// Creates a blit command encoder to encode into this command buffer
    pub fn new_blit_command_encoder(&self) -> Option<BlitCommandEncoder> {
        unsafe {
            let encoder_ptr: *mut AnyObject = msg_send![self.0, blitCommandEncoder];
            if encoder_ptr.is_null() {
                None
            } else {
                Some(BlitCommandEncoder::from_ptr(
                    encoder_ptr as *mut metal::MTLBlitCommandEncoder,
                ))
            }
        }
    }

    /// Pushes a debug group onto the command buffer
    pub fn push_debug_group(&self, name: &str) {
        unsafe {
            let name_str = NSString::from_str(name);
            let _: () = msg_send![self.0, pushDebugGroup:name_str.as_raw_object()];
        }
    }

    /// Pops the current debug group from the command buffer
    pub fn pop_debug_group(&self) {
        unsafe {
            let _: () = msg_send![self.0, popDebugGroup];
        }
    }
}

impl fmt::Debug for MPSCommandBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MPSCommandBuffer")
            .field("label", &self.label())
            .field("status", &self.status())
            .finish()
    }
}

impl Drop for MPSCommandBuffer {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSCommandBuffer {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _);
                MPSCommandBuffer(obj)
            } else {
                MPSCommandBuffer(std::ptr::null_mut())
            }
        }
    }
}
