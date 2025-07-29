use super::ExecutionStage;
use metal::SharedEvent;
use metal::foreign_types::ForeignType;
use objc2::rc::{Allocated, Retained};
use objc2::runtime::NSObject;
use objc2::{ClassType, extern_class, msg_send};
use objc2_foundation::NSObjectProtocol;

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphExecutionDescriptor"]
    pub struct ExecutionDescriptor;
);

unsafe impl NSObjectProtocol for ExecutionDescriptor {}

impl ExecutionDescriptor {
    /// Create a new execution descriptor
    pub fn new() -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let allocated: Allocated<Self> = msg_send![class, alloc];
            let initialized: Retained<Self> = msg_send![allocated, init];
            initialized
        }
    }

    /// Set wait until completed flag
    pub fn set_wait_until_completed(&self, wait: bool) {
        unsafe {
            let _: () = msg_send![self, setWaitUntilCompleted: wait];
        }
    }

    /// Set the state of the execution descriptor to prefer synchronous execution
    pub fn prefer_synchronous_execution(&self) {
        self.set_wait_until_completed(true);
    }

    /// Set the state of the execution descriptor to prefer asynchronous execution
    pub fn prefer_asynchronous_execution(&self) {
        self.set_wait_until_completed(false);
    }

    /// Wait for a Metal shared event with a specific value before scheduling execution
    pub fn wait_for_event(&self, event: &SharedEvent, value: u64) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut std::ffi::c_void;
            let _: () = msg_send![self, waitForEvent: event_ptr, value: value];
        }
    }

    /// Signal a Metal shared event with a value at a specific execution stage
    pub fn signal_event(&self, event: &SharedEvent, execution_stage: ExecutionStage, value: u64) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut std::ffi::c_void;
            let _: () = msg_send![self, signalEvent: event_ptr, atExecutionEvent: execution_stage as u64, value: value];
        }
    }
}
