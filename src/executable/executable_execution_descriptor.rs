use super::{ExecutableCompletionHandler, ExecutableScheduledHandler, ExecutionStage};
use crate::TensorData;
use block2::DynBlock;
use metal::SharedEvent;
use metal::foreign_types::ForeignType;
use objc2::rc::{Allocated, Retained};
use objc2::runtime::NSObject;
use objc2::{extern_class, extern_conformance, extern_methods, msg_send};
use objc2_foundation::{CopyingHelper, NSArray, NSCopying, NSError, NSObjectProtocol};
use std::ptr::NonNull;

use crate::GraphObject;

extern_class!(
    /// A class that consists of all the levers  to synchronize and schedule executable execution.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphexecutableexecutiondescriptor?language=objc)
    #[unsafe(super(GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[name = "MPSGraphExecutableExecutionDescriptor"]
    pub struct ExecutableExecutionDescriptor;
);

extern_conformance!(
    unsafe impl NSCopying for ExecutableExecutionDescriptor {}
);

unsafe impl CopyingHelper for ExecutableExecutionDescriptor {
    type Result = Self;
}

extern_conformance!(
    unsafe impl NSObjectProtocol for ExecutableExecutionDescriptor {}
);

impl ExecutableExecutionDescriptor {
    extern_methods!(
        #[unsafe(method(init))]
        #[unsafe(method_family = init)]
        pub fn init(this: Allocated<Self>) -> Retained<Self>;

        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        pub fn new() -> Retained<Self>;

        /// Flag for the graph executable to wait till the execution has completed.
        ///
        /// Default value is false.
        #[unsafe(method(waitUntilCompleted))]
        #[unsafe(method_family = none)]
        pub fn wait_until_completed(&self) -> bool;

        /// Setter for [`waitUntilCompleted`][Self::waitUntilCompleted].
        #[unsafe(method(setWaitUntilCompleted:))]
        #[unsafe(method_family = none)]
        pub fn set_wait_until_completed(&self, wait_until_completed: bool);
    );
}

impl ExecutableExecutionDescriptor {
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

    /// Getter for [`completionHandler`][Self::completionHandler].
    pub fn completion_handler(&self) -> ExecutableCompletionHandler {
        unsafe {
            let block_ptr: *mut DynBlock<dyn Fn(NonNull<NSArray<TensorData>>, *mut NSError)> =
                msg_send![self, completionHandler];
            ExecutableCompletionHandler::copy(block_ptr)
        }
    }

    /// Setter for [`completionHandler`][Self::completionHandler].
    pub fn set_completion_handler(&self, completion_handler: ExecutableCompletionHandler) {
        unsafe {
            let _: () = msg_send![self, setCompletionHandler: &*completion_handler];
        }
    }

    /// A notification that appears when graph-executable execution is scheduled.
    ///
    /// Default value is nil.
    pub fn scheduled_handler(&self) -> ExecutableScheduledHandler {
        unsafe {
            let block_ptr: *mut DynBlock<dyn Fn(NonNull<NSArray<TensorData>>, *mut NSError)> =
                msg_send![self, scheduledHandler];
            ExecutableScheduledHandler::copy(block_ptr)
        }
    }

    /// Setter for [`scheduledHandler`][Self::scheduledHandler].
    pub fn set_scheduled_handler(&self, scheduled_handler: ExecutableScheduledHandler) {
        unsafe {
            let _: () = msg_send![self, setScheduledHandler: &*scheduled_handler];
        }
    }
}

/// Private methods
impl ExecutableExecutionDescriptor {
    pub fn set_enable_commit_and_continue(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableCommitAndContinue: enable];
        }
    }

    pub fn set_simulate_ane_compile_failure(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setSimulateANECompileFailure: enable];
        }
    }

    pub fn set_simulate_ane_load_model_failure(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setSimulateANELoadModelFailure: enable];
        }
    }

    pub fn set_disable_synchronize_results(&self, disable: bool) {
        unsafe {
            let _: () = msg_send![self, setDisableSynchronizeResults: disable];
        }
    }

    pub fn set_disable_ane_caching(&self, disable: bool) {
        unsafe {
            let _: () = msg_send![self, setDisableANECaching: disable];
        }
    }

    pub fn set_disable_ane_fallback(&self, disable: bool) {
        unsafe {
            let _: () = msg_send![self, setDisableANEFallback: disable];
        }
    }

    pub fn set_enable_profiling_op_names(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableProfilingOpNames: enable];
        }
    }

    pub fn set_brief_profiling_op_names(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setBriefProfilingOpNames: enable];
        }
    }

    pub fn set_break_up_metal_encoders(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setBreakUpMetalEncoders: enable];
        }
    }

    pub fn set_generate_runtime_execution_report(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setGenerateRuntimeExecutionReport: enable];
        }
    }

    pub fn set_maximum_number_of_encoding_threads(&self, value: u64) {
        unsafe {
            let _: () = msg_send![self, setMaximumNumberOfEncodingThreads: value];
        }
    }

    pub fn number_of_commits_by_mps_graph(&self) -> u64 {
        unsafe { msg_send![self, numberOfCommitsByMPSGraph] }
    }
}
