use super::{
    GraphOptions, RetainedTensorDataHashMap, TensorDataDictionary, TensorDataHashMap,
    TensorShapedTypeHashMap,
};
use crate::GraphObject;
use crate::command_buffer::CommandBuffer;
use crate::device::Device;
use crate::executable::{CompilationDescriptor, Executable, ExecutionDescriptor};
use crate::operation::Operation;
use crate::{NSDictionaryExt, ToNSDictionary};
use crate::{Tensor, TensorData};
use metal::{CommandQueue, foreign_types::ForeignType};
use objc2::rc::{Allocated, Retained, autoreleasepool};
use objc2::runtime::NSObject;
use objc2::{extern_class, extern_conformance, extern_methods, msg_send};
use objc2_foundation::{NSArray, NSMutableDictionary, NSObjectProtocol};

extern_class!(
    /// The optimized representation of a compute graph of operations and tensors.
    ///
    /// An MPSGraph is a symbolic representation of operations to be utilized to execute compute graphs on a device.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph?language=objc)
    #[unsafe(super(GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[name = "MPSGraph"]
    pub struct Graph;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for Graph {}
);

impl Graph {
    extern_methods!(
        /// Options for the graph.
        ///
        /// The default value is `MPSGraphOptionsDefault`.
        #[unsafe(method(options))]
        #[unsafe(method_family = none)]
        pub fn options(&self) -> GraphOptions;

        /// Setter for [`options`][Self::options].
        #[unsafe(method(setOptions:))]
        #[unsafe(method_family = none)]
        pub fn set_options(&self, options: GraphOptions);

        /// Creates a new graph to insert nodes in.
        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        pub fn new() -> Retained<Self>;

        /// Initialize an MPSGraph to insert nodes in.
        #[unsafe(method(init))]
        #[unsafe(method_family = init)]
        pub fn init(this: Allocated<Self>) -> Retained<Self>;
    );
}

impl Graph {
    /// Compiles the graph for the given feeds to returns the target tensor values, ensuring all target operations would be executed.
    ///
    /// This call blocks until execution has completed. The compilation descriptor helps specialize the executable returned.
    ///
    /// - Parameters:
    /// - device: Device to optimize for.
    /// - feeds: Feeds dictionary for the placeholder tensors.
    /// - targets: Tensors for which the caller wishes TensorData to be returned.
    /// - target_operations: Operations to be completed at the end of the run.
    /// - descriptor: compilation descriptor to set different compilation parameters.
    /// - Returns: A valid Executable object
    pub fn compile(
        &self,
        device: &Device,
        feeds: &TensorShapedTypeHashMap,
        targets: &[&Tensor],
        target_operations: Option<&[&Operation]>,
        descriptor: Option<&CompilationDescriptor>,
    ) -> Retained<Executable> {
        autoreleasepool(|_| unsafe {
            let feeds_dict = feeds.to_dictionary();
            let targets_array = NSArray::from_slice(targets);
            let target_operations_array = target_operations.map(|ops| NSArray::from_slice(ops));
            msg_send![
                self,
                compileWithDevice: device,
                feeds: &*feeds_dict,
                targetTensors: &*targets_array,
                targetOperations: target_operations_array.as_deref(),
                compilationDescriptor: descriptor.as_deref()
            ]
        })
    }

    /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
    ///
    /// This call blocks until execution has completed.
    ///
    /// - Parameters:
    /// - feeds: Feeds dictionary for the placeholder tensors.
    /// - targetTensors: Tensors for which the caller wishes MPSGraphTensorData to be returned.
    /// - targetOperations: Operations to be completed at the end of the run.
    /// - Returns: A valid MPSGraphTensor : MPSGraphTensorData dictionary with results synchronized to the CPU memory.
    pub fn run(
        &self,
        feeds: &TensorDataHashMap,
        targets: &[&Tensor],
        target_operations: Option<&[&Operation]>,
    ) -> RetainedTensorDataHashMap {
        autoreleasepool(|_| unsafe {
            let feeds_dict = feeds.to_dictionary();
            let targets_array = NSArray::from_slice(targets);
            let target_operations_array = target_operations.map(|ops| NSArray::from_slice(ops));
            let result: Retained<TensorDataDictionary> = msg_send![
                self,
                runWithFeeds: &*feeds_dict,
                targetTensors: &*targets_array,
                targetOperations: target_operations_array.as_deref(),
            ];
            result.to_hashmap()
        })
    }

    /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
    ///
    /// This call blocks until execution has completed.
    ///
    /// - Parameters:
    /// - command_queue: CommandQueue passed to exectute the graph on.
    /// - feeds: Feeds dictionary for the placeholder tensors.
    /// - target_tensors: Tensors for which the caller wishes MPSGraphTensorData to be returned.
    /// - target_operations: Operations to be completed at the end of the run.
    /// - Returns: A valid Tensors hashmap with results synchronized to the CPU memory.
    pub fn run_with_command_queue(
        &self,
        command_queue: &CommandQueue,
        feeds: &TensorDataHashMap,
        target_tensors: &[&Tensor],
        target_operations: Option<&[&Operation]>,
    ) -> RetainedTensorDataHashMap {
        autoreleasepool(|_| unsafe {
            let cmd_queue_ptr = command_queue.as_ptr() as *mut std::ffi::c_void;
            let feeds_dict = feeds.to_dictionary();
            let targets_array = NSArray::from_slice(target_tensors);
            let target_operations_array = target_operations.map(|ops| NSArray::from_slice(ops));
            let result: Retained<TensorDataDictionary> = msg_send![
                self,
                runWithMTLCommandQueue: cmd_queue_ptr,
                feeds: &*feeds_dict,
                targetTensors: &*targets_array,
                targetOperations: target_operations_array.as_deref(),
            ];
            result.to_hashmap()
        })
    }

    /// Runs the graph for the given feeds and returns the target tensor values in the results dictionary provided by the user.
    ///
    /// It also ensures all target operations also executed. This call blocks until execution has completed.
    ///
    /// - Parameters:
    /// - command_queue: CommandQueue passed to exectute the graph on.
    /// - feeds: Feeds dictionary for the placeholder tensors.
    /// - target_operations: Operations to be completed at the end of the run.
    /// - results: Tensors hashmap passed by user, these will be filled with graph output data.
    pub fn run_with_command_queue_in_place_results(
        &self,
        command_queue: &CommandQueue,
        feeds: &TensorDataHashMap,
        target_operations: Option<&[&Operation]>,
        results: &mut RetainedTensorDataHashMap,
    ) {
        autoreleasepool(|_| unsafe {
            let cmd_queue_ptr = command_queue.as_ptr() as *mut std::ffi::c_void;
            let feeds_dict = feeds.to_dictionary();
            let target_operations_array = target_operations.map(|ops| NSArray::from_slice(ops));
            let results_dict = NSMutableDictionary::<Tensor, TensorData>::new();
            let _: () = msg_send![
                self,
                runWithMTLCommandQueue: cmd_queue_ptr,
                feeds: &*feeds_dict,
                targetOperations: target_operations_array.as_deref(),
                resultsDictionary: &*results_dict,
            ];
            results.extend(results_dict.to_hashmap());
        })
    }

    /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
    ///
    /// This call is asynchronous and will return immediately if a completionHandler is set.
    ///
    /// - Parameters:
    /// - feeds: Feeds dictionary for the placeholder tensors.
    /// - target_tensors: Tensors for which the caller wishes TensorData to be returned.
    /// - target_operations: Operations to be completed at the end of the run.
    /// - execution_descriptor: ExecutionDescriptor to be passed in and used.
    /// - Returns: A valid Tensors hashmap with results synchronized to the CPU memory.
    pub fn run_async(
        &self,
        feeds: &TensorDataHashMap,
        target_tensors: &[&Tensor],
        target_operations: Option<&[&Operation]>,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) -> RetainedTensorDataHashMap {
        autoreleasepool(|_| unsafe {
            let feeds_dict = feeds.to_dictionary();
            let targets_array = NSArray::from_slice(target_tensors);
            let target_operations_array = target_operations.map(|ops| NSArray::from_slice(ops));
            let results: Retained<TensorDataDictionary> = msg_send![
                self,
                runAsyncWithFeeds: &*feeds_dict,
                targetTensors: &*targets_array,
                targetOperations: target_operations_array.as_deref(),
                executionDescriptor: execution_descriptor.as_deref()
            ];
            results.to_hashmap()
        })
    }

    /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
    ///
    /// This call is asynchronous and will return immediately if a completionHandler is set.
    ///
    /// - Parameters:
    /// - command_queue: CommandQueue passed to exectute the graph on.
    /// - feeds: Feeds dictionary for the placeholder tensors.
    /// - target_tensors: Tensors for which the caller wishes TensorData to be returned.
    /// - target_operations: Operations to be completed at the end of the run.
    /// - execution_descriptor: ExecutionDescriptor to be passed in and used.
    /// - Returns: A valid Tensors hashmap with results synchronized to the CPU memory.
    pub fn run_async_with_command_queue(
        &self,
        command_queue: &CommandQueue,
        feeds: &TensorDataHashMap,
        target_tensors: &[&Tensor],
        target_operations: Option<&[&Operation]>,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) -> RetainedTensorDataHashMap {
        autoreleasepool(|_| unsafe {
            let cmd_queue_ptr = command_queue.as_ptr() as *mut std::ffi::c_void;
            let feeds_dict = feeds.to_dictionary();
            let targets_array = NSArray::from_slice(target_tensors);
            let target_operations_array = target_operations.map(|ops| NSArray::from_slice(ops));
            let results: Retained<TensorDataDictionary> = msg_send![
                self,
                runAsyncWithMTLCommandQueue: cmd_queue_ptr,
                feeds: &*feeds_dict,
                targetTensors: &*targets_array,
                targetOperations: target_operations_array.as_deref(),
                executionDescriptor: execution_descriptor.as_deref()
            ];
            results.to_hashmap()
        })
    }

    /// Encodes the graph for the given feeds to returns the target tensor values in the results dictionary provided by the user.
    ///
    /// It ensures all target operations also executed. This call is asynchronous and will return immediately if a completionHandler is set.
    ///
    /// - Parameters:
    /// - command_queue: CommandQueue passed to exectute the graph on.
    /// - feeds: Feeds dictionary for the placeholder tensors.
    /// - target_operations: Operations to be completed at the end of the run.
    /// - results: Tensors hashmap passed by user, these will be filled with graph output data.
    /// - execution_descriptor: ExecutionDescriptor to be passed in and used.
    pub fn run_async_with_command_queue_in_place_results(
        &self,
        command_queue: &CommandQueue,
        feeds: &TensorDataHashMap,
        target_operations: Option<&[&Operation]>,
        results: &mut RetainedTensorDataHashMap,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) {
        autoreleasepool(|_| unsafe {
            let cmd_queue_ptr = command_queue.as_ptr() as *mut std::ffi::c_void;
            let feeds_dict = feeds.to_dictionary();
            let target_operations_array = target_operations.map(|ops| NSArray::from_slice(ops));
            let results_dict = NSMutableDictionary::<Tensor, TensorData>::new();
            let _: () = msg_send![
                self,
                runAsyncWithMTLCommandQueue: cmd_queue_ptr,
                feeds: &*feeds_dict,
                targetOperations: target_operations_array.as_deref(),
                resultsDictionary: &*results_dict,
                executionDescriptor: execution_descriptor.as_deref()
            ];
            results.extend(results_dict.to_hashmap());
        })
    }

    /// Encodes the graph for the given feeds to returns the target tensor values, ensuring all target operations also executed.
    ///
    /// This call is asynchronous and will return immediately if a completionHandler is set.
    ///
    /// - Parameters:
    /// - command_buffer: commandBuffer passed to exectute the graph on, it is an MPSCommandBuffer, commitAndContinue might be called, please don't rely on underlying MTLCommandBuffer to remain uncommitted.
    /// - feeds: Feeds dictionary for the placeholder tensors.
    /// - target_tensors: Tensors for which the caller wishes TensorData to be returned.
    /// - target_operations: Operations to be completed at the end of the run.
    /// - execution_descriptor: ExecutionDescriptor to be passed in and used.
    /// - Returns: A valid Tensors hashmap with results synchronized to the CPU memory.
    pub fn encode_to_command_buffer(
        &self,
        command_buffer: &CommandBuffer,
        feeds: &TensorDataHashMap,
        target_tensors: &[&Tensor],
        target_operations: Option<&[&Operation]>,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) -> RetainedTensorDataHashMap {
        autoreleasepool(|_| unsafe {
            let feeds_dict = feeds.to_dictionary();
            let targets_array = NSArray::from_slice(target_tensors);
            let target_operations_array = target_operations.map(|ops| NSArray::from_slice(ops));
            let results: Retained<TensorDataDictionary> = msg_send![
                self,
                encodeToCommandBuffer: command_buffer,
                feeds: &*feeds_dict,
                targetTensors: &*targets_array,
                targetOperations: target_operations_array.as_deref(),
                executionDescriptor: execution_descriptor.as_deref()
            ];
            results.to_hashmap()
        })
    }

    /// Encodes the graph for the given feeds to returns the target tensor values in the results dictionary provided by the user.
    ///
    /// It ensures all target operations also executed. This call is asynchronous and will return immediately if a completionHandler is set.
    ///
    /// - Parameters:
    /// - command_buffer: commandBuffer passed to execute the graph on, commitAndContinue might be called, please don't rely on underlying MTLCommandBuffer to remain uncommitted.
    /// - feeds: Feeds dictionary for the placeholder tensors.
    /// - target_operations: Operations to be completed at the end of the run.
    /// - results: Tensors hashmap passed by user, these will be filled with graph output data.
    /// - execution_descriptor: ExecutionDescriptor to be passed in and used.
    pub fn encode_to_command_buffer_in_place_results(
        &self,
        command_buffer: &CommandBuffer,
        feeds: &TensorDataHashMap,
        target_operations: Option<&[&Operation]>,
        results: &mut RetainedTensorDataHashMap,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) {
        autoreleasepool(|_| unsafe {
            let feeds_dict = feeds.to_dictionary();
            let target_operations_array = target_operations.map(|ops| NSArray::from_slice(ops));
            let results_dict = NSMutableDictionary::<Tensor, TensorData>::new();
            let _: () = msg_send![
                self,
                encodeToCommandBuffer: command_buffer,
                feeds: &*feeds_dict,
                targetOperations: target_operations_array.as_deref(),
                resultsDictionary: &*results_dict,
                executionDescriptor: execution_descriptor.as_deref()
            ];
            results.extend(results_dict.to_hashmap());
        })
    }
}

impl Graph {
    pub fn dump(&self) {
        unsafe {
            let _: () = msg_send![self, dump];
        }
    }
}
