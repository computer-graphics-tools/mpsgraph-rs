use metal::{CommandQueue, foreign_types::ForeignType};
use objc2::rc::autoreleasepool;
use objc2::rc::{Allocated, Retained};
use objc2::runtime::NSObject;
use objc2::{ClassType, extern_class, extern_conformance, extern_methods, msg_send};
use objc2_foundation::{NSArray, NSObjectProtocol, NSString, NSURL};
use std::path::Path;

use crate::{
    CommandBuffer, CompilationDescriptor, Device, ExecutableExecutionDescriptor,
    ExecutableSerializationDescriptor, GraphObject, GraphOptions, GraphType, ShapedType, Tensor,
    TensorData,
};

extern_class!(
    /// The compiled representation of a compute graph executable.
    ///
    /// An `MPSGraphExecutable` is a compiled graph for specific feeds for specific target tensors and target operations.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphexecutable?language=objc)
    #[unsafe(super(GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[name = "MPSGraphExecutable"]
    pub struct Executable;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for Executable {}
);

impl Executable {
    extern_methods!(
        #[unsafe(method(init))]
        #[unsafe(method_family = init)]
        pub fn init(this: Allocated<Self>) -> Retained<Self>;

        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        pub fn new() -> Retained<Self>;

        /// Options for the graph executable.
        #[unsafe(method(options))]
        #[unsafe(method_family = none)]
        pub fn options(&self) -> GraphOptions;

        /// Setter for [`options`][Self::options].
        #[unsafe(method(setOptions:))]
        #[unsafe(method_family = none)]
        pub fn set_options(&self, options: GraphOptions);
    );
}

impl Executable {
    /// Tensors fed to the graph, can be used to order the inputs when executable is created with a graph.
    pub fn feed_tensors(&self) -> Option<Box<[Retained<Tensor>]>> {
        unsafe {
            let array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![self, feedTensors];
            array_opt.map(|array| array.to_vec().into_boxed_slice())
        }
    }

    /// Tensors targeted by the graph, can be used to order the outputs when executable was created with a graph.
    pub fn target_tensors(&self) -> Option<Box<[Retained<Tensor>]>> {
        unsafe {
            let array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![self, targetTensors];
            array_opt.map(|array| array.to_vec().into_boxed_slice())
        }
    }

    /// Specialize the executable and optimize it.
    ///
    /// Use this method to choose when specialization happens, else it occurs at encode time automatically.
    ///
    /// - Parameters:
    /// - device: optional device to compile with.
    /// - input_types: input types expected to be passed to the executable.
    /// - compilation_descriptor: compilation descriptor to be used to specialize, since the executable was created with a compilationDescriptor already this one overrides those settings to the extent it can.
    pub fn specialize(
        &self,
        device: Option<&Device>,
        input_types: &[&GraphType],
        compilation_descriptor: Option<&CompilationDescriptor>,
    ) {
        let input_types = NSArray::from_slice(input_types);
        unsafe {
            let _: () = msg_send![
                self,
                specializeWithDevice: device.as_deref(),
                inputTypes: &*input_types,
                compilationDescriptor: compilation_descriptor.as_deref()
            ];
        }
    }

    /// Get output shapes for a specialized executable.
    ///
    /// In case specialization has not been done yet then calling this function will specialize for the given input shapes.
    ///
    /// - Parameters:
    /// - device: Optional MPSGraph device to compile with
    /// - inputTypes: Input types expected to be passed to the executable.
    /// - compilationDescriptor: CompilationDescriptor to be used to specialize, since the executable was created with a compilationDescriptor already this one overrides those settings to the extent it can.
    pub fn get_output_types(
        &self,
        device: Option<&Device>,
        input_types: &[&GraphType],
        compilation_descriptor: Option<&CompilationDescriptor>,
    ) -> Option<Box<[Retained<ShapedType>]>> {
        let input_types = NSArray::from_slice(input_types);
        unsafe {
            let result: Option<Retained<NSArray<ShapedType>>> = msg_send![
                self,
                getOutputTypesWithDevice: device.as_deref(),
                inputTypes: &*input_types,
                compilationDescriptor: compilation_descriptor.as_deref()
            ];
            result.map(|r| r.to_vec().into_boxed_slice())
        }
    }

    /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
    ///
    /// This call is synchronous and will return on completion of execution.
    ///
    /// - Parameters:
    /// - command_queue: CommandQueue passed to exectute the graph on.
    /// - inputs: Feeds tensorData for the placeholder tensors, same order as arguments of main function.
    /// - results: Results tensorData for which the caller wishes TensorData to be returned.
    /// - execution_descriptor: ExecutionDescriptor to be passed in and used.
    /// - Returns: A valid TensorData array with results synchronized to the CPU memory if MPSGraphOptionsSynchronizeResults set.
    pub fn run_with_command_queue(
        &self,
        command_queue: &CommandQueue,
        inputs: &[&TensorData],
        results: Option<&[&TensorData]>,
        execution_descriptor: Option<&ExecutableExecutionDescriptor>,
    ) -> Box<[Retained<TensorData>]> {
        autoreleasepool(|_pool| unsafe {
            let cmd_queue_ptr = command_queue.as_ptr() as *mut std::ffi::c_void;
            let inputs_array = NSArray::from_slice(&inputs);
            let results_array = results.map(|r_slice| NSArray::from_slice(&r_slice));
            let result: Retained<NSArray<TensorData>> = msg_send![
                self,
                runWithMTLCommandQueue: cmd_queue_ptr,
                inputsArray: &*inputs_array,
                resultsArray: results_array.as_deref(),
                executionDescriptor: execution_descriptor.as_deref()
            ];
            result.to_vec().into_boxed_slice()
        })
    }

    /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
    /// This call is asynchronous and will return immediately.
    ///
    /// - Parameters:
    /// - command_queue: CommandQueue passed to exectute the graph on.
    /// - inputs: Feeds tensorData for the placeholder tensors, same order as arguments of main function.
    /// - results: Tensors for which the caller wishes TensorData to be returned.
    /// - execution_descriptor: ExecutionDescriptor to be passed in and used.
    /// - Returns: A valid TensorData array with results synchronized to the CPU memory if MPSGraphOptionsSynchronizeResults set.
    pub fn run_async_with_command_queue(
        &self,
        command_queue: &CommandQueue,
        inputs: &[&TensorData],
        results: Option<&[&TensorData]>,
        execution_descriptor: Option<&ExecutableExecutionDescriptor>,
    ) -> Box<[Retained<TensorData>]> {
        autoreleasepool(|_pool| unsafe {
            let cmd_queue_ptr = command_queue.as_ptr() as *mut std::ffi::c_void;
            let inputs_array = NSArray::from_slice(&inputs);
            let results_array = results.map(|r_slice| NSArray::from_slice(&r_slice));
            let result: Retained<NSArray<TensorData>> = msg_send![
                self,
                runAsyncWithMTLCommandQueue: cmd_queue_ptr,
                inputsArray: &*inputs_array,
                resultsArray: results_array.as_deref(),
                executionDescriptor: execution_descriptor.as_deref()
            ];
            result.to_vec().into_boxed_slice()
        })
    }

    /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
    /// This call is asynchronous and will return immediately after finishing encoding.
    ///
    /// - Parameters:
    /// - command_buffer: CommandBuffer passed to exectute the graph on, commitAndContinue might be called, please don't rely on underlying MTLCommandBuffer to remain uncommitted
    /// - inputs: Feeds tensorData for the placeholder tensors, same order as arguments of main function
    /// - results: Tensors for which the caller wishes TensorData to be returned
    /// - execution_descriptor: ExecutionDescriptor to be passed in and used,
    /// - Returns: A valid TensorData array with results synchronized to the CPU memory if MPSGraphOptionsSynchronizeResults set.
    pub fn encode_to_command_buffer(
        &self,
        command_buffer: &CommandBuffer,
        inputs: &[&TensorData],
        results: Option<&[&TensorData]>,
        execution_descriptor: Option<&ExecutableExecutionDescriptor>,
    ) -> Box<[Retained<TensorData>]> {
        autoreleasepool(|_pool| unsafe {
            let inputs_array = NSArray::from_slice(inputs);
            let results_array = results.map(|r_slice| NSArray::from_slice(r_slice));
            let result: Retained<NSArray<TensorData>> = msg_send![
                self,
                encodeToCommandBuffer: command_buffer,
                inputsArray: &*inputs_array,
                resultsArray: results_array.as_deref(),
                executionDescriptor: execution_descriptor.as_deref()
            ];
            result.to_vec().into_boxed_slice()
        })
    }

    /// Serialize the MPSGraph executable at the provided path.
    ///
    /// - Parameters:
    /// - path: The path where to serialize the MPSGraph executable.
    /// - descriptor: The descriptor to be used to serialize the graph.
    pub fn serialize_to_graph_package(
        &self,
        path: &Path,
        descriptor: Option<&ExecutableSerializationDescriptor>,
    ) {
        autoreleasepool(|_| unsafe {
            if let Some(path_str) = path.to_str() {
                let path_ns = NSString::from_str(path_str);
                let url = NSURL::fileURLWithPath(&path_ns);
                let _: () = msg_send![
                    self,
                    serializeToMPSGraphPackageAtURL: &*url,
                    descriptor: descriptor
                ];
            } else {
                eprintln!("Error: Could not convert path to string: {:?}", path);
            }
        })
    }

    /// Initialize the executable with the Metal Performance Shaders Graph package at the provided URL.
    ///
    /// - Parameters:
    /// - path: The path where to read the serialized MPSGraphExecutable.
    /// - compilation_descriptor: Compilation descriptor to be used to specialize, since the executable was created with a compilationDescriptor already this one overrides those settings to the extent it can.
    pub fn from_serialized_package(
        path: &Path,
        compilation_descriptor: Option<&CompilationDescriptor>,
    ) -> Option<Retained<Self>> {
        autoreleasepool(|_| unsafe {
            let path_str = path.to_str()?;
            let path_ns = NSString::from_str(path_str);
            let url = NSURL::fileURLWithPath(&path_ns);
            let class = Self::class();
            let allocated: Allocated<Self> = msg_send![class, alloc];
            msg_send![
                allocated,
                initWithMPSGraphPackageAtURL: &*url,
                compilationDescriptor: compilation_descriptor.as_deref()
            ]
        })
    }

    /// Initialize the executable with the Core ML model package at the provided URL.
    ///
    /// - Parameters:
    /// - path: The path where to read the Core ML model package.
    /// - compilation_descriptor: Compilation descriptor to be used to specialize, since the executable was created with a compilationDescriptor already this one overrides those settings to the extent it can.
    pub fn from_core_ml_package(
        path: &Path,
        compilation_descriptor: Option<&CompilationDescriptor>,
    ) -> Option<Retained<Self>> {
        autoreleasepool(|_| unsafe {
            let path_str = path.to_str()?;
            let path_ns = NSString::from_str(path_str);
            let url = NSURL::fileURLWithPath(&path_ns);
            let class = Self::class();
            let allocated: Allocated<Self> = msg_send![class, alloc];
            msg_send![
                allocated,
                initWithCoreMLPackageAtURL: &*url,
                compilationDescriptor: compilation_descriptor.as_deref()
            ]
        })
    }
}

impl Executable {
    pub fn dump(&self) {
        unsafe {
            let _: () = msg_send![self, dump];
        }
    }
}
