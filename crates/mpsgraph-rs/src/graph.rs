#![allow(clippy::let_and_return)]

use crate::command_buffer::MPSCommandBuffer;
use crate::core::{AsRawObject, MPSDataType, MPSGraphOptions};
use crate::device::Device;
use crate::executable::{
    CompilationDescriptor, MPSGraphExecutable, ExecutionDescriptor,
};
use crate::operation::MPSGraphOperation;
use crate::shape::Shape;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use metal::foreign_types::ForeignType;
use metal::{CommandBuffer, CommandQueue, SharedEvent};
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::NSData;
use objc2_foundation::NSString;
use std::collections::HashMap;
use std::fmt;

/// Trait for scalar types that can be used in Graph operations
/// This trait is used for both single scalar values and arrays of values
pub trait TensorDataScalar: Copy {
    /// Convert a scalar value to f64 for use with Objective-C scalar methods
    fn to_f64(&self) -> f64;

    /// Convert a slice of values to an NSData object that can be used with Graph
    fn to_nsdata(values: &[Self]) -> Retained<NSData>;
}

// Implement for common numeric types
impl TensorDataScalar for f32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }

    fn to_nsdata(values: &[Self]) -> Retained<NSData> {
        unsafe {
            NSData::with_bytes(std::slice::from_raw_parts(
                values.as_ptr() as *const u8,
                std::mem::size_of_val(values),
            ))
        }
    }
}

impl TensorDataScalar for f64 {
    fn to_f64(&self) -> f64 {
        *self
    }

    fn to_nsdata(values: &[Self]) -> Retained<NSData> {
        unsafe {
            NSData::with_bytes(std::slice::from_raw_parts(
                values.as_ptr() as *const u8,
                std::mem::size_of_val(values),
            ))
        }
    }
}

impl TensorDataScalar for i32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }

    fn to_nsdata(values: &[Self]) -> Retained<NSData> {
        unsafe {
            NSData::with_bytes(std::slice::from_raw_parts(
                values.as_ptr() as *const u8,
                std::mem::size_of_val(values),
            ))
        }
    }
}

impl TensorDataScalar for i64 {
    fn to_f64(&self) -> f64 {
        *self as f64 // This may lose precision for large values
    }

    fn to_nsdata(values: &[Self]) -> Retained<NSData> {
        unsafe {
            NSData::with_bytes(std::slice::from_raw_parts(
                values.as_ptr() as *const u8,
                std::mem::size_of_val(values),
            ))
        }
    }
}

impl TensorDataScalar for u32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }

    fn to_nsdata(values: &[Self]) -> Retained<NSData> {
        unsafe {
            NSData::with_bytes(std::slice::from_raw_parts(
                values.as_ptr() as *const u8,
                std::mem::size_of_val(values),
            ))
        }
    }
}

impl TensorDataScalar for u64 {
    fn to_f64(&self) -> f64 {
        *self as f64 // This may lose precision for large values
    }

    fn to_nsdata(values: &[Self]) -> Retained<NSData> {
        unsafe {
            NSData::with_bytes(std::slice::from_raw_parts(
                values.as_ptr() as *const u8,
                std::mem::size_of_val(values),
            ))
        }
    }
}

#[link(name = "MetalPerformanceShadersGraph", kind = "framework")]
extern "C" {
    #[allow(dead_code)]
    fn MPSGraphCreate() -> *mut AnyObject;
}

/// A wrapper for MPSGraph objects
pub struct Graph(pub(crate) *mut AnyObject);

// Implement Send + Sync for the wrapper type
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// Creates a new Graph
    pub fn new() -> Self {
        unsafe {
            // Create without using external MPSGraphCreate function
            let class_name = c"MPSGraph";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            let obj: *mut AnyObject = msg_send![cls, alloc];
            let graph: *mut AnyObject = msg_send![obj, init];
            Graph(graph)
        }
    }

    /// Sets the options for this graph
    pub fn set_options(&self, options: MPSGraphOptions) {
        unsafe {
            let _: () = msg_send![self.0, setOptions:options as u64];
        }
    }

    /// Creates a placeholder tensor with dimensions
    pub fn placeholder_with_shape(
        &self,
        shape_dims: &[usize],
        data_type: MPSDataType,
        name: Option<&str>,
    ) -> Tensor {
        let shape = Shape::from_slice(shape_dims);
        self.placeholder(&shape, data_type, name)
    }

    /// Creates a placeholder tensor in the graph
    pub fn placeholder(
        &self,
        shape: &Shape,
        data_type: MPSDataType,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut::<AnyObject>(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, placeholderWithShape: shape.0,
                dataType:  data_type as u32,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

    // This is a convenience method not in the original API
    // Kept for compatibility with existing code
    /// Creates a constant tensor with given values and dimensions
    pub fn constant_with_shape<T: TensorDataScalar>(
        &self,
        values: &[T],
        shape_dims: &[usize],
        data_type: MPSDataType,
    ) -> Tensor {
        let shape = Shape::from_slice(shape_dims);
        self.constant(values, &shape, data_type)
    }

    /// Creates a constant tensor with given values and shape
    pub fn constant<T: TensorDataScalar>(
        &self,
        values: &[T],
        shape: &Shape,
        data_type: MPSDataType,
    ) -> Tensor {
        unsafe {
            // Create NSData with buffer values
            let data = T::to_nsdata(values);

            // Get raw NSData pointer for Objective-C
            let data_ptr: *mut AnyObject =
                data.as_ref() as *const objc2_foundation::NSData as *mut AnyObject;

            // Create constant tensor
            let tensor: *mut AnyObject = msg_send![
                self.0,
                constantWithData: data_ptr,
                shape: shape.0,
                dataType: data_type as u32
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

    /// Creates a constant with given scalar value (wraps constantWithScalar:dataType:)
    pub fn constant_scalar<T: TensorDataScalar>(
        &self,
        value: T,
        data_type: MPSDataType,
    ) -> Tensor {
        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                constantWithScalar: value.to_f64(),
                dataType: data_type as u32
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

    /// Creates a constant with given scalar value and shape (wraps constantWithScalar:shape:dataType:)
    pub fn constant_scalar_with_shape<T: TensorDataScalar>(
        &self,
        value: T,
        shape: &Shape,
        data_type: MPSDataType,
    ) -> Tensor {
        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                constantWithScalar: value.to_f64(),
                shape: shape.0,
                dataType: data_type as u32
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

    /// Creates a random uniform tensor with given shape
    pub fn random_uniform(
        &self,
        shape: &Shape,
        lower_bound: f32,
        upper_bound: f32,
        data_type: MPSDataType,
        seed: u32,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Create the random tensor
            let tensor: *mut AnyObject = msg_send![
                self.0,
                randomUniformTensorWithShape: shape.0,
                lowerBound: lower_bound as f64,
                upperBound: upper_bound as f64,
                dataType: data_type as u32,
                seed: seed,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

    /// Creates a random normal tensor with given shape
    pub fn random_normal(
        &self,
        shape: &Shape,
        mean: f32,
        stddev: f32,
        data_type: MPSDataType,
        seed: u32,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Create the random tensor
            let tensor: *mut AnyObject = msg_send![
                self.0,
                randomNormalTensorWithShape: shape.0,
                mean: mean as f64,
                standardDeviation: stddev as f64,
                dataType: data_type as u32,
                seed: seed,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

    /// Compiles the graph against a given set of feeds and targets
    pub fn compile(
        &self,
        device: &Device,
        feeds: &HashMap<Tensor, TensorData>,
        targets: &[Tensor],
        descriptor: Option<&CompilationDescriptor>,
    ) -> MPSGraphExecutable {
        unsafe {
            // Get the device pointer
            let device_obj = device.0;

            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());

            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }

            let feed_dict =
                crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);

            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter().map(|t| t.0).collect();

            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);

            // Get descriptor pointer if provided
            let descriptor_ptr = match descriptor {
                Some(desc) => desc.0,
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Compile the graph
            let executable: *mut AnyObject = msg_send![
                self.0,
                compileWithDevice: device_obj,
                feeds: feed_dict,
                targetTensors: targets_array,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
                compilationDescriptor: descriptor_ptr
            ];

            let executable = objc2::ffi::objc_retain(executable as *mut _);
            MPSGraphExecutable(executable)
        }
    }

    /// Compiles the graph against a given set of feeds, targets, and target operations
    pub fn compile_with_targets_and_ops(
        &self,
        device: &Device,
        feeds: &HashMap<Tensor, TensorData>,
        targets: &[Tensor],
        target_ops: &[MPSGraphOperation],
        descriptor: Option<&CompilationDescriptor>,
    ) -> MPSGraphExecutable {
        unsafe {
            // Get the device pointer
            let device_obj = device.0;

            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());

            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }

            let feed_dict =
                crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);

            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter().map(|t| t.0).collect();

            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);

            // Create ops array
            let ops_raw: Vec<*mut AnyObject> = target_ops.iter().map(|op| op.0).collect();

            let ops_array = crate::core::create_ns_array_from_pointers(&ops_raw);

            // Get descriptor pointer if provided
            let descriptor_ptr = match descriptor {
                Some(desc) => desc.0,
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Compile the graph
            let executable: *mut AnyObject = msg_send![
                self.0,
                compileWithDevice: device_obj,
                feeds: feed_dict,
                targetTensors: targets_array,
                targetOperations: ops_array,
                compilationDescriptor: descriptor_ptr
            ];

            let executable = objc2::ffi::objc_retain(executable as *mut _);
            MPSGraphExecutable(executable)
        }
    }

    /// Runs the graph synchronously against a given set of feeds and returns a result dictionary
    pub fn run_with_feeds(
        &self,
        feeds: &HashMap<Tensor, TensorData>,
        targets: &[Tensor],
    ) -> HashMap<Tensor, TensorData> {
        unsafe {
            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());

            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }

            let feed_dict =
                crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);

            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter().map(|t| t.0).collect();

            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);

            // Run the graph
            let results: *mut AnyObject = msg_send![
                self.0,
                runWithFeeds: feed_dict,
                targetTensors: targets_array,
                targetOperations: std::ptr::null_mut::<AnyObject>()
            ];

            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);

            // Don't release the results dictionary - caused crashes
            // // objc2::ffi::objc_release(results as *mut _);

            result_hash
        }
    }

    /// Runs the graph synchronously against a given set of feeds and returns a result dictionary
    /// with target operations specified
    pub fn run_with_feeds_and_ops(
        &self,
        feeds: &HashMap<Tensor, TensorData>,
        targets: &[Tensor],
        target_ops: &[MPSGraphOperation],
    ) -> HashMap<Tensor, TensorData> {
        unsafe {
            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());

            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }

            println!("DEBUG: Creating feed dictionary");
            let feed_dict =
                crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);

            // Create targets array
            println!("DEBUG: Creating targets array");
            let targets_raw: Vec<*mut AnyObject> = targets.iter().map(|t| t.0).collect();
            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);

            // Create ops array
            println!("DEBUG: Creating ops array");
            let ops_raw: Vec<*mut AnyObject> = target_ops.iter().map(|op| op.0).collect();
            let ops_array = crate::core::create_ns_array_from_pointers(&ops_raw);

            println!("DEBUG: About to run graph with feeds");

            // Run the graph - explicitly wrap just the Objective-C call in an autoreleasepool
            let results = objc2::rc::autoreleasepool(|_| {
                println!("DEBUG: Inside autoreleasepool for graph execution");
                let results: *mut AnyObject = msg_send![
                    self.0,
                    runWithFeeds: feed_dict,
                    targetTensors: targets_array,
                    targetOperations: ops_array
                ];
                println!("DEBUG: Graph run completed, results pointer: {:p}", results);
                results
            });

            // Convert the result dictionary to a Rust HashMap
            let result_hash = if !results.is_null() {
                println!("DEBUG: Converting results to HashMap");
                let hash = self.convert_dictionary_to_hash_map(results);

                // Don't release the results dictionary - caused crashes (since we got it from autoreleasepool)
                // objc2::ffi::objc_release(results as *mut _);
                println!("DEBUG: Released results dictionary");

                hash
            } else {
                println!("WARNING: Graph run returned null results!");
                HashMap::new()
            };

            println!("DEBUG: Returning result hash");
            result_hash
        }
    }

    /// Runs the graph synchronized against a given device
    pub fn run_with_feeds_on_device(
        &self,
        device: &Device,
        feeds: &HashMap<Tensor, TensorData>,
        targets: &[Tensor],
    ) -> HashMap<Tensor, TensorData> {
        unsafe {
            // Get the device pointer
            let device_obj = device.0;

            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());

            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }

            let feed_dict =
                crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);

            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter().map(|t| t.0).collect();

            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);

            // Run the graph
            let results: *mut AnyObject = msg_send![
                self.0,
                runWithFeeds: feed_dict,
                targetTensors: targets_array,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
                onDevice: device_obj
            ];

            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);

            // Don't release the results dictionary - caused crashes
            // // objc2::ffi::objc_release(results as *mut _);

            result_hash
        }
    }

    /// Runs the graph synchronously on a device with both target tensors and operations
    pub fn run_with_feeds_and_ops_on_device(
        &self,
        device: &Device,
        feeds: &HashMap<Tensor, TensorData>,
        targets: &[Tensor],
        target_ops: &[MPSGraphOperation],
    ) -> HashMap<Tensor, TensorData> {
        unsafe {
            // Get the device pointer
            let device_obj = device.0;

            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());

            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }

            let feed_dict =
                crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);

            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter().map(|t| t.0).collect();

            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);

            // Create ops array
            let ops_raw: Vec<*mut AnyObject> = target_ops.iter().map(|op| op.0).collect();

            let ops_array = crate::core::create_ns_array_from_pointers(&ops_raw);

            // Run the graph
            let results: *mut AnyObject = msg_send![
                self.0,
                runWithFeeds: feed_dict,
                targetTensors: targets_array,
                targetOperations: ops_array,
                onDevice: device_obj
            ];

            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);

            // Don't release the results dictionary - caused crashes
            // // objc2::ffi::objc_release(results as *mut _);

            result_hash
        }
    }

    /// Runs the graph asynchronously with feeds and returns the target tensor values
    ///  
    /// This call is asynchronous and will return immediately if a completionHandler is set
    /// in the execution descriptor.
    ///
    /// - Parameters:
    ///   - feeds: Feeds dictionary for the placeholder tensors
    ///   - target_tensors: Tensors for which the caller wishes TensorData to be returned
    ///   - target_operations: Operations to be completed at the end of the run
    ///   - execution_descriptor: ExecutionDescriptor to be passed in and used
    /// - Returns: A valid Tensor : TensorData dictionary with results
    pub fn run_async_with_feeds(
        &self,
        feeds: &HashMap<Tensor, TensorData>,
        target_tensors: &[Tensor],
        target_operations: Option<&[MPSGraphOperation]>,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) -> HashMap<Tensor, TensorData> {
        unsafe {
            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());

            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }

            let feed_dict =
                crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);

            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = target_tensors.iter().map(|t| t.0).collect();

            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);

            // Create operations array if provided
            let ops_array = match target_operations {
                Some(ops) => {
                    let ops_raw: Vec<*mut AnyObject> = ops.iter().map(|op| op.0).collect();
                    crate::core::create_ns_array_from_pointers(&ops_raw)
                }
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Get execution descriptor pointer if provided
            let descriptor_ptr = match execution_descriptor {
                Some(desc) => desc.0,
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Run the graph asynchronously
            let results: *mut AnyObject = msg_send![
                self.0,
                runAsyncWithFeeds: feed_dict,
                targetTensors: targets_array,
                targetOperations: ops_array,
                executionDescriptor: descriptor_ptr
            ];

            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);

            // Don't release the results dictionary - caused crashes
            // objc2::ffi::objc_release(results as *mut _);

            result_hash
        }
    }

    /// Runs the graph asynchronously on a command queue
    ///
    /// This call is asynchronous and will return immediately if a completionHandler is set
    /// in the execution descriptor.
    ///
    /// - Parameters:
    ///   - command_queue: CommandQueue passed to execute the graph on
    ///   - feeds: Feeds dictionary for the placeholder tensors
    ///   - target_tensors: Tensors for which the caller wishes TensorData to be returned
    ///   - target_operations: Operations to be completed at the end of the run
    ///   - execution_descriptor: ExecutionDescriptor to be passed in and used
    /// - Returns: A valid Tensor : TensorData dictionary with results
    pub fn run_async_with_command_queue(
        &self,
        command_queue: &CommandQueue,
        feeds: &HashMap<Tensor, TensorData>,
        target_tensors: &[Tensor],
        target_operations: Option<&[MPSGraphOperation]>,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) -> HashMap<Tensor, TensorData> {
        unsafe {
            // Get the command queue pointer
            let queue_ptr = command_queue.as_ptr() as *mut AnyObject;

            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());

            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }

            let feed_dict =
                crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);

            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = target_tensors.iter().map(|t| t.0).collect();

            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);

            // Create operations array if provided
            let ops_array = match target_operations {
                Some(ops) => {
                    let ops_raw: Vec<*mut AnyObject> = ops.iter().map(|op| op.0).collect();
                    crate::core::create_ns_array_from_pointers(&ops_raw)
                }
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Get execution descriptor pointer if provided
            let descriptor_ptr = match execution_descriptor {
                Some(desc) => desc.0,
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Run the graph asynchronously with command queue
            let results: *mut AnyObject = msg_send![
                self.0,
                runAsyncWithMTLCommandQueue: queue_ptr,
                feeds: feed_dict,
                targetTensors: targets_array,
                targetOperations: ops_array,
                executionDescriptor: descriptor_ptr
            ];

            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);

            // Don't release the results dictionary - caused crashes
            // objc2::ffi::objc_release(results as *mut _);

            result_hash
        }
    }

    /// Runs the graph asynchronously with a command queue and results dictionary
    ///
    /// This call is asynchronous and will return immediately if a completionHandler is set
    /// in the execution descriptor.
    ///
    /// - Parameters:
    ///   - command_queue: CommandQueue passed to execute the graph on
    ///   - feeds: Feeds dictionary for the placeholder tensors
    ///   - target_operations: Operations to be completed at the end of the run
    ///   - results_dict: Dictionary of tensors to receive the results
    ///   - execution_descriptor: ExecutionDescriptor to be passed in and used
    pub fn run_async_with_command_queue_results_dict(
        &self,
        command_queue: &CommandQueue,
        feeds: &HashMap<Tensor, TensorData>,
        target_operations: Option<&[MPSGraphOperation]>,
        results_dict: &HashMap<Tensor, TensorData>,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) {
        unsafe {
            // Get the command queue pointer
            let queue_ptr = command_queue.as_ptr() as *mut AnyObject;

            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());

            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }

            let feed_dict =
                crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);

            // Create the results dictionary
            let mut results_keys = Vec::with_capacity(results_dict.len());
            let mut results_values = Vec::with_capacity(results_dict.len());

            for (tensor, data) in results_dict {
                results_keys.push(tensor.0);
                results_values.push(data.0);
            }

            let results_dict_obj =
                crate::core::create_ns_dictionary_from_pointers(&results_keys, &results_values);

            // Create operations array if provided
            let ops_array = match target_operations {
                Some(ops) => {
                    let ops_raw: Vec<*mut AnyObject> = ops.iter().map(|op| op.0).collect();
                    crate::core::create_ns_array_from_pointers(&ops_raw)
                }
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Get execution descriptor pointer if provided
            let descriptor_ptr = match execution_descriptor {
                Some(desc) => desc.0,
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Run the graph asynchronously with supplied results dictionary
            let _: () = msg_send![
                self.0,
                runAsyncWithMTLCommandQueue: queue_ptr,
                feeds: feed_dict,
                targetOperations: ops_array,
                resultsDictionary: results_dict_obj,
                executionDescriptor: descriptor_ptr
            ];

            // Release dictionaries
            objc2::ffi::objc_release(feed_dict as *mut _);
            objc2::ffi::objc_release(results_dict_obj as *mut _);
        }
    }

    /// Encodes the graph to a command buffer for execution (MTLCommandBuffer version)
    ///
    /// This call is asynchronous and will return immediately if a completionHandler is set
    /// in the execution descriptor.
    ///
    /// - Parameters:
    ///   - command_buffer: CommandBuffer to encode the graph execution into
    ///   - feeds: Feeds dictionary for the placeholder tensors
    ///   - target_tensors: Tensors for which the caller wishes TensorData to be returned
    ///   - target_operations: Operations to be completed at the end of the run
    ///   - execution_descriptor: ExecutionDescriptor to be passed in and used
    /// - Returns: A valid Tensor : TensorData dictionary with results
    pub fn encode_to_metal_command_buffer(
        &self,
        command_buffer: &CommandBuffer,
        feeds: &HashMap<Tensor, TensorData>,
        target_tensors: &[Tensor],
        target_operations: Option<&[MPSGraphOperation]>,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) -> HashMap<Tensor, TensorData> {
        // Create MPSCommandBuffer from the Metal command buffer
        let mps_command_buffer = MPSCommandBuffer::from_command_buffer(command_buffer);

        // Call the MPSCommandBuffer version
        self.encode_to_command_buffer(
            &mps_command_buffer,
            feeds,
            target_tensors,
            target_operations,
            execution_descriptor,
        )
    }

    /// Encodes the graph to a command buffer with results dictionary (MTLCommandBuffer version)
    ///
    /// This call is asynchronous and will return immediately if a completionHandler is set
    /// in the execution descriptor.
    ///
    /// - Parameters:
    ///   - command_buffer: CommandBuffer to encode the graph execution into
    ///   - feeds: Feeds dictionary for the placeholder tensors
    ///   - target_operations: Operations to be completed at the end of the run
    ///   - results_dict: Dictionary of tensors to receive the results
    ///   - execution_descriptor: ExecutionDescriptor to be passed in and used
    pub fn encode_to_metal_command_buffer_with_results(
        &self,
        command_buffer: &CommandBuffer,
        feeds: &HashMap<Tensor, TensorData>,
        target_operations: Option<&[MPSGraphOperation]>,
        results_dict: &HashMap<Tensor, TensorData>,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) {
        // Create MPSCommandBuffer from the Metal command buffer
        let mps_command_buffer = MPSCommandBuffer::from_command_buffer(command_buffer);

        // Call the MPSCommandBuffer version
        self.encode_to_command_buffer_with_results(
            &mps_command_buffer,
            feeds,
            target_operations,
            results_dict,
            execution_descriptor,
        )
    }

    /// Encodes the graph to a MPSCommandBuffer for execution
    ///
    /// This call is asynchronous and will return immediately if a completionHandler is set
    /// in the execution descriptor.
    ///
    /// - Parameters:
    ///   - command_buffer: MPSCommandBuffer to encode the graph execution into
    ///   - feeds: Feeds dictionary for the placeholder tensors
    ///   - target_tensors: Tensors for which the caller wishes TensorData to be returned
    ///   - target_operations: Operations to be completed at the end of the run
    ///   - execution_descriptor: ExecutionDescriptor to be passed in and used
    /// - Returns: A valid Tensor : TensorData dictionary with results
    pub fn encode_to_command_buffer(
        &self,
        command_buffer: &MPSCommandBuffer,
        feeds: &HashMap<Tensor, TensorData>,
        target_tensors: &[Tensor],
        target_operations: Option<&[MPSGraphOperation]>,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) -> HashMap<Tensor, TensorData> {
        unsafe {
            // Get the command buffer pointer
            let buffer_ptr = command_buffer.0;

            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());

            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }

            let feed_dict =
                crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);

            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = target_tensors.iter().map(|t| t.0).collect();

            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);

            // Create operations array if provided
            let ops_array = match target_operations {
                Some(ops) => {
                    let ops_raw: Vec<*mut AnyObject> = ops.iter().map(|op| op.0).collect();
                    crate::core::create_ns_array_from_pointers(&ops_raw)
                }
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Get execution descriptor pointer if provided
            let descriptor_ptr = match execution_descriptor {
                Some(desc) => desc.0,
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Encode the graph to command buffer
            let results: *mut AnyObject = msg_send![
                self.0,
                encodeToCommandBuffer: buffer_ptr,
                feeds: feed_dict,
                targetTensors: targets_array,
                targetOperations: ops_array,
                executionDescriptor: descriptor_ptr
            ];

            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);

            // Don't release the results dictionary - caused crashes
            // objc2::ffi::objc_release(results as *mut _);

            result_hash
        }
    }

    /// Encodes the graph to a MPSCommandBuffer with results dictionary
    ///
    /// This call is asynchronous and will return immediately if a completionHandler is set
    /// in the execution descriptor.
    ///
    /// - Parameters:
    ///   - command_buffer: MPSCommandBuffer to encode the graph execution into
    ///   - feeds: Feeds dictionary for the placeholder tensors
    ///   - target_operations: Operations to be completed at the end of the run
    ///   - results_dict: Dictionary of tensors to receive the results
    ///   - execution_descriptor: ExecutionDescriptor to be passed in and used
    pub fn encode_to_command_buffer_with_results(
        &self,
        command_buffer: &MPSCommandBuffer,
        feeds: &HashMap<Tensor, TensorData>,
        target_operations: Option<&[MPSGraphOperation]>,
        results_dict: &HashMap<Tensor, TensorData>,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) {
        unsafe {
            // Get the command buffer pointer
            let buffer_ptr = command_buffer.0;

            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());

            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }

            let feed_dict =
                crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);

            // Create the results dictionary
            let mut results_keys = Vec::with_capacity(results_dict.len());
            let mut results_values = Vec::with_capacity(results_dict.len());

            for (tensor, data) in results_dict {
                results_keys.push(tensor.0);
                results_values.push(data.0);
            }

            let results_dict_obj =
                crate::core::create_ns_dictionary_from_pointers(&results_keys, &results_values);

            // Create operations array if provided
            let ops_array = match target_operations {
                Some(ops) => {
                    let ops_raw: Vec<*mut AnyObject> = ops.iter().map(|op| op.0).collect();
                    crate::core::create_ns_array_from_pointers(&ops_raw)
                }
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Get execution descriptor pointer if provided
            let descriptor_ptr = match execution_descriptor {
                Some(desc) => desc.0,
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Encode the graph to command buffer with results dictionary
            let _: () = msg_send![
                self.0,
                encodeToCommandBuffer: buffer_ptr,
                feeds: feed_dict,
                targetOperations: ops_array,
                resultsDictionary: results_dict_obj,
                executionDescriptor: descriptor_ptr
            ];

            // Release dictionaries
            objc2::ffi::objc_release(feed_dict as *mut _);
            objc2::ffi::objc_release(results_dict_obj as *mut _);
        }
    }

    /// Enqueue a graph run on a command queue
    pub fn encode_to_command_queue(
        &self,
        device: &Device,
        command_queue: &CommandQueue,
        feeds: &HashMap<Tensor, TensorData>,
        targets: &[Tensor],
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) -> HashMap<Tensor, TensorData> {
        unsafe {
            // Get the device pointer
            let _device_obj = device.0; // Keep reference for safety

            // Get the queue pointer
            let queue_ptr = command_queue.as_ptr() as *mut AnyObject;

            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());

            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }

            let feed_dict =
                crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);

            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter().map(|t| t.0).collect();

            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);

            // Get execution descriptor pointer if provided
            let descriptor_ptr = match execution_descriptor {
                Some(desc) => desc.0,
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Encode and run the graph
            let results: *mut AnyObject = msg_send![
                self.0,
                encodeToCommandQueueAndReturnError: queue_ptr,
                feeds: feed_dict,
                targetTensors: targets_array,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
                executionDescriptor: descriptor_ptr,
                error: std::ptr::null_mut::<AnyObject>()
            ];

            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);

            // Don't release the results dictionary - caused crashes
            // objc2::ffi::objc_release(results as *mut _);

            result_hash
        }
    }

    /// Encode and run a graph on a command buffer (legacy method)
    pub fn encode_to_command_buffer_legacy(
        &self,
        command_buffer: &CommandBuffer,
        feeds: &HashMap<Tensor, TensorData>,
        targets: &[Tensor],
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) -> HashMap<Tensor, TensorData> {
        unsafe {
            // Get the buffer pointer
            let buffer_ptr = command_buffer.as_ptr() as *mut AnyObject;

            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());

            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }

            let feed_dict =
                crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);

            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter().map(|t| t.0).collect();

            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);

            // Get execution descriptor pointer if provided
            let descriptor_ptr = match execution_descriptor {
                Some(desc) => desc.0,
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Encode and run the graph
            let results: *mut AnyObject = msg_send![
                self.0,
                encodeToCommandBuffer: buffer_ptr,
                feeds: feed_dict,
                targetTensors: targets_array,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
                executionDescriptor: descriptor_ptr
            ];

            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);

            // Don't release the results dictionary - caused crashes
            // objc2::ffi::objc_release(results as *mut _);

            result_hash
        }
    }

    /// Encodes and runs a graph on a command buffer with a completed event
    pub fn encode_to_command_buffer_with_event(
        &self,
        command_buffer: &CommandBuffer,
        feeds: &HashMap<Tensor, TensorData>,
        targets: &[Tensor],
        execution_descriptor: Option<&ExecutionDescriptor>,
        event: Option<&SharedEvent>,
    ) -> HashMap<Tensor, TensorData> {
        unsafe {
            // Get the buffer pointer
            let buffer_ptr = command_buffer.as_ptr() as *mut AnyObject;

            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());

            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }

            let feed_dict =
                crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);

            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter().map(|t| t.0).collect();

            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);

            // Get execution descriptor pointer if provided
            let descriptor_ptr = match execution_descriptor {
                Some(desc) => desc.0,
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Get event pointer if provided
            let event_ptr = match event {
                Some(evt) => evt.as_ptr() as *mut AnyObject,
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Encode and run the graph
            let results: *mut AnyObject = msg_send![
                self.0,
                encodeToCommandBuffer: buffer_ptr,
                feeds: feed_dict,
                targetTensors: targets_array,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
                executionDescriptor: descriptor_ptr,
                waitEvent: event_ptr,
                signalEvent: event_ptr,
                signalValue: 1
            ];

            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);

            // Don't release the results dictionary - caused crashes
            // objc2::ffi::objc_release(results as *mut _);

            result_hash
        }
    }

    /// Runs the graph with a command queue, feeds and outputs specified
    pub fn run_with_command_queue_feeds_outputs(
        &self,
        command_queue: &CommandQueue,
        feeds: HashMap<&Tensor, &TensorData>,
        results_dict: HashMap<&Tensor, &TensorData>,
        _execution_descriptor: Option<&ExecutionDescriptor>, // Ignored for now, not supported in this API
    ) {
        unsafe {
            // Get the queue pointer
            let queue_ptr = command_queue.as_ptr() as *mut AnyObject;

            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());

            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }

            let feed_dict =
                crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);

            // Create the results dictionary
            let mut results_keys = Vec::with_capacity(results_dict.len());
            let mut results_values = Vec::with_capacity(results_dict.len());

            for (tensor, data) in results_dict {
                results_keys.push(tensor.0);
                results_values.push(data.0);
            }

            let results_dict_obj =
                crate::core::create_ns_dictionary_from_pointers(&results_keys, &results_values);

            // Run the graph with both feeds and results dictionaries
            // Note: ExecutionDescriptor is not supported in this API method
            let _: () = msg_send![
                self.0,
                runWithMTLCommandQueue: queue_ptr,
                feeds: feed_dict,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
                resultsDictionary: results_dict_obj
            ];

            // Release dictionaries
            objc2::ffi::objc_release(feed_dict as *mut _);
            objc2::ffi::objc_release(results_dict_obj as *mut _);
        }
    }

    /// Helper method to convert an NSDictionary to a Rust HashMap
    fn convert_dictionary_to_hash_map(
        &self,
        dictionary: *mut AnyObject,
    ) -> HashMap<Tensor, TensorData> {
        // Wrap the entire operation in an autoreleasepool to ensure proper memory management
        objc2::rc::autoreleasepool(|_| {
            println!("DEBUG: Inside autoreleasepool for dictionary conversion");

            unsafe {
                if dictionary.is_null() {
                    println!("WARNING: Dictionary pointer is NULL!");
                    return HashMap::new();
                }

                println!("DEBUG: Converting dictionary: {:p}", dictionary);
                let mut result = HashMap::new();

                // Get an enumerator for the dictionary keys
                println!("DEBUG: Getting key enumerator");
                let enumerator: *mut AnyObject = msg_send![dictionary, keyEnumerator];

                if enumerator.is_null() {
                    println!("WARNING: Key enumerator is NULL!");
                    return HashMap::new();
                }

                println!("DEBUG: Got enumerator: {:p}", enumerator);

                // Use a mutable variable for the key outside the loop condition
                let mut key: *mut AnyObject;
                let mut entry_count = 0;

                println!("DEBUG: Starting dictionary enumeration");
                while {
                    // nextObject both advances the enumerator AND returns the current object
                    key = msg_send![enumerator, nextObject];
                    let has_next = !key.is_null();
                    if has_next {
                        println!("DEBUG: Got key: {:p} (entry {})", key, entry_count);
                        entry_count += 1;
                    } else {
                        println!("DEBUG: No more keys");
                    }
                    has_next
                } {
                    println!("DEBUG: Getting value for key: {:p}", key);
                    let value: *mut AnyObject = msg_send![dictionary, objectForKey: key];

                    if value.is_null() {
                        println!("WARNING: Value for key {:p} is NULL! Skipping entry.", key);
                        continue;
                    }

                    println!("DEBUG: Got value: {:p}", value);

                    // Create Rust wrappers WITHOUT retaining the objects, since we're in an autorelease pool
                    // and we've adjusted the tensor implementation to not release
                    let tensor = Tensor(key);
                    let tensor_data = TensorData(value);

                    println!(
                        "DEBUG: Created Rust wrappers: tensor={:p}, data={:p}",
                        key, value
                    );

                    // Add to the result HashMap
                    result.insert(tensor, tensor_data);
                    println!("DEBUG: Added entry to HashMap");
                }

                println!(
                    "DEBUG: Dictionary conversion complete with {} entries",
                    result.len()
                );
                result
            }
        })
    }
}

// Concatenation operations extension
impl Graph {
    /// Creates a concatenation operation
    pub fn concatenate(
        &self,
        tensors: &[Tensor],
        dimension: i64,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Create array of tensors
            let tensors_raw: Vec<*mut AnyObject> = tensors.iter().map(|t| t.0).collect();

            let tensors_array = crate::core::create_ns_array_from_pointers(&tensors_raw);

            let tensor: *mut AnyObject = msg_send![
                self.0,
                concatenationWithTensors: tensors_array,
                dimension: dimension,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }
}

// Matrix operations extension
impl Graph {
    /// Creates a transpose operation
    pub fn transpose(
        &self,
        x: &Tensor,
        dimensions: &[usize],
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut::<AnyObject>(),
            };

            // Create a shape object from the dimensions for convenience
            let dimensions_shape = Shape::from_slice(dimensions);

            // Create the operation
            let tensor: *mut AnyObject = msg_send![
                self.0,
                transposeTensor: x.0,
                permutation: dimensions_shape.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            // Convert to NSObject and release
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for Graph {
    fn clone(&self) -> Self {
        unsafe {
            // Retain and return new instance
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _);
                Graph(obj)
            } else {
                Graph(std::ptr::null_mut::<AnyObject>())
            }
        }
    }
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graph").finish()
    }
}

// Helper function to create NSArray from dimensions
#[allow(dead_code)]
fn create_dimensions(dimensions: &[usize]) -> *mut AnyObject {
    let shape = Shape::from_slice(dimensions);
    shape.0
}
