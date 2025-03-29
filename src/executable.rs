use objc2::runtime::AnyObject;
use objc2::msg_send;
use std::fmt;
use std::ptr;
use std::collections::HashMap;
use metal::foreign_types::ForeignType;
use crate::tensor::MPSGraphTensor;
use crate::tensor_data::MPSGraphTensorData;
use crate::core::{MPSGraphOptimization, MPSGraphOptimizationProfile, NSString, AsRawObject, MPSDataType};
use crate::command_buffer::MPSCommandBuffer;

/// A wrapper for MPSGraphExecutable objects
pub struct MPSGraphExecutable(pub(crate) *mut AnyObject);

// Implement Send + Sync for the wrapper type
unsafe impl Send for MPSGraphExecutable {}
unsafe impl Sync for MPSGraphExecutable {}

/// Result type for graph execution
pub type MPSGraphExecutionResult = HashMap<MPSGraphTensor, MPSGraphTensorData>;

impl MPSGraphExecutable {
    /// Create a new executable from a serialized package at the specified URL
    ///
    /// - Parameters:
    ///   - url: The URL string where the package is stored (file:// URL)
    ///   - compilation_descriptor: Optional compilation descriptor for specialization
    ///
    /// - Returns: A new executable instance, or None if creation failed
    pub fn from_serialized_package(url_string: &str, compilation_descriptor: Option<&MPSGraphCompilationDescriptor>) -> Option<Self> {
        unsafe {
            // Convert URL to NSURL
            let nsurl_class = objc2::runtime::AnyClass::get(c"NSURL").unwrap();
            let url_str = NSString::from_str(url_string);
            let nsurl: *mut AnyObject = msg_send![nsurl_class, URLWithString: url_str.as_raw_object()];
            
            if nsurl.is_null() {
                return None;
            }
            
            // Get the MPSGraphExecutable class
            let class_name = c"MPSGraphExecutable";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            
            // Initialize from URL
            let obj: *mut AnyObject = msg_send![cls, alloc];
            
            // Get compilation descriptor pointer or null
            let descriptor_ptr = if let Some(desc) = compilation_descriptor {
                desc.0
            } else {
                std::ptr::null_mut()
            };
            
            let executable: *mut AnyObject = msg_send![obj, initWithMPSGraphPackageAtURL: nsurl, 
                                                     compilationDescriptor: descriptor_ptr];
            
            // Release NSURL as we don't need it anymore
            objc2::ffi::objc_release(nsurl as *mut _);
            
            if executable.is_null() {
                return None;
            }
            
            Some(MPSGraphExecutable(executable))
        }
    }
    
    /// Create a new executable from a CoreML model package at the specified URL
    /// 
    /// This functionality is available in iOS 18/macOS 15 and newer.
    ///
    /// - Parameters:
    ///   - url: The URL string where the CoreML model package is stored (file:// URL)
    ///   - compilation_descriptor: Optional compilation descriptor for specialization
    ///
    /// - Returns: A new executable instance, or None if creation failed
    pub fn from_coreml_package(url_string: &str, compilation_descriptor: Option<&MPSGraphCompilationDescriptor>) -> Option<Self> {
        unsafe {
            // Convert URL to NSURL
            let nsurl_class = objc2::runtime::AnyClass::get(c"NSURL").unwrap();
            let url_str = NSString::from_str(url_string);
            let nsurl: *mut AnyObject = msg_send![nsurl_class, URLWithString: url_str.as_raw_object()];
            
            if nsurl.is_null() {
                return None;
            }
            
            // Get the MPSGraphExecutable class
            let class_name = c"MPSGraphExecutable";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            
            // Initialize from URL
            let obj: *mut AnyObject = msg_send![cls, alloc];
            
            // Get compilation descriptor pointer or null
            let descriptor_ptr = if let Some(desc) = compilation_descriptor {
                desc.0
            } else {
                std::ptr::null_mut()
            };
            
            let executable: *mut AnyObject = msg_send![obj, initWithCoreMLPackageAtURL: nsurl, 
                                                     compilationDescriptor: descriptor_ptr];
            
            // Release NSURL as we don't need it anymore
            objc2::ffi::objc_release(nsurl as *mut _);
            
            if executable.is_null() {
                return None;
            }
            
            Some(MPSGraphExecutable(executable))
        }
    }
    
    /// Execute the graph on a device
    pub fn run_with_feeds(&self, feeds: &HashMap<MPSGraphTensor, MPSGraphTensorData>, output_tensors: &[MPSGraphTensor]) -> MPSGraphExecutionResult {
        unsafe {
            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);
            
            // Create output tensors array
            let output_tensors_raw: Vec<*mut AnyObject> = output_tensors.iter()
                .map(|t| t.0)
                .collect();
            
            let output_tensors_array = crate::core::create_ns_array_from_pointers(&output_tensors_raw);
            
            // Run the executable
            let results: *mut AnyObject = msg_send![self.0, runWithFeeds: feed_dict,
                outputTensors: output_tensors_array,
                executionDescriptor: std::ptr::null_mut::<AnyObject>(),
            ];
            
            // Convert the result dictionary to a Rust HashMap
            let result_hash = convert_dictionary_to_hash_map(results);
            
            // Release the results dictionary
            objc2::ffi::objc_release(results as *mut _);
            
            result_hash
        }
    }
    
    /// Execute the graph on a device with execution descriptor
    pub fn run_with_feeds_and_descriptor(&self, feeds: &HashMap<MPSGraphTensor, MPSGraphTensorData>, output_tensors: &[MPSGraphTensor], execution_descriptor: &MPSGraphExecutionDescriptor) -> MPSGraphExecutionResult {
        unsafe {
            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);
            
            // Create output tensors array
            let output_tensors_raw: Vec<*mut AnyObject> = output_tensors.iter()
                .map(|t| t.0)
                .collect();
            
            let output_tensors_array = crate::core::create_ns_array_from_pointers(&output_tensors_raw);
            
            // Run the executable
            let results: *mut AnyObject = msg_send![self.0, runWithFeeds: feed_dict,
                outputTensors: output_tensors_array,
                executionDescriptor: execution_descriptor.0,
            ];
            
            // Convert the result dictionary to a Rust HashMap
            let result_hash = convert_dictionary_to_hash_map(results);
            
            // Release the results dictionary
            objc2::ffi::objc_release(results as *mut _);
            
            result_hash
        }
    }
    
    /// Execute the graph asynchronously on a command queue
    ///
    /// This method runs the executable asynchronously and returns immediately.
    /// When execution completes, the completion handler will be called.
    ///
    /// - Parameters:
    ///   - command_queue: The Metal command queue to use for execution
    ///   - feeds: A dictionary mapping input tensors to their values
    ///   - output_tensors: An array of tensors whose values should be computed
    ///   - execution_descriptor: Descriptor controlling execution options
    ///   - completion_handler: A callback to be invoked when execution completes
    pub fn run_async_with_command_queue<F>(
        &self, 
        command_queue: &metal::CommandQueue,
        feeds: &HashMap<MPSGraphTensor, MPSGraphTensorData>, 
        output_tensors: &[MPSGraphTensor],
        execution_descriptor: &MPSGraphExecutionDescriptor,
        completion_handler: Option<F>
    ) -> MPSGraphExecutionResult 
    where F: FnOnce(MPSGraphExecutionResult) + 'static {
        unsafe {
            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);
            
            // Create output tensors array
            let output_tensors_raw: Vec<*mut AnyObject> = output_tensors.iter()
                .map(|t| t.0)
                .collect();
            
            let output_tensors_array = crate::core::create_ns_array_from_pointers(&output_tensors_raw);
            
            // Get the command queue pointer
            let command_queue_ptr = command_queue.as_ptr() as *mut AnyObject;
            
            // Run the executable
            let results: *mut AnyObject;
            
            if let Some(callback) = completion_handler {
                // For now, we'll use the synchronous version since true Objective-C blocks support 
                // is complex to implement correctly with the current crate versions.
                results = msg_send![self.0, runAsyncWithMTLCommandQueue: command_queue_ptr,
                    feeds: feed_dict,
                    outputTensors: output_tensors_array,
                    executionDescriptor: execution_descriptor.0,
                ];
                
                // Get the results and call the callback directly
                let result_hash = convert_dictionary_to_hash_map(results);
                
                // Execute callback synchronously
                callback(result_hash.clone());
            } else {
                // No completion handler provided
                results = msg_send![self.0, runAsyncWithMTLCommandQueue: command_queue_ptr,
                    feeds: feed_dict,
                    outputTensors: output_tensors_array,
                    executionDescriptor: execution_descriptor.0,
                ];
            }
            
            // Convert the result dictionary to a Rust HashMap
            let result_hash = convert_dictionary_to_hash_map(results);
            
            // Release the results dictionary
            objc2::ffi::objc_release(results as *mut _);
            
            result_hash
        }
    }
    
    /// Execute the executable with array-based inputs and outputs
    ///
    /// This method is an alternative to the dictionary-based API, allowing execution
    /// with parallel arrays of input tensors and values.
    ///
    /// - Parameters:
    ///   - input_tensors: Array of input tensors
    ///   - input_values: Array of input tensor data (must match the order of input_tensors)
    ///   - output_tensors: Array of output tensors to compute
    ///   - execution_descriptor: Optional descriptor controlling execution
    ///
    /// - Returns: Array of output tensor data in the same order as output_tensors
    pub fn run_with_inputs_outputs(
        &self,
        input_tensors: &[MPSGraphTensor],
        input_values: &[MPSGraphTensorData],
        output_tensors: &[MPSGraphTensor],
        execution_descriptor: Option<&MPSGraphExecutableExecutionDescriptor>
    ) -> Vec<MPSGraphTensorData> {
        assert_eq!(input_tensors.len(), input_values.len(), 
                   "Input tensors and values must have the same length");
        
        unsafe {
            // Create input tensors array
            let input_tensors_raw: Vec<*mut AnyObject> = input_tensors.iter()
                .map(|t| t.0)
                .collect();
            
            let input_tensors_array = crate::core::create_ns_array_from_pointers(&input_tensors_raw);
            
            // Create input values array
            let input_values_raw: Vec<*mut AnyObject> = input_values.iter()
                .map(|d| d.0)
                .collect();
            
            let input_values_array = crate::core::create_ns_array_from_pointers(&input_values_raw);
            
            // Create output tensors array
            let output_tensors_raw: Vec<*mut AnyObject> = output_tensors.iter()
                .map(|t| t.0)
                .collect();
            
            let output_tensors_array = crate::core::create_ns_array_from_pointers(&output_tensors_raw);
            
            // Execute with arrays
            let descriptor_ptr = if let Some(desc) = execution_descriptor {
                desc.0
            } else {
                std::ptr::null_mut()
            };
            
            let results: *mut AnyObject = msg_send![self.0, 
                runWithInputTensors: input_tensors_array,
                inputValues: input_values_array,
                outputTensors: output_tensors_array,
                executionDescriptor: descriptor_ptr
            ];
            
            // Convert NSArray of results to Vec<MPSGraphTensorData>
            let count: usize = msg_send![results, count];
            let mut result_vec = Vec::with_capacity(count);
            
            for i in 0..count {
                let tensor_data: *mut AnyObject = msg_send![results, objectAtIndex: i];
                objc2::ffi::objc_retain(tensor_data as *mut _);
                result_vec.push(MPSGraphTensorData(tensor_data));
            }
            
            // Release the results array
            objc2::ffi::objc_release(results as *mut _);
            
            result_vec
        }
    }
    
    /// Run the executable with a command queue, providing explicit input and output
    /// operations
    ///
    /// - Parameters:
    ///   - command_queue: The Metal command queue to use for execution
    ///   - input_operations: Array of input operations
    ///   - input_data: Array of input tensor data (must match the order of input_operations)
    ///   - output_operations: Array of output operations to compute
    ///   - execution_descriptor: Optional descriptor controlling execution
    ///
    /// - Returns: Array of output tensor data in the same order as output_operations
    pub fn run_with_operations_on_command_queue(
        &self,
        command_queue: &metal::CommandQueue,
        input_operations: &[crate::operation::MPSGraphOperation],
        input_data: &[MPSGraphTensorData],
        output_operations: &[crate::operation::MPSGraphOperation],
        execution_descriptor: Option<&MPSGraphExecutableExecutionDescriptor>
    ) -> Vec<MPSGraphTensorData> {
        assert_eq!(input_operations.len(), input_data.len(), 
                   "Input operations and data must have the same length");
        
        unsafe {
            // Convert command queue to pointer
            let command_queue_ptr = command_queue.as_ptr() as *mut AnyObject;
            
            // Create input operations array
            let input_ops_raw: Vec<*mut AnyObject> = input_operations.iter()
                .map(|op| op.0)
                .collect();
            
            let input_ops_array = crate::core::create_ns_array_from_pointers(&input_ops_raw);
            
            // Create input data array
            let input_data_raw: Vec<*mut AnyObject> = input_data.iter()
                .map(|d| d.0)
                .collect();
            
            let input_data_array = crate::core::create_ns_array_from_pointers(&input_data_raw);
            
            // Create output operations array
            let output_ops_raw: Vec<*mut AnyObject> = output_operations.iter()
                .map(|op| op.0)
                .collect();
            
            let output_ops_array = crate::core::create_ns_array_from_pointers(&output_ops_raw);
            
            // Execute with operations
            let descriptor_ptr = if let Some(desc) = execution_descriptor {
                desc.0
            } else {
                std::ptr::null_mut()
            };
            
            let results: *mut AnyObject = msg_send![self.0, 
                runWithMTLCommandQueue: command_queue_ptr,
                inputOperations: input_ops_array,
                inputDataArray: input_data_array,
                outputOperations: output_ops_array,
                executionDescriptor: descriptor_ptr
            ];
            
            // Convert NSArray of results to Vec<MPSGraphTensorData>
            let count: usize = msg_send![results, count];
            let mut result_vec = Vec::with_capacity(count);
            
            for i in 0..count {
                let tensor_data: *mut AnyObject = msg_send![results, objectAtIndex: i];
                objc2::ffi::objc_retain(tensor_data as *mut _);
                result_vec.push(MPSGraphTensorData(tensor_data));
            }
            
            // Release the results array
            objc2::ffi::objc_release(results as *mut _);
            
            result_vec
        }
    }
    
    /// Encode the executable to a Metal command buffer
    ///
    /// This is a compatibility method that creates an MPSCommandBuffer from the Metal command buffer
    /// and then calls the MPSCommandBuffer version of the method.
    /// 
    /// - Parameters:
    ///   - command_buffer: The Metal command buffer to encode to
    ///   - inputs: Array of input tensor data
    ///   - results: Array of output tensor data (may be nil)
    ///   - execution_descriptor: Optional descriptor controlling execution
    ///
    /// - Returns: Array of output tensor data in the same order as the results parameter
    pub fn encode_to_metal_command_buffer(
        &self,
        command_buffer: &metal::CommandBuffer,
        inputs: &[MPSGraphTensorData],
        results: Option<&[MPSGraphTensorData]>,
        execution_descriptor: Option<&MPSGraphExecutableExecutionDescriptor>
    ) -> Vec<MPSGraphTensorData> {
        // Create an MPSCommandBuffer from the Metal command buffer
        let mps_command_buffer = MPSCommandBuffer::from_command_buffer(command_buffer);
        
        // Call the MPSCommandBuffer version
        self.encode_to_command_buffer(&mps_command_buffer, inputs, results, execution_descriptor)
    }
    
    /// Encode the executable to an MPSCommandBuffer
    ///
    /// - Parameters:
    ///   - command_buffer: The MPSCommandBuffer to encode to
    ///   - inputs: Array of input tensor data
    ///   - results: Array of output tensor data (may be nil)
    ///   - execution_descriptor: Optional descriptor controlling execution
    ///
    /// - Returns: Array of output tensor data in the same order as the results parameter
    pub fn encode_to_command_buffer(
        &self,
        command_buffer: &MPSCommandBuffer,
        inputs: &[MPSGraphTensorData],
        results: Option<&[MPSGraphTensorData]>,
        execution_descriptor: Option<&MPSGraphExecutableExecutionDescriptor>
    ) -> Vec<MPSGraphTensorData> {
        unsafe {
            // Create input values array
            let input_values_raw: Vec<*mut AnyObject> = inputs.iter()
                .map(|d| d.0)
                .collect();
            
            let input_values_array = crate::core::create_ns_array_from_pointers(&input_values_raw);
            
            // Create results array if provided
            let results_array = match results {
                Some(results_vec) => {
                    let results_raw: Vec<*mut AnyObject> = results_vec.iter()
                        .map(|d| d.0)
                        .collect();
                    
                    crate::core::create_ns_array_from_pointers(&results_raw)
                },
                None => std::ptr::null_mut()
            };
            
            // Get execution descriptor pointer if provided
            let descriptor_ptr = match execution_descriptor {
                Some(desc) => desc.0,
                None => std::ptr::null_mut()
            };
            
            // Encode to command buffer
            let result_array: *mut AnyObject = msg_send![self.0, 
                encodeToCommandBuffer: command_buffer.0,
                inputsArray: input_values_array,
                resultsArray: results_array,
                executionDescriptor: descriptor_ptr
            ];
            
            // Convert NSArray of results to Vec<MPSGraphTensorData>
            let count: usize = msg_send![result_array, count];
            let mut result_vec = Vec::with_capacity(count);
            
            for i in 0..count {
                let tensor_data: *mut AnyObject = msg_send![result_array, objectAtIndex: i];
                objc2::ffi::objc_retain(tensor_data as *mut _);
                result_vec.push(MPSGraphTensorData(tensor_data));
            }
            
            // Release the results array
            objc2::ffi::objc_release(result_array as *mut _);
            
            result_vec
        }
    }
    
    /// Specialize the executable for specific input tensor shapes and data types
    ///
    /// This optimizes the executable for the given input tensor shapes and data types,
    /// which can improve performance for subsequent executions.
    ///
    /// - Parameters:
    ///   - device: The device to specialize for
    ///   - tensor_shapes: A dictionary mapping tensors to their shapes
    ///   - tensor_data_types: A dictionary mapping tensors to their data types
    ///
    /// - Returns: A new specialized executable
    pub fn specialize_with_device(
        &self,
        device: &crate::device::MPSGraphDevice,
        tensor_shapes: &HashMap<MPSGraphTensor, crate::shape::MPSShape>,
        tensor_data_types: &HashMap<MPSGraphTensor, MPSDataType>
    ) -> Option<Self> {
        unsafe {
            // Convert tensor_shapes to NSDictionary
            let mut shape_keys = Vec::with_capacity(tensor_shapes.len());
            let mut shape_values = Vec::with_capacity(tensor_shapes.len());
            
            for (tensor, shape) in tensor_shapes {
                shape_keys.push(tensor.0);
                shape_values.push(shape.0);
            }
            
            let shapes_dict = crate::core::create_ns_dictionary_from_pointers(&shape_keys, &shape_values);
            
            // Convert tensor_data_types to NSDictionary
            let mut type_keys = Vec::with_capacity(tensor_data_types.len());
            let mut type_values = Vec::new();
            
            for (tensor, data_type) in tensor_data_types {
                type_keys.push(tensor.0);
                
                // NSNumber with the data type value
                let ns_number_class = objc2::runtime::AnyClass::get(c"NSNumber").unwrap();
                let data_type_value = data_type.as_u32() as u64;
                let number: *mut AnyObject = msg_send![ns_number_class, numberWithUnsignedInteger: data_type_value];
                
                type_values.push(number);
            }
            
            let types_dict = crate::core::create_ns_dictionary_from_pointers(&type_keys, &type_values);
            
            // Call specializeWithDevice method
            let device_ptr = device.0;
            let specialized: *mut AnyObject = msg_send![self.0,
                specializeWithDevice: device_ptr,
                tensorShapesDescriptorDictionary: shapes_dict,
                tensorDataTypesDictionary: types_dict
            ];
            
            // Release the dictionaries
            objc2::ffi::objc_release(shapes_dict as *mut _);
            objc2::ffi::objc_release(types_dict as *mut _);
            
            if specialized.is_null() {
                return None;
            }
            
            Some(MPSGraphExecutable(specialized))
        }
    }
    
    /// Get output tensor data types for this executable
    ///
    /// This method returns the data types of the output tensors that would be produced
    /// when running this executable with the given input tensor shapes and data types.
    ///
    /// - Parameters:
    ///   - device: The device to get types for
    ///   - feed_tensor_shapes: A dictionary mapping feed tensors to their shapes
    ///   - feed_tensor_data_types: A dictionary mapping feed tensors to their data types
    ///
    /// - Returns: A dictionary mapping output tensors to their data types, or None if the operation fails
    pub fn get_output_types_with_device(
        &self,
        device: &crate::device::MPSGraphDevice,
        feed_tensor_shapes: &HashMap<MPSGraphTensor, crate::shape::MPSShape>,
        feed_tensor_data_types: &HashMap<MPSGraphTensor, MPSDataType>
    ) -> Option<HashMap<MPSGraphTensor, MPSDataType>> {
        unsafe {
            // Convert feed_tensor_shapes to NSDictionary
            let mut shape_keys = Vec::with_capacity(feed_tensor_shapes.len());
            let mut shape_values = Vec::with_capacity(feed_tensor_shapes.len());
            
            for (tensor, shape) in feed_tensor_shapes {
                shape_keys.push(tensor.0);
                shape_values.push(shape.0);
            }
            
            let shapes_dict = crate::core::create_ns_dictionary_from_pointers(&shape_keys, &shape_values);
            
            // Convert feed_tensor_data_types to NSDictionary
            let mut type_keys = Vec::with_capacity(feed_tensor_data_types.len());
            let mut type_values = Vec::new();
            
            for (tensor, data_type) in feed_tensor_data_types {
                type_keys.push(tensor.0);
                
                // NSNumber with the data type value
                let ns_number_class = objc2::runtime::AnyClass::get(c"NSNumber").unwrap();
                let data_type_value = data_type.as_u32() as u64;
                let number: *mut AnyObject = msg_send![ns_number_class, numberWithUnsignedInteger: data_type_value];
                
                type_values.push(number);
            }
            
            let types_dict = crate::core::create_ns_dictionary_from_pointers(&type_keys, &type_values);
            
            // Call getOutputTypesWithDevice method
            let device_ptr = device.0;
            let output_types_dict: *mut AnyObject = msg_send![self.0,
                getOutputTypesWithDevice: device_ptr,
                feedTensorShapesDictionary: shapes_dict,
                feedTensorDataTypesDictionary: types_dict
            ];
            
            // Release the input dictionaries
            objc2::ffi::objc_release(shapes_dict as *mut _);
            objc2::ffi::objc_release(types_dict as *mut _);
            
            if output_types_dict.is_null() {
                return None;
            }
            
            // Convert the output types dictionary to a Rust HashMap
            let mut result = HashMap::new();
            
            // Get an enumerator for the dictionary keys
            let enumerator: *mut AnyObject = msg_send![output_types_dict, keyEnumerator];
            
            while {
                let key: *mut AnyObject = msg_send![enumerator, nextObject];
                !key.is_null()
            } {
                let key: *mut AnyObject = msg_send![enumerator, currentObject];
                let value: *mut AnyObject = msg_send![output_types_dict, objectForKey: key];
                
                // Retain the key to avoid it being deallocated
                objc2::ffi::objc_retain(key as *mut _);
                
                // Create Tensor wrapper
                let tensor = MPSGraphTensor(key);
                
                // Extract data type from NSNumber
                let data_type_value: u64 = msg_send![value, unsignedIntegerValue];
                let data_type = MPSDataType::from_u32(data_type_value as u32);
                
                // Add to the result HashMap
                result.insert(tensor, data_type);
            }
            
            // Release the output types dictionary
            objc2::ffi::objc_release(output_types_dict as *mut _);
            
            Some(result)
        }
    }
    
    /// Serializes the executable to a file URL
    ///
    /// - Parameters:
    ///   - url_string: The URL string where the executable will be saved
    ///   - descriptor: A descriptor controlling serialization options
    ///
    /// - Returns: true if serialization was successful
    pub fn serialize_to_url(&self, url_string: &str, descriptor: &MPSGraphExecutableSerializationDescriptor) -> bool {
        unsafe {
            // Convert URL to NSURL
            let nsurl_class = objc2::runtime::AnyClass::get(c"NSURL").unwrap();
            let url_string = NSString::from_str(url_string);
            let nsurl: *mut AnyObject = msg_send![nsurl_class, URLWithString: url_string.as_raw_object()];
            
            // Serialize
            let result: bool = msg_send![
                self.0,
                serializeToMPSGraphPackageAtURL: nsurl,
                descriptor: descriptor.0
            ];
            
            // Release NSURL
            objc2::ffi::objc_release(nsurl as *mut _);
            
            result
        }
    }
}

/// Helper function to convert an NSDictionary to a Rust HashMap
fn convert_dictionary_to_hash_map(dictionary: *mut AnyObject) -> HashMap<MPSGraphTensor, MPSGraphTensorData> {
    unsafe {
        let mut result = HashMap::new();
        
        // Get an enumerator for the dictionary keys
        let enumerator: *mut AnyObject = msg_send![dictionary, keyEnumerator];
        
        while {
            let key: *mut AnyObject = msg_send![enumerator, nextObject];
            !key.is_null()
        } {
            let key: *mut AnyObject = msg_send![enumerator, currentObject];
            let value: *mut AnyObject = msg_send![dictionary, objectForKey: key];
            
            // Retain the objects to avoid them being deallocated
            objc2::ffi::objc_retain(key as *mut _);
            objc2::ffi::objc_retain(value as *mut _);
            
            // Create Rust wrappers
            let tensor = MPSGraphTensor(key);
            let tensor_data = MPSGraphTensorData(value);
            
            // Add to the result HashMap
            result.insert(tensor, tensor_data);
        }
        
        result
    }
}

impl Drop for MPSGraphExecutable {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraphExecutable {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
                MPSGraphExecutable(obj)
            } else {
                MPSGraphExecutable(ptr::null_mut())
            }
        }
    }
}

impl fmt::Debug for MPSGraphExecutable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MPSGraphExecutable")
            .finish()
    }
}

/// A wrapper for MPSGraphCompilationDescriptor
pub struct MPSGraphCompilationDescriptor(pub(crate) *mut AnyObject);

impl MPSGraphCompilationDescriptor {
    /// Create a new compilation descriptor
    pub fn new() -> Self {
        unsafe {
            let class_name = c"MPSGraphCompilationDescriptor";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            let obj: *mut AnyObject = msg_send![cls, alloc];
            let descriptor: *mut AnyObject = msg_send![obj, init];
            MPSGraphCompilationDescriptor(descriptor)
        }
    }
    
    /// Set the optimization level
    pub fn set_optimization_level(&self, level: MPSGraphOptimization) {
        unsafe {
            let _: () = msg_send![self.0, setOptimizationLevel: level as u64];
        }
    }
    
    /// Set the optimization profile
    pub fn set_optimization_profile(&self, profile: MPSGraphOptimizationProfile) {
        unsafe {
            let _: () = msg_send![self.0, setOptimizationProfile: profile as u64];
        }
    }
    
    /// Set whether to debug compile
    pub fn set_debug_compile(&self, debug_compile: bool) {
        unsafe {
            let _: () = msg_send![self.0, setDebugCompile: debug_compile];
        }
    }
}

impl Default for MPSGraphCompilationDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MPSGraphCompilationDescriptor {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraphCompilationDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
                MPSGraphCompilationDescriptor(obj)
            } else {
                MPSGraphCompilationDescriptor(ptr::null_mut())
            }
        }
    }
}

impl fmt::Debug for MPSGraphCompilationDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MPSGraphCompilationDescriptor")
            .finish()
    }
}

/// A wrapper for MPSGraphExecutionDescriptor
pub struct MPSGraphExecutionDescriptor(pub(crate) *mut AnyObject);

impl MPSGraphExecutionDescriptor {
    /// Create a new execution descriptor
    pub fn new() -> Self {
        unsafe {
            let class_name = c"MPSGraphExecutionDescriptor";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            let obj: *mut AnyObject = msg_send![cls, alloc];
            let descriptor: *mut AnyObject = msg_send![obj, init];
            MPSGraphExecutionDescriptor(descriptor)
        }
    }
    
    /// Set wait until completed flag
    ///
    /// If set to true, the execution will block until completed (synchronous behavior).
    /// This provides a simpler alternative to using callbacks for synchronization.
    ///
    /// - Parameter wait: Whether to wait until execution completes
    pub fn set_wait_until_completed(&self, wait: bool) {
        unsafe {
            let _: () = msg_send![self.0, setWaitUntilCompleted: wait];
        }
    }
    
    /// Set the state of the execution descriptor to prefer synchronous execution
    /// 
    /// This is a simplified alternative to using callbacks which can be complex with Objective-C blocks.
    /// It configures the descriptor to wait until execution is completed, effectively making
    /// execution synchronous.
    pub fn prefer_synchronous_execution(&self) {
        unsafe {
            // Set wait until completed to true
            let _: () = msg_send![self.0, setWaitUntilCompleted: true];
        }
    }
    
    /// Set the state of the execution descriptor to prefer asynchronous execution
    ///
    /// This configures the descriptor for asynchronous execution, but note that
    /// true callback support requires proper Objective-C block implementations
    /// which are not fully supported in this version.
    pub fn prefer_asynchronous_execution(&self) {
        unsafe {
            // Set wait until completed to false
            let _: () = msg_send![self.0, setWaitUntilCompleted: false];
        }
    }
    
    /// Wait for a Metal shared event with a specific value before scheduling execution
    /// 
    /// - Parameters:
    ///   - event: The MTLSharedEvent to wait on
    ///   - value: The value to wait for
    pub fn wait_for_event(&self, event: &metal::SharedEvent, value: u64) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut AnyObject;
            let _: () = msg_send![self.0, waitForEvent: event_ptr, value: value];
        }
    }
    
    /// Signal a Metal shared event with a value at a specific execution stage
    /// 
    /// - Parameters:
    ///   - event: The MTLSharedEvent to signal
    ///   - execution_stage: The stage at which to signal the event
    ///   - value: The value to signal with
    pub fn signal_event(&self, event: &metal::SharedEvent, execution_stage: MPSGraphExecutionStage, value: u64) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut AnyObject;
            let _: () = msg_send![self.0, signalEvent: event_ptr, atExecutionEvent: execution_stage as u64, value: value];
        }
    }
}

/// Represents the stages of execution for a graph
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum MPSGraphExecutionStage {
    /// Execution is completed
    Completed = 0,
}

/// Represents the deployment platform for a graph
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum MPSGraphDeploymentPlatform {
    /// macOS platform
    MacOS = 0,
    /// iOS platform
    IOS = 1,
    /// tvOS platform
    TVOS = 2,
    /// visionOS platform
    VisionOS = 3,
}

/// A wrapper for MPSGraphExecutableSerializationDescriptor
pub struct MPSGraphExecutableSerializationDescriptor(pub(crate) *mut AnyObject);

impl MPSGraphExecutableSerializationDescriptor {
    /// Create a new serialization descriptor
    pub fn new() -> Self {
        unsafe {
            let class_name = c"MPSGraphExecutableSerializationDescriptor";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            let obj: *mut AnyObject = msg_send![cls, alloc];
            let descriptor: *mut AnyObject = msg_send![obj, init];
            MPSGraphExecutableSerializationDescriptor(descriptor)
        }
    }
    
    /// Set append flag - if true, appends to existing file instead of overwriting
    pub fn set_append(&self, append: bool) {
        unsafe {
            let _: () = msg_send![self.0, setAppend: append];
        }
    }
    
    /// Set deployment platform
    pub fn set_deployment_platform(&self, platform: MPSGraphDeploymentPlatform) {
        unsafe {
            let _: () = msg_send![self.0, setDeploymentPlatform: platform as u64];
        }
    }
    
    /// Set minimum deployment target as a string (e.g., "13.0")
    pub fn set_minimum_deployment_target(&self, target: &str) {
        unsafe {
            let target_str = NSString::from_str(target);
            let _: () = msg_send![self.0, setMinimumDeploymentTarget: target_str.as_raw_object()];
        }
    }
}

impl Default for MPSGraphExecutableSerializationDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MPSGraphExecutableSerializationDescriptor {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraphExecutableSerializationDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
                MPSGraphExecutableSerializationDescriptor(obj)
            } else {
                MPSGraphExecutableSerializationDescriptor(ptr::null_mut())
            }
        }
    }
}

impl fmt::Debug for MPSGraphExecutableSerializationDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MPSGraphExecutableSerializationDescriptor")
            .finish()
    }
}

impl Default for MPSGraphExecutionDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MPSGraphExecutionDescriptor {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraphExecutionDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
                MPSGraphExecutionDescriptor(obj)
            } else {
                MPSGraphExecutionDescriptor(ptr::null_mut())
            }
        }
    }
}

impl fmt::Debug for MPSGraphExecutionDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MPSGraphExecutionDescriptor")
            .finish()
    }
}

/// A wrapper for MPSGraphExecutableExecutionDescriptor
pub struct MPSGraphExecutableExecutionDescriptor(pub(crate) *mut AnyObject);

impl MPSGraphExecutableExecutionDescriptor {
    /// Create a new executable execution descriptor
    pub fn new() -> Self {
        unsafe {
            let class_name = c"MPSGraphExecutableExecutionDescriptor";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            let obj: *mut AnyObject = msg_send![cls, alloc];
            let descriptor: *mut AnyObject = msg_send![obj, init];
            MPSGraphExecutableExecutionDescriptor(descriptor)
        }
    }
    
    /// Set wait until completed flag
    ///
    /// If set to true, the execution will block until completed (synchronous behavior).
    /// This provides a simpler alternative to using callbacks for synchronization.
    ///
    /// - Parameter wait: Whether to wait until execution completes
    pub fn set_wait_until_completed(&self, wait: bool) {
        unsafe {
            let _: () = msg_send![self.0, setWaitUntilCompleted: wait];
        }
    }
    
    /// Set the state of the execution descriptor to prefer synchronous execution
    /// 
    /// This is a simplified alternative to using callbacks which can be complex with Objective-C blocks.
    /// It configures the descriptor to wait until execution is completed, effectively making
    /// execution synchronous.
    pub fn prefer_synchronous_execution(&self) {
        unsafe {
            // Set wait until completed to true
            let _: () = msg_send![self.0, setWaitUntilCompleted: true];
        }
    }
    
    /// Set the state of the execution descriptor to prefer asynchronous execution
    ///
    /// This configures the descriptor for asynchronous execution, but note that
    /// true callback support requires proper Objective-C block implementations
    /// which are not fully supported in this version.
    pub fn prefer_asynchronous_execution(&self) {
        unsafe {
            // Set wait until completed to false
            let _: () = msg_send![self.0, setWaitUntilCompleted: false];
        }
    }
    
    /// Wait for a Metal shared event with a specific value before scheduling execution
    /// 
    /// - Parameters:
    ///   - event: The MTLSharedEvent to wait on
    ///   - value: The value to wait for
    pub fn wait_for_event(&self, event: &metal::SharedEvent, value: u64) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut AnyObject;
            let _: () = msg_send![self.0, waitForEvent: event_ptr, value: value];
        }
    }
    
    /// Signal a Metal shared event with a value at a specific execution stage
    /// 
    /// - Parameters:
    ///   - event: The MTLSharedEvent to signal
    ///   - execution_stage: The stage at which to signal the event
    ///   - value: The value to signal with
    pub fn signal_event(&self, event: &metal::SharedEvent, execution_stage: MPSGraphExecutionStage, value: u64) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut AnyObject;
            let _: () = msg_send![self.0, signalEvent: event_ptr, atExecutionEvent: execution_stage as u64, value: value];
        }
    }
}

impl Default for MPSGraphExecutableExecutionDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MPSGraphExecutableExecutionDescriptor {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraphExecutableExecutionDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
                MPSGraphExecutableExecutionDescriptor(obj)
            } else {
                MPSGraphExecutableExecutionDescriptor(ptr::null_mut())
            }
        }
    }
}

impl fmt::Debug for MPSGraphExecutableExecutionDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MPSGraphExecutableExecutionDescriptor")
            .finish()
    }
}