use objc2::runtime::AnyObject;
use objc2::msg_send;
use std::fmt;
use std::collections::HashMap;
use metal::{CommandQueue, CommandBuffer, SharedEvent};
use metal::foreign_types::ForeignType;
use objc2_foundation::NSData;
use crate::core::{MPSDataType, MPSGraphOptions, AsRawObject};
use crate::shape::MPSShape;
use objc2_foundation::NSString;
use crate::tensor::MPSGraphTensor;
use crate::tensor_data::MPSGraphTensorData;
use crate::executable::{MPSGraphExecutable, MPSGraphCompilationDescriptor, MPSGraphExecutionDescriptor};
use crate::device::MPSGraphDevice;
use crate::operation::MPSGraphOperation;

#[link(name = "MetalPerformanceShadersGraph", kind = "framework")]
extern "C" {
    #[allow(dead_code)]
    fn MPSGraphCreate() -> *mut AnyObject;
}

/// A wrapper for MPSGraph objects
pub struct MPSGraph(pub(crate) *mut AnyObject);

// Implement Send + Sync for the wrapper type
unsafe impl Send for MPSGraph {}
unsafe impl Sync for MPSGraph {}

impl Default for MPSGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MPSGraph {
    /// Creates a new MPS Graph
    pub fn new() -> Self {
        unsafe {
            // Create without using external MPSGraphCreate function
            let class_name = c"MPSGraph";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            let obj: *mut AnyObject = msg_send![cls, alloc];
            let graph: *mut AnyObject = msg_send![obj, init];
            MPSGraph(graph)
        }
    }
    
    /// Sets the options for this graph
    pub fn set_options(&self, options: MPSGraphOptions) {
        unsafe {
            let _: () = msg_send![self.0, setOptions:options as u64];
        }
    }
    
    /// Creates a placeholder tensor with dimensions
    pub fn placeholder_with_shape(&self, 
                     shape_dims: &[usize], 
                     data_type: MPSDataType, 
                     name: Option<&str>) -> MPSGraphTensor {
        let shape = MPSShape::from_slice(shape_dims);
        self.placeholder(&shape, data_type, name)
    }
    
    /// Creates a placeholder tensor in the graph
    pub fn placeholder(&self, shape: &MPSShape, data_type: MPSDataType, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut::<AnyObject>(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, placeholderWithShape: shape.0,
                dataType:  data_type as u32,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a constant tensor with given values and dimensions
    pub fn constant_with_shape(&self, 
                    values:  &[f32], 
                    shape_dims:  &[usize], 
                    name:  Option<&str>) -> MPSGraphTensor {
        let shape = MPSShape::from_slice(shape_dims);
        self.constant(values, &shape, name)
    }
    
    /// Creates a constant tensor with given values and shape
    pub fn constant(&self, 
                  values:  &[f32], 
                  shape:  &MPSShape, 
                  name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut::<AnyObject>(),
            };
            
            // Create NSData with float values
            let data = NSData::with_bytes(std::slice::from_raw_parts(
                values.as_ptr() as *const u8,
                values.len() * std::mem::size_of::<f32>()
            ));
            
            // Get raw NSData pointer for Objective-C
            let data_ptr: *mut AnyObject = std::mem::transmute::<&NSData, *mut AnyObject>(data.as_ref());
            
            // Create constant tensor
            let tensor: *mut AnyObject = msg_send![self.0, constantWithData: data_ptr,
                shape: shape.0,
                dataType: MPSDataType::Float32 as u32,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a constant with given scalar value
    pub fn constant_scalar(&self, 
                         value:  f32, 
                         data_type:  MPSDataType, 
                         name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut::<AnyObject>(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, constantWithScalar: value as f64,
                dataType: data_type as u32,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a random uniform tensor with given shape
    pub fn random_uniform(&self, 
                        shape:  &MPSShape, 
                        lower_bound:  f32, 
                        upper_bound:  f32, 
                        data_type:  MPSDataType, 
                        seed:  u32, 
                        name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut::<AnyObject>(),
            };
            
            // Create the random tensor
            let tensor: *mut AnyObject = msg_send![self.0, randomUniformTensorWithShape: shape.0,
                lowerBound: lower_bound as f64,
                upperBound: upper_bound as f64,
                dataType: data_type as u32,
                seed: seed,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a random normal tensor with given shape
    pub fn random_normal(&self, 
                       shape:  &MPSShape, 
                       mean:  f32, 
                       stddev:  f32, 
                       data_type:  MPSDataType, 
                       seed:  u32, 
                       name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut::<AnyObject>(),
            };
            
            // Create the random tensor
            let tensor: *mut AnyObject = msg_send![self.0, randomNormalTensorWithShape: shape.0,
                mean: mean as f64,
                standardDeviation: stddev as f64,
                dataType: data_type as u32,
                seed: seed,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Compiles the graph against a given set of feeds and targets
    pub fn compile(&self,
                 device:  &MPSGraphDevice,
                 feeds:  &HashMap<MPSGraphTensor, MPSGraphTensorData>,
                 targets:  &[MPSGraphTensor],
                 descriptor:  Option<&MPSGraphCompilationDescriptor>) -> MPSGraphExecutable {
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
            
            let feed_dict = crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);
            
            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter()
                .map(|t| t.0)
                .collect();
            
            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);
            
            // Get descriptor pointer if provided
            let descriptor_ptr = match descriptor {
                Some(desc) => desc.0,
                None => std::ptr::null_mut::<AnyObject>(),
            };
            
            // Compile the graph
            let executable: *mut AnyObject = msg_send![self.0, compileWithDevice: device_obj, 
                feeds: feed_dict, 
                targetTensors: targets_array,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
                compilationDescriptor: descriptor_ptr,
            ];
            
            let executable = objc2::ffi::objc_retain(executable as *mut _) as *mut AnyObject;
            MPSGraphExecutable(executable)
        }
    }
    
    /// Compiles the graph against a given set of feeds, targets, and target operations
    pub fn compile_with_targets_and_ops(&self,
                                      device:  &MPSGraphDevice,
                                      feeds:  &HashMap<MPSGraphTensor, MPSGraphTensorData>,
                                      targets:  &[MPSGraphTensor],
                                      target_ops:  &[MPSGraphOperation],
                                      descriptor:  Option<&MPSGraphCompilationDescriptor>) -> MPSGraphExecutable {
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
            
            let feed_dict = crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);
            
            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter()
                .map(|t| t.0)
                .collect();
            
            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);
            
            // Create ops array
            let ops_raw: Vec<*mut AnyObject> = target_ops.iter()
                .map(|op| op.0)
                .collect();
            
            let ops_array = crate::core::create_ns_array_from_pointers(&ops_raw);
            
            // Get descriptor pointer if provided
            let descriptor_ptr = match descriptor {
                Some(desc) => desc.0,
                None => std::ptr::null_mut::<AnyObject>(),
            };
            
            // Compile the graph
            let executable: *mut AnyObject = msg_send![self.0, compileWithDevice: device_obj, 
                feeds: feed_dict, 
                targetTensors: targets_array,
                targetOperations: ops_array,
                compilationDescriptor: descriptor_ptr,
            ];
            
            let executable = objc2::ffi::objc_retain(executable as *mut _) as *mut AnyObject;
            MPSGraphExecutable(executable)
        }
    }
    
    /// Runs the graph synchronously against a given set of feeds and returns a result dictionary
    pub fn run_with_feeds(&self,
                       feeds:  &HashMap<MPSGraphTensor, MPSGraphTensorData>,
                       targets:  &[MPSGraphTensor]) -> HashMap<MPSGraphTensor, MPSGraphTensorData> {
        unsafe {
            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);
            
            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter()
                .map(|t| t.0)
                .collect();
            
            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);
            
            // Run the graph
            let results: *mut AnyObject = msg_send![self.0, runWithFeeds: feed_dict, 
                targetTensors: targets_array,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
            ];
            
            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);
            
            // Release the results dictionary
            objc2::ffi::objc_release(results as *mut _);
            
            result_hash
        }
    }
    
    /// Runs the graph synchronously against a given set of feeds and returns a result dictionary
    /// with target operations specified
    pub fn run_with_feeds_and_ops(&self,
                               feeds:  &HashMap<MPSGraphTensor, MPSGraphTensorData>,
                               targets:  &[MPSGraphTensor],
                               target_ops:  &[MPSGraphOperation]) -> HashMap<MPSGraphTensor, MPSGraphTensorData> {
        unsafe {
            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);
            
            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter()
                .map(|t| t.0)
                .collect();
            
            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);
            
            // Create ops array
            let ops_raw: Vec<*mut AnyObject> = target_ops.iter()
                .map(|op| op.0)
                .collect();
            
            let ops_array = crate::core::create_ns_array_from_pointers(&ops_raw);
            
            // Run the graph
            let results: *mut AnyObject = msg_send![self.0, runWithFeeds: feed_dict, 
                targetTensors: targets_array,
                targetOperations: ops_array,
            ];
            
            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);
            
            // Release the results dictionary
            objc2::ffi::objc_release(results as *mut _);
            
            result_hash
        }
    }
    
    /// Runs the graph synchronized against a given device
    pub fn run_with_feeds_on_device(&self,
                                 device:  &MPSGraphDevice,
                                 feeds:  &HashMap<MPSGraphTensor, MPSGraphTensorData>,
                                 targets:  &[MPSGraphTensor]) -> HashMap<MPSGraphTensor, MPSGraphTensorData> {
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
            
            let feed_dict = crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);
            
            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter()
                .map(|t| t.0)
                .collect();
            
            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);
            
            // Run the graph
            let results: *mut AnyObject = msg_send![self.0, runWithFeeds: feed_dict, 
                targetTensors: targets_array,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
                onDevice: device_obj,
            ];
            
            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);
            
            // Release the results dictionary
            objc2::ffi::objc_release(results as *mut _);
            
            result_hash
        }
    }
    
    /// Runs the graph synchronously on a device with both target tensors and operations
    pub fn run_with_feeds_and_ops_on_device(&self,
                                         device:  &MPSGraphDevice,
                                         feeds:  &HashMap<MPSGraphTensor, MPSGraphTensorData>,
                                         targets:  &[MPSGraphTensor],
                                         target_ops:  &[MPSGraphOperation]) -> HashMap<MPSGraphTensor, MPSGraphTensorData> {
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
            
            let feed_dict = crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);
            
            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter()
                .map(|t| t.0)
                .collect();
            
            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);
            
            // Create ops array
            let ops_raw: Vec<*mut AnyObject> = target_ops.iter()
                .map(|op| op.0)
                .collect();
            
            let ops_array = crate::core::create_ns_array_from_pointers(&ops_raw);
            
            // Run the graph
            let results: *mut AnyObject = msg_send![self.0, runWithFeeds: feed_dict, 
                targetTensors: targets_array,
                targetOperations: ops_array,
                onDevice: device_obj,
            ];
            
            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);
            
            // Release the results dictionary
            objc2::ffi::objc_release(results as *mut _);
            
            result_hash
        }
    }
    
    /// Enqueue a graph run on a command queue
    pub fn encode_to_command_queue(&self,
                                device:  &MPSGraphDevice,
                                command_queue:  &CommandQueue,
                                feeds:  &HashMap<MPSGraphTensor, MPSGraphTensorData>,
                                targets:  &[MPSGraphTensor],
                                execution_descriptor:  Option<&MPSGraphExecutionDescriptor>) -> HashMap<MPSGraphTensor, MPSGraphTensorData> {
        unsafe {
            // Get the device pointer
            let device_obj = device.0;
            
            // Get the queue pointer
            let queue_ptr = command_queue.as_ptr() as *mut AnyObject;
            
            // Create the feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);
            
            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter()
                .map(|t| t.0)
                .collect();
            
            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);
            
            // Get execution descriptor pointer if provided
            let descriptor_ptr = match execution_descriptor {
                Some(desc) => desc.0,
                None => std::ptr::null_mut::<AnyObject>(),
            };
            
            // Encode and run the graph
            let results: *mut AnyObject = msg_send![self.0, encodeToCommandQueueAndReturnError: queue_ptr,
                feeds: feed_dict,
                targetTensors: targets_array,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
                executionDescriptor: descriptor_ptr,
                error: std::ptr::null_mut::<AnyObject>(),
            ];
            
            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);
            
            // Release the results dictionary
            objc2::ffi::objc_release(results as *mut _);
            
            result_hash
        }
    }
    
    /// Encode and run a graph on a command buffer
    pub fn encode_to_command_buffer(&self,
                                 command_buffer:  &CommandBuffer,
                                 feeds:  &HashMap<MPSGraphTensor, MPSGraphTensorData>,
                                 targets:  &[MPSGraphTensor],
                                 execution_descriptor:  Option<&MPSGraphExecutionDescriptor>) -> HashMap<MPSGraphTensor, MPSGraphTensorData> {
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
            
            let feed_dict = crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);
            
            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter()
                .map(|t| t.0)
                .collect();
            
            let targets_array = crate::core::create_ns_array_from_pointers(&targets_raw);
            
            // Get execution descriptor pointer if provided
            let descriptor_ptr = match execution_descriptor {
                Some(desc) => desc.0,
                None => std::ptr::null_mut::<AnyObject>(),
            };
            
            // Encode and run the graph
            let results: *mut AnyObject = msg_send![self.0, encodeToCommandBuffer: buffer_ptr,
                feeds: feed_dict,
                targetTensors: targets_array,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
                executionDescriptor: descriptor_ptr,
            ];
            
            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);
            
            // Release the results dictionary
            objc2::ffi::objc_release(results as *mut _);
            
            result_hash
        }
    }
    
    /// Encodes and runs a graph on a command buffer with a completed event
    pub fn encode_to_command_buffer_with_event(&self,
                                            command_buffer:  &CommandBuffer,
                                            feeds:  &HashMap<MPSGraphTensor, MPSGraphTensorData>,
                                            targets:  &[MPSGraphTensor],
                                            execution_descriptor:  Option<&MPSGraphExecutionDescriptor>,
                                            event:  Option<&SharedEvent>) -> HashMap<MPSGraphTensor, MPSGraphTensorData> {
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
            
            let feed_dict = crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);
            
            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter()
                .map(|t| t.0)
                .collect();
            
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
            let results: *mut AnyObject = msg_send![self.0, encodeToCommandBuffer: buffer_ptr,
                feeds: feed_dict,
                targetTensors: targets_array,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
                executionDescriptor: descriptor_ptr,
                waitEvent: event_ptr,
                signalEvent: event_ptr,
                signalValue: 1,
            ];
            
            // Convert the result dictionary to a Rust HashMap
            let result_hash = self.convert_dictionary_to_hash_map(results);
            
            // Release the results dictionary
            objc2::ffi::objc_release(results as *mut _);
            
            result_hash
        }
    }
    
    /// Runs the graph with a command queue, feeds and outputs specified
    pub fn run_with_command_queue_feeds_outputs(
        &self,
        command_queue: &CommandQueue,
        feeds: HashMap<&MPSGraphTensor, &MPSGraphTensorData>,
        results_dict: HashMap<&MPSGraphTensor, &MPSGraphTensorData>
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
            
            let feed_dict = crate::core::create_ns_dictionary_from_pointers(&feed_keys, &feed_values);
            
            // Create the results dictionary
            let mut results_keys = Vec::with_capacity(results_dict.len());
            let mut results_values = Vec::with_capacity(results_dict.len());
            
            for (tensor, data) in results_dict {
                results_keys.push(tensor.0);
                results_values.push(data.0);
            }
            
            let results_dict_obj = crate::core::create_ns_dictionary_from_pointers(&results_keys, &results_values);
            
            // Run the graph with both feeds and results dictionaries
            println!("Executing graph with feeds and outputs...");
            let _: () = msg_send![self.0, runWithMTLCommandQueue: queue_ptr,
                feeds: feed_dict,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
                resultsDictionary: results_dict_obj,
            ];
            println!("Graph execution completed.");
            
            // Release dictionaries
            objc2::ffi::objc_release(feed_dict as *mut _);
            objc2::ffi::objc_release(results_dict_obj as *mut _);
        }
    }
    
    /// Helper method to convert an NSDictionary to a Rust HashMap
    fn convert_dictionary_to_hash_map(&self, dictionary: *mut AnyObject) -> HashMap<MPSGraphTensor, MPSGraphTensorData> {
        unsafe {
            let mut result = HashMap::new();
            
            // Get an enumerator for the dictionary keys
            let enumerator: *mut AnyObject = msg_send![dictionary, keyEnumerator];
            
            // Use a mutable variable for the key outside the loop condition
            let mut key: *mut AnyObject;
            
            while {
                // nextObject both advances the enumerator AND returns the current object
                key = msg_send![enumerator, nextObject];
                !key.is_null()
            } {
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
}

// Concatenation operations extension
impl MPSGraph {
    /// Creates a concatenation operation
    pub fn concatenate(&self, tensors: &[MPSGraphTensor], dimension: i64, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut::<AnyObject>(),
            };
            
            // Create array of tensors
            let tensors_raw: Vec<*mut AnyObject> = tensors.iter()
                .map(|t| t.0)
                .collect();
            
            let tensors_array = crate::core::create_ns_array_from_pointers(&tensors_raw);
            
            let tensor: *mut AnyObject = msg_send![self.0, concatenationWithTensors: tensors_array,
                dimension: dimension,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}

// Matrix operations extension
impl MPSGraph {
    /// Creates a transpose operation
    pub fn transpose(&self, x: &MPSGraphTensor, dimensions: &[usize], name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut::<AnyObject>(),
            };
            
            // Create a shape object from the dimensions for convenience
            let dimensions_shape = MPSShape::from_slice(dimensions);
            
            // Create the operation
            let tensor: *mut AnyObject = msg_send![self.0, transposeTensor: x.0,
                permutation: dimensions_shape.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}

impl Drop for MPSGraph {
    fn drop(&mut self) {
        unsafe {
            // Convert to NSObject and release
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraph {
    fn clone(&self) -> Self {
        unsafe {
            // Retain and return new instance
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
                MPSGraph(obj)
            } else {
                MPSGraph(std::ptr::null_mut::<AnyObject>())
            }
        }
    }
}

impl fmt::Debug for MPSGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MPSGraph")
            .finish()
    }
}

// Helper function to create NSArray from dimensions
#[allow(dead_code)]
fn create_dimensions(dimensions: &[usize]) -> *mut AnyObject {
    let shape = MPSShape::from_slice(dimensions);
    shape.0
}