use objc2::runtime::AnyObject;
use objc2::msg_send;
use std::fmt;
use std::collections::HashMap;
use std::ffi::c_void;
use metal::CommandQueue;
use metal::foreign_types::ForeignType;
use objc2_foundation::{NSString, NSData};
use crate::core::{MPSDataType, MPSShape, MPSGraphOptions, OurNSString, OurNSArray, OurNSDictionary, AsRawObject};
use crate::tensor::MPSGraphTensor;
use crate::tensor_data::MPSGraphTensorData;
use crate::executable::MPSGraphExecutable;
use crate::device::MPSGraphDevice;

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
            let _: () = msg_send![self.0, setOptions: options as u64];
        }
    }
    
    /// Gets the current options for this graph
    pub fn options(&self) -> MPSGraphOptions {
        unsafe {
            let value: u64 = msg_send![self.0, options];
            std::mem::transmute(value)
        }
    }
    
    /// Creates a placeholder tensor in the graph using shape dimensions
    /// This is a convenience method that converts shape dimensions to an MPSShape
    pub fn placeholder_with_shape(&self, shape_dims: &[usize], data_type: MPSDataType, name: Option<&str>) -> MPSGraphTensor {
        let shape = MPSShape::from_slice(shape_dims);
        self.placeholder(&shape, data_type, name)
    }
    
    /// Creates a placeholder tensor in the graph
    pub fn placeholder(&self, shape: &MPSShape, data_type: MPSDataType, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => OurNSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, placeholderWithShape: shape.0,
                dataType:  data_type as u32,
                name: name_obj,
            ];
            
            // Retain using objc2 functions
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a constant tensor with scalar value, specifying shape explicitly
    pub fn constant_scalar_with_shape(&self, value: f32, shape: &MPSShape, data_type: MPSDataType, name: Option<&str>) -> MPSGraphTensor {
        // Use constant_from_bytes to implement this method
        // We need to workaround issues with method resolution in the Objective-C API
        let data = [value];
        self.constant_from_bytes(&data, shape, data_type, name)
    }
    
    /// Creates a constant tensor with data from a slice
    /// Creates a tensor filled with zeros that has the same shape as the input tensor
    pub fn zeros_like(&self, tensor: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let result: *mut AnyObject = msg_send![self.0, zerosLikeTensor: tensor.0,
                name: name_obj,
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a scalar constant tensor with the given value
    pub fn constant_scalar_value<T: Copy>(&self, value: T, data_type: MPSDataType, name: Option<&str>) -> MPSGraphTensor {
        // Create a slice with a single value
        let data = [value];
        let shape = MPSShape::scalar();
        self.constant_from_bytes(&data, &shape, data_type, name)
    }
    
    /// Legacy method for backward compatibility
    pub fn constant_scalar(&self, value: f32, shape: &MPSShape, data_type: MPSDataType, name: Option<&str>) -> MPSGraphTensor {
        self.constant_scalar_with_shape(value, shape, data_type, name)
    }
    
    /// Creates a constant tensor from raw bytes
    pub fn constant_from_bytes<T: Copy>(&self, 
                                data:  &[T], 
                                shape:  &MPSShape, 
                                data_type:  MPSDataType, 
                                name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let _data_ptr = data.as_ptr() as *const c_void;
            let _data_len = std::mem::size_of_val(data); // Used for debugging
            
            // Create NSData for the input data
            let data_size = std::mem::size_of_val(data);
            let ns_data = NSData::with_bytes(std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data_size
            ));
            let ns_data_ptr = ns_data.as_raw_object();
            
            // Try a different method name or signature
            let tensor: *mut AnyObject = msg_send![self.0, constantWithSharedData: ns_data_ptr,
                shape:  shape.0,
                dataType:  data_type as u32,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Compiles the graph for the given feeds and targets
    pub fn compile(&self, 
               device:  Option<&MPSGraphDevice>,
               feeds:  HashMap<&MPSGraphTensor, MPSShape>,
               targets:  &[&MPSGraphTensor],
               name:  Option<&str>) -> MPSGraphExecutable {
        unsafe {
            // Create device object if provided
            let device_obj = match device {
                Some(dev) => dev.0,
                None => std::ptr::null_mut(),
            };
            
            // Create name string if provided
            let name_obj = match name {
                Some(s) => OurNSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Create feeds dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, shape) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(shape.0);
            }
            
            let feed_dict = OurNSDictionary::from_keys_and_objects(&feed_keys, &feed_values);
            
            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter()
                .map(|t| t.0)
                .collect();
            
            let targets_array = OurNSArray::from_objects(&targets_raw);
            
            // Compile the graph
            let executable: *mut AnyObject = msg_send![self.0, compileWithDevice: device_obj, 
                feeds: feed_dict.0, 
                targetTensors: targets_array.0, 
                targetOperations: std::ptr::null_mut::<AnyObject>(), 
                name: name_obj,
            ];
            
            let executable = objc2::ffi::objc_retain(executable as *mut _) as *mut AnyObject;
            MPSGraphExecutable(executable)
        }
    }
    
    /// Runs the graph with the given feeds and returns the targets
    pub fn run_legacy(&self,
           feeds:  HashMap<&MPSGraphTensor, MPSGraphTensorData>,
           targets:  &[&MPSGraphTensor]) -> HashMap<MPSGraphTensor, MPSGraphTensorData> {
        unsafe {
            // Create feed dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = OurNSDictionary::from_keys_and_objects(&feed_keys, &feed_values);
            
            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter()
                .map(|t| t.0)
                .collect();
            
            let targets_array = OurNSArray::from_objects(&targets_raw);
            
            // Run the graph - fixing the syntax for objc2
            let results: *mut AnyObject = msg_send![self.0, runWithFeeds: feed_dict.0, 
                targetTensors: targets_array.0, 
                targetOperations: std::ptr::null_mut::<AnyObject>(),
            ];
            
            // Parse the results
            let mut output = HashMap::new();
            
            // Check if results is valid
            if results.is_null() {
                return output;
            }
            
            // Get all keys
            let keys: *mut AnyObject = msg_send![results, allKeys];
            let keys_count: usize = msg_send![keys, count];
            
            for i in 0..keys_count {
                let key: *mut AnyObject = msg_send![keys, objectAtIndex: i];
                if key.is_null() {
                    continue;
                }
                
                let value: *mut AnyObject = msg_send![results, objectForKey: key];
                if value.is_null() {
                    continue;
                }
                
                // Retain objects to manage their memory
                let _: () = msg_send![key, retain];
                let _: () = msg_send![value, retain];
                
                output.insert(MPSGraphTensor(key), MPSGraphTensorData(value));
            }
            
            // Release temporary objects
            objc2::ffi::objc_release(results as *mut _);
            
            output
        }
    }
    
    /// Runs the graph with the given MTL command queue
    pub fn run_with_command_queue(&self,
                             command_queue:  &CommandQueue,
                             feeds:  HashMap<&MPSGraphTensor, MPSGraphTensorData>,
                             targets:  &[&MPSGraphTensor]) -> HashMap<MPSGraphTensor, MPSGraphTensorData> {
        unsafe {
            // Create feed dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = OurNSDictionary::from_keys_and_objects(&feed_keys, &feed_values);
            
            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter()
                .map(|t| t.0)
                .collect();
            
            let targets_array = OurNSArray::from_objects(&targets_raw);
            
            // Run the graph
            // Convert the Metal command queue pointer to void* to avoid RefEncode issues
            let cmd_queue_ptr = command_queue.as_ptr() as *mut std::ffi::c_void;
            
            let results: *mut AnyObject = msg_send![self.0, runWithMTLCommandQueue: cmd_queue_ptr, 
                feeds: feed_dict.0, 
                targetTensors: targets_array.0, 
                targetOperations: std::ptr::null_mut::<AnyObject>(),
            ];
            
            // Parse the results
            let mut output = HashMap::new();
            
            // Check if results is valid
            if results.is_null() {
                return output;
            }
            
            // Get all keys
            let keys: *mut AnyObject = msg_send![results, allKeys];
            let keys_count: usize = msg_send![keys, count];
            
            for i in 0..keys_count {
                let key: *mut AnyObject = msg_send![keys, objectAtIndex: i];
                if key.is_null() {
                    continue;
                }
                
                let value: *mut AnyObject = msg_send![results, objectForKey: key];
                if value.is_null() {
                    continue;
                }
                
                // Retain objects to manage their memory
                let _: () = msg_send![key, retain];
                let _: () = msg_send![value, retain];
                
                output.insert(MPSGraphTensor(key), MPSGraphTensorData(value));
            }
            
            // Release temporary objects
            objc2::ffi::objc_release(results as *mut _);
            
            output
        }
    }
    
    /// Execute the graph and return results
    pub fn run_graph(
        &self,
        targets:  &[MPSGraphTensor],
        device:  &crate::device::MPSGraphDevice,
        options:  Option<MPSGraphOptions>
    ) -> HashMap<MPSGraphTensor, MPSGraphTensorData> {
        // Empty feeds
        let feeds = HashMap::new();
        self.run_with_feeds(&feeds, targets, device, options)
    }
    
    /// Legacy run method for backward compatibility with tests
    pub fn run(
        &self,
        feeds: HashMap<&MPSGraphTensor, MPSGraphTensorData>,
        targets: &[&MPSGraphTensor]) -> HashMap<MPSGraphTensor, MPSGraphTensorData> {
        println!("DEBUG run: Starting run method");
        println!("DEBUG run: Running with {} feeds and {} targets", feeds.len(), targets.len());

        unsafe {
            // Create feed dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in &feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = OurNSDictionary::from_keys_and_objects(&feed_keys, &feed_values);
            
            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter()
                .map(|t| t.0)
                .collect();
            
            let targets_array = OurNSArray::from_objects(&targets_raw);
            
            // Run the graph with the updated message syntax for objc2
            println!("DEBUG run: About to execute graph");
            let results: *mut AnyObject = msg_send![self.0, runWithFeeds: feed_dict.0, 
                targetTensors: targets_array.0, 
                targetOperations: std::ptr::null_mut::<AnyObject>(),
            ];
            println!("DEBUG run: Got results pointer: {:p}", results);
            
            // Parse the results
            let mut output = HashMap::new();
            
            // Check if results is valid
            if results.is_null() {
                println!("DEBUG run: Results pointer is null");
                return output;
            }
            
            // Get all keys
            println!("DEBUG run: Getting keys");
            let keys: *mut AnyObject = msg_send![results, allKeys];
            if keys.is_null() {
                println!("DEBUG run: Keys pointer is null");
                return output;
            }
            
            let keys_count: usize = msg_send![keys, count];
            println!("DEBUG run: Found {} keys", keys_count);
            
            for i in 0..keys_count {
                let key: *mut AnyObject = msg_send![keys, objectAtIndex: i];
                if key.is_null() {
                    println!("DEBUG run: Key {} is null", i);
                    continue;
                }
                
                let value: *mut AnyObject = msg_send![results, objectForKey: key];
                if value.is_null() {
                    println!("DEBUG run: Value for key {} is null", i);
                    continue;
                }
                
                // Retain objects to manage their memory
                objc2::ffi::objc_retain(key as *mut _);
                objc2::ffi::objc_retain(value as *mut _);
                
                output.insert(MPSGraphTensor(key), MPSGraphTensorData(value));
            }
            
            // Release temporary objects
            objc2::ffi::objc_release(results as *mut _);
            
            println!("DEBUG run: Successfully completed");
            output
        }
    }
    
    /// Execute the graph with feeds and return results
    pub fn run_with_feeds(
        &self,
        feeds:  &HashMap<MPSGraphTensor, MPSGraphTensorData>,
        targets:  &[MPSGraphTensor],
        device:  &crate::device::MPSGraphDevice,
        options:  Option<MPSGraphOptions>
    ) -> HashMap<MPSGraphTensor, MPSGraphTensorData> {
        unsafe {
            // Get Metal device
            let metal_device_ptr: *mut AnyObject = msg_send![device.0, mtlDevice];
            // Convert from Object to MTLDevice
            let mtl_device_ptr = metal_device_ptr as *mut std::ffi::c_void as *mut metal::MTLDevice;
            let metal_device = metal::Device::from_ptr(mtl_device_ptr);
            
            // Create command queue
            let command_queue = metal_device.new_command_queue();
            
            // Create feed dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = OurNSDictionary::from_keys_and_objects(&feed_keys, &feed_values);
            
            // Create targets array
            let targets_raw: Vec<*mut AnyObject> = targets.iter()
                .map(|t| t.0)
                .collect();
            
            let targets_array = OurNSArray::from_objects(&targets_raw);
            
            // Run the graph
            // Convert MTLCommandQueue pointer to void* first to avoid RefEncode trait issues
            let cmd_queue_ptr = command_queue.as_ptr() as *mut std::ffi::c_void;
            
            let results: *mut AnyObject = msg_send![self.0, runWithMTLCommandQueue: cmd_queue_ptr, 
                feeds: feed_dict.0, 
                targetTensors: targets_array.0, 
                targetOperations: std::ptr::null_mut::<AnyObject>(), 
                options: options.unwrap_or(MPSGraphOptions::Default) as u64
            ];
            
            // Parse the results
            let mut output = HashMap::new();
            
            // Check if results is valid
            if results.is_null() {
                return output;
            }
            
            // Get all keys
            let keys: *mut AnyObject = msg_send![results, allKeys];
            let keys_count: usize = msg_send![keys, count];
            
            for i in 0..keys_count {
                let key: *mut AnyObject = msg_send![keys, objectAtIndex: i];
                if key.is_null() {
                    continue;
                }
                
                let value: *mut AnyObject = msg_send![results, objectForKey: key];
                if value.is_null() {
                    continue;
                }
                
                // Retain objects to manage their memory
                let _: () = msg_send![key, retain];
                let _: () = msg_send![value, retain];
                
                output.insert(MPSGraphTensor(key), MPSGraphTensorData(value));
            }
            
            // Release temporary objects
            objc2::ffi::objc_release(results as *mut _);
            
            output
        }
    }
    
    /// Returns all placeholder tensors in this graph
    pub fn placeholder_tensors(&self) -> Vec<MPSGraphTensor> {
        unsafe {
            let tensors: *mut AnyObject = msg_send![self.0, placeholderTensors];
            let count: usize = msg_send![tensors, count];
            let mut result = Vec::with_capacity(count);
            
            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![tensors, objectAtIndex: i];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
}

// Standard arithmetic operations extension
impl MPSGraph {
    /// Creates an addition operation
    pub fn add(&self, lhs: &MPSGraphTensor, rhs: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, additionWithPrimaryTensor: lhs.0,
                secondaryTensor: rhs.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a subtraction operation
    pub fn subtract(&self, lhs: &MPSGraphTensor, rhs: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, subtractionWithPrimaryTensor: lhs.0, 
                secondaryTensor: rhs.0, 
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a multiplication operation
    pub fn multiply(&self, lhs: &MPSGraphTensor, rhs: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, multiplicationWithPrimaryTensor: lhs.0, 
                secondaryTensor: rhs.0, 
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a division operation
    pub fn divide(&self, lhs: &MPSGraphTensor, rhs: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, divisionWithPrimaryTensor: lhs.0,
                secondaryTensor: rhs.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a power operation (x^y)
    pub fn power(&self, lhs: &MPSGraphTensor, rhs: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, powerWithPrimaryTensor: lhs.0,
                secondaryTensor: rhs.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a negation operation (-x)
    pub fn negative(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, negativeWithTensor: x.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates an absolute value operation (|x|)
    pub fn abs(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, absoluteWithTensor: x.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}

// Math functions extension
impl MPSGraph {
    /// Creates an exponent operation (e^x)
    pub fn exp(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, exponentWithTensor: x.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a logarithm operation (ln(x))
    pub fn log(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, logarithmWithTensor: x.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a square root operation (sqrt(x))
    pub fn sqrt(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, squareRootWithTensor: x.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a reciprocal square root operation (1/sqrt(x))
    pub fn rsqrt(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, reciprocalSquareRootWithTensor: x.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a square operation (x^2)
    pub fn square(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, squareWithTensor: x.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a type cast operation to convert tensor to another data type
    pub fn cast(&self, x: &MPSGraphTensor, data_type: MPSDataType, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, castTensor: x.0,
                toType: data_type as u64,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a sine operation (sin(x))
    pub fn sin(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, sineWithTensor: x.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a cosine operation (cos(x))
    pub fn cos(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, cosineWithTensor: x.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a tangent operation (tan(x))
    pub fn tan(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, tangentWithTensor: x.0,
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
                Some(s) => OurNSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Create a shape object from the dimensions for convenience
            let dimensions_shape = MPSShape::from_slice(dimensions);
            
            // Create the operation
            let tensor: *mut AnyObject = msg_send![self.0, transposeTensor: x.0,
                dimension: dimensions_shape.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}

// Activation functions extension
impl MPSGraph {
    /// Creates a ReLU activation
    pub fn relu(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, reLUWithTensor: x.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a sigmoid activation
    pub fn sigmoid(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, sigmoidWithTensor: x.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a tanh activation
    pub fn tanh(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, tanhWithTensor: x.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a softmax activation
    pub fn softmax(&self, x: &MPSGraphTensor, axis: i64, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, softMaxWithTensor: x.0,
                axis: axis,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}

// Tensor shape operations extension
impl MPSGraph {
    /// Reshapes a tensor to a new shape
    pub fn reshape(&self, x: &MPSGraphTensor, new_shape: &MPSShape, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, reshapeTensor: x.0,
                withShape: new_shape.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a broadcast operation to expand a tensor to a larger shape
    pub fn broadcast(&self, x: &MPSGraphTensor, shape: &MPSShape, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, broadcastTensor: x.0,
                toShape: shape.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a concatenation operation along the specified axis
    pub fn concatenate(&self, tensors: &[&MPSGraphTensor], dimension: usize, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Convert tensors to NSArray
            let tensors_raw: Vec<*mut AnyObject> = tensors.iter()
                .map(|t| t.0)
                .collect();
            
            let tensors_array = crate::core::OurNSArray::from_objects(&tensors_raw);
            
            let tensor: *mut AnyObject = msg_send![self.0, concatenationWithTensors: tensors_array.0,
                dimension: dimension,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}

// Reduction operations extension
impl MPSGraph {
    /// Creates a sum reduction along specified axes
    pub fn reduce_sum(&self, x: &MPSGraphTensor, axes: &[usize], name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Convert axes to NSArray of NSNumbers
            let axes_shape = MPSShape::from_slice(axes);
            
            let tensor: *mut AnyObject = msg_send![self.0, reductionSumWithTensor: x.0,
                axes: axes_shape.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a mean reduction along specified axes
    pub fn reduce_mean(&self, x: &MPSGraphTensor, axes: &[usize], name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Convert axes to NSArray of NSNumbers
            let axes_shape = MPSShape::from_slice(axes);
            
            let tensor: *mut AnyObject = msg_send![self.0, reductionMeanWithTensor: x.0,
                axes: axes_shape.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a max reduction along specified axes
    pub fn reduce_max(&self, x: &MPSGraphTensor, axes: &[usize], name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Convert axes to NSArray of NSNumbers
            let axes_shape = MPSShape::from_slice(axes);
            
            let tensor: *mut AnyObject = msg_send![self.0, reductionMaximumWithTensor: x.0,
                axes: axes_shape.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a min reduction along specified axes
    pub fn reduce_min(&self, x: &MPSGraphTensor, axes: &[usize], name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Convert axes to NSArray of NSNumbers
            let axes_shape = MPSShape::from_slice(axes);
            
            let tensor: *mut AnyObject = msg_send![self.0, reductionMinimumWithTensor: x.0,
                axes: axes_shape.0,
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
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraph {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
                MPSGraph(obj)
            } else {
                MPSGraph(std::ptr::null_mut())
            }
        }
    }
}

impl fmt::Debug for MPSGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MPSGraph")
    }
}