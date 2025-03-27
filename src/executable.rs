use objc2::runtime::AnyObject;
use objc2::msg_send;
use std::fmt;
use std::ptr;
use std::collections::HashMap;
use metal::CommandBuffer;
use metal::foreign_types::ForeignType;
use crate::tensor::MPSGraphTensor;
use crate::tensor_data::MPSGraphTensorData;
use crate::core::{MPSGraphOptimization, MPSGraphOptimizationProfile};

/// A wrapper for MPSGraphExecutable objects
pub struct MPSGraphExecutable(pub(crate) *mut AnyObject);

// Implement Send + Sync for the wrapper type
unsafe impl Send for MPSGraphExecutable {}
unsafe impl Sync for MPSGraphExecutable {}

/// Result type for graph execution
pub type MPSGraphExecutionResult = HashMap<MPSGraphTensor, MPSGraphTensorData>;

impl MPSGraphExecutable {
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
    
    /// Encode the graph execution to a command buffer
    pub fn encode_to_command_buffer(&self, command_buffer: &CommandBuffer, feeds: &HashMap<MPSGraphTensor, MPSGraphTensorData>, output_tensors: &[MPSGraphTensor]) -> MPSGraphExecutionResult {
        unsafe {
            // Get the command buffer pointer
            let buffer_ptr = command_buffer.as_ptr() as *mut AnyObject;
            
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
            
            // Encode to command buffer
            let results: *mut AnyObject = msg_send![self.0, encodeToCommandBuffer: buffer_ptr,
                feeds: feed_dict,
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
    
    /// Encode the graph execution to a command buffer with a descriptor
    pub fn encode_to_command_buffer_with_descriptor(&self, command_buffer: &CommandBuffer, feeds: &HashMap<MPSGraphTensor, MPSGraphTensorData>, output_tensors: &[MPSGraphTensor], execution_descriptor: &MPSGraphExecutionDescriptor) -> MPSGraphExecutionResult {
        unsafe {
            // Get the command buffer pointer
            let buffer_ptr = command_buffer.as_ptr() as *mut AnyObject;
            
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
            
            // Encode to command buffer
            let results: *mut AnyObject = msg_send![self.0, encodeToCommandBuffer: buffer_ptr,
                feeds: feed_dict,
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
    pub fn set_wait_until_completed(&self, wait: bool) {
        unsafe {
            let _: () = msg_send![self.0, setWaitUntilCompleted: wait];
        }
    }
    
    /// Set scheduled handler
    pub fn set_scheduled_handler(&self, handler: extern "C" fn()) {
        unsafe {
            let _: () = msg_send![self.0, setScheduledHandler: handler];
        }
    }
    
    /// Set completion handler
    pub fn set_completion_handler(&self, handler: extern "C" fn()) {
        unsafe {
            let _: () = msg_send![self.0, setCompletionHandler: handler];
        }
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