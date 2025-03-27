use objc2::runtime::AnyObject;
use objc2::msg_send;
use std::fmt;
use std::ptr;
use std::collections::HashMap;
use metal::{CommandQueue, CommandBuffer};
use metal::foreign_types::ForeignType;
use crate::tensor::MPSGraphTensor;
use crate::tensor_data::MPSGraphTensorData;
use crate::core::{OurNSDictionary, OurNSArray};

/// A wrapper for MPSGraphExecutable objects
pub struct MPSGraphExecutable(pub(crate) *mut AnyObject);

// Implement Send + Sync for the wrapper type
unsafe impl Send for MPSGraphExecutable {}
unsafe impl Sync for MPSGraphExecutable {}

/// Result type for graph execution
pub type MPSGraphExecutionResult = HashMap<MPSGraphTensor, MPSGraphTensorData>;

impl MPSGraphExecutable {
    /// Execute the graph using the provided inputs and list of required output tensors
    pub fn run_with_feeds(&self, 
                       feeds:  HashMap<MPSGraphTensor, MPSGraphTensorData>,
                       output_tensors:  &[MPSGraphTensor]) -> MPSGraphExecutionResult {
        unsafe {
            // Create feed dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = OurNSDictionary::from_keys_and_objects(&feed_keys, &feed_values);
            
            // Create output tensors array
            let output_tensors_raw: Vec<*mut AnyObject> = output_tensors.iter()
                .map(|t| t.0)
                .collect();
            
            let output_tensors_array = OurNSArray::from_objects(&output_tensors_raw);
            
            // Execute the graph
            let results: *mut AnyObject = msg_send![self.0, runWithFeeds: feed_dict.0,
                targetTensors: output_tensors_array.0,
                targetOperations: std::ptr::null_mut::<AnyObject>()
            ];
            
            // Parse the results
            let count: usize = msg_send![results, count];
            let mut output = HashMap::with_capacity(count);
            
            // Get enumerator for the dictionary
            let enumerator: *mut AnyObject = msg_send![results, keyEnumerator];
            
            while {
                let key: *mut AnyObject = msg_send![enumerator, nextObject];
                !key.is_null()
            } {
                let key: *mut AnyObject = msg_send![enumerator, currentObject];
                let value: *mut AnyObject = msg_send![results, objectForKey: key];
                
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
    
    /// Execute the graph using a Metal command queue
    pub fn run_with_command_queue(&self,
                             command_queue:  &CommandQueue,
                             feeds:  HashMap<MPSGraphTensor, MPSGraphTensorData>,
                             output_tensors:  &[MPSGraphTensor]) -> MPSGraphExecutionResult {
        unsafe {
            // Create feed dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = OurNSDictionary::from_keys_and_objects(&feed_keys, &feed_values);
            
            // Create output tensors array
            let output_tensors_raw: Vec<*mut AnyObject> = output_tensors.iter()
                .map(|t| t.0)
                .collect();
            
            let output_tensors_array = OurNSArray::from_objects(&output_tensors_raw);
            
            // Execute the graph
            let command_queue_ptr = command_queue.as_ptr() as *mut std::ffi::c_void;
            let results: *mut AnyObject = msg_send![self.0, runWithMTLCommandQueue: command_queue_ptr,
                feeds: feed_dict.0,
                targetTensors: output_tensors_array.0,
                targetOperations: std::ptr::null_mut::<AnyObject>()
            ];
            
            // Parse the results
            let count: usize = msg_send![results, count];
            let mut output = HashMap::with_capacity(count);
            
            // Get enumerator for the dictionary
            let enumerator: *mut AnyObject = msg_send![results, keyEnumerator];
            
            while {
                let key: *mut AnyObject = msg_send![enumerator, nextObject];
                !key.is_null()
            } {
                let key: *mut AnyObject = msg_send![enumerator, currentObject];
                let value: *mut AnyObject = msg_send![results, objectForKey: key];
                
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
    
    /// Encode the graph execution into a command buffer
    pub fn encode_to_command_buffer(&self,
                               command_buffer:  &CommandBuffer,
                               feeds:  HashMap<MPSGraphTensor, MPSGraphTensorData>,
                               output_tensors:  &[MPSGraphTensor]) -> MPSGraphExecutionResult {
        unsafe {
            // Create feed dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = OurNSDictionary::from_keys_and_objects(&feed_keys, &feed_values);
            
            // Create output tensors array
            let output_tensors_raw: Vec<*mut AnyObject> = output_tensors.iter()
                .map(|t| t.0)
                .collect();
            
            let output_tensors_array = OurNSArray::from_objects(&output_tensors_raw);
            
            // Execute the graph
            let command_buffer_ptr = command_buffer.as_ptr() as *mut std::ffi::c_void;
            let results: *mut AnyObject = msg_send![self.0, encodeToCommandBuffer: command_buffer_ptr,
                feeds: feed_dict.0,
                targetTensors: output_tensors_array.0,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
                executionDescriptor: std::ptr::null_mut::<AnyObject>()
            ];
            
            // Parse the results
            let count: usize = msg_send![results, count];
            let mut output = HashMap::with_capacity(count);
            
            // Get enumerator for the dictionary
            let enumerator: *mut AnyObject = msg_send![results, keyEnumerator];
            
            while {
                let key: *mut AnyObject = msg_send![enumerator, nextObject];
                !key.is_null()
            } {
                let key: *mut AnyObject = msg_send![enumerator, currentObject];
                let value: *mut AnyObject = msg_send![results, objectForKey: key];
                
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
        write!(f, "MPSGraphExecutable")
    }
}