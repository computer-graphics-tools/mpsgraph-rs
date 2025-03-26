use objc::runtime::Object;
use std::fmt;
use std::collections::HashMap;
use metal::{CommandQueue, CommandBuffer};
use metal::foreign_types::ForeignType;
use crate::tensor::MPSGraphTensor;
use crate::tensor_data::MPSGraphTensorData;
use crate::core::{NSDictionary, NSArray};

/// A wrapper for MPSGraphExecutable objects
pub struct MPSGraphExecutable(pub(crate) *mut Object);

// Implement Send + Sync for the wrapper type
unsafe impl Send for MPSGraphExecutable {}
unsafe impl Sync for MPSGraphExecutable {}

/// Result type for graph execution
pub type MPSGraphExecutionResult = HashMap<MPSGraphTensor, MPSGraphTensorData>;

impl MPSGraphExecutable {
    /// Execute the graph using the provided inputs and list of required output tensors
    pub fn run_with_feeds(&self, 
                       feeds: HashMap<MPSGraphTensor, MPSGraphTensorData>,
                       output_tensors: &[MPSGraphTensor]) -> MPSGraphExecutionResult {
        unsafe {
            // Create feed dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = NSDictionary::from_keys_and_objects(&feed_keys, &feed_values);
            
            // Create output tensors array
            let output_tensors_raw: Vec<*mut Object> = output_tensors.iter()
                .map(|t| t.0)
                .collect();
            
            let output_tensors_array = NSArray::from_objects(&output_tensors_raw);
            
            // Execute the graph
            let results: *mut Object = msg_send![self.0,
                runWithFeeds:feed_dict.0
                targetTensors:output_tensors_array.0
                targetOperations:std::ptr::null_mut::<Object>()
            ];
            
            // Parse the results
            let count: usize = msg_send![results, count];
            let mut output = HashMap::with_capacity(count);
            
            // Get enumerator for the dictionary
            let enumerator: *mut Object = msg_send![results, keyEnumerator];
            
            while {
                let key: *mut Object = msg_send![enumerator, nextObject];
                !key.is_null()
            } {
                let key: *mut Object = msg_send![enumerator, currentObject];
                let value: *mut Object = msg_send![results, objectForKey:key];
                
                // Retain objects to manage their memory
                let _: () = msg_send![key, retain];
                let _: () = msg_send![value, retain];
                
                output.insert(MPSGraphTensor(key), MPSGraphTensorData(value));
            }
            
            // Release temporary objects
            let _: () = msg_send![results, release];
            
            output
        }
    }
    
    /// Execute the graph using a Metal command queue
    pub fn run_with_command_queue(&self,
                             command_queue: &CommandQueue,
                             feeds: HashMap<MPSGraphTensor, MPSGraphTensorData>,
                             output_tensors: &[MPSGraphTensor]) -> MPSGraphExecutionResult {
        unsafe {
            // Create feed dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = NSDictionary::from_keys_and_objects(&feed_keys, &feed_values);
            
            // Create output tensors array
            let output_tensors_raw: Vec<*mut Object> = output_tensors.iter()
                .map(|t| t.0)
                .collect();
            
            let output_tensors_array = NSArray::from_objects(&output_tensors_raw);
            
            // Execute the graph
            let results: *mut Object = msg_send![self.0,
                runWithMTLCommandQueue:command_queue.as_ptr()
                feeds:feed_dict.0
                targetTensors:output_tensors_array.0
                targetOperations:std::ptr::null_mut::<Object>()
            ];
            
            // Parse the results
            let count: usize = msg_send![results, count];
            let mut output = HashMap::with_capacity(count);
            
            // Get enumerator for the dictionary
            let enumerator: *mut Object = msg_send![results, keyEnumerator];
            
            while {
                let key: *mut Object = msg_send![enumerator, nextObject];
                !key.is_null()
            } {
                let key: *mut Object = msg_send![enumerator, currentObject];
                let value: *mut Object = msg_send![results, objectForKey:key];
                
                // Retain objects to manage their memory
                let _: () = msg_send![key, retain];
                let _: () = msg_send![value, retain];
                
                output.insert(MPSGraphTensor(key), MPSGraphTensorData(value));
            }
            
            // Release temporary objects
            let _: () = msg_send![results, release];
            
            output
        }
    }
    
    /// Encode the graph execution into a command buffer
    pub fn encode_to_command_buffer(&self,
                               command_buffer: &CommandBuffer,
                               feeds: HashMap<MPSGraphTensor, MPSGraphTensorData>,
                               output_tensors: &[MPSGraphTensor]) -> MPSGraphExecutionResult {
        unsafe {
            // Create feed dictionary
            let mut feed_keys = Vec::with_capacity(feeds.len());
            let mut feed_values = Vec::with_capacity(feeds.len());
            
            for (tensor, data) in feeds {
                feed_keys.push(tensor.0);
                feed_values.push(data.0);
            }
            
            let feed_dict = NSDictionary::from_keys_and_objects(&feed_keys, &feed_values);
            
            // Create output tensors array
            let output_tensors_raw: Vec<*mut Object> = output_tensors.iter()
                .map(|t| t.0)
                .collect();
            
            let output_tensors_array = NSArray::from_objects(&output_tensors_raw);
            
            // Execute the graph
            let results: *mut Object = msg_send![self.0,
                encodeToCommandBuffer:command_buffer.as_ptr()
                feeds:feed_dict.0
                targetTensors:output_tensors_array.0
                targetOperations:std::ptr::null_mut::<Object>()
                executionDescriptor:std::ptr::null_mut::<Object>()
            ];
            
            // Parse the results
            let count: usize = msg_send![results, count];
            let mut output = HashMap::with_capacity(count);
            
            // Get enumerator for the dictionary
            let enumerator: *mut Object = msg_send![results, keyEnumerator];
            
            while {
                let key: *mut Object = msg_send![enumerator, nextObject];
                !key.is_null()
            } {
                let key: *mut Object = msg_send![enumerator, currentObject];
                let value: *mut Object = msg_send![results, objectForKey:key];
                
                // Retain objects to manage their memory
                let _: () = msg_send![key, retain];
                let _: () = msg_send![value, retain];
                
                output.insert(MPSGraphTensor(key), MPSGraphTensorData(value));
            }
            
            // Release temporary objects
            let _: () = msg_send![results, release];
            
            output
        }
    }
}

impl Drop for MPSGraphExecutable {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

impl Clone for MPSGraphExecutable {
    fn clone(&self) -> Self {
        unsafe {
            let obj: *mut Object = msg_send![self.0, retain];
            MPSGraphExecutable(obj)
        }
    }
}

impl fmt::Debug for MPSGraphExecutable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MPSGraphExecutable")
    }
}