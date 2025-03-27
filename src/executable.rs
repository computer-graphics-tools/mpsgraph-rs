use objc2::runtime::AnyObject;
use objc2::msg_send;
use std::fmt;
use std::ptr;
use std::collections::HashMap;
use metal::{CommandQueue, CommandBuffer, SharedEvent};
use metal::foreign_types::ForeignType;
use crate::tensor::MPSGraphTensor;
use crate::tensor_data::MPSGraphTensorData;
use crate::core::{OurNSDictionary, OurNSArray, MPSGraphOptimization, MPSGraphOptimizationProfile, MPSGraphExecutionStage};

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
                targetOperations: std::ptr::null_mut::<AnyObject>(),
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
                targetOperations: std::ptr::null_mut::<AnyObject>(),
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
        self.encode_to_command_buffer_with_descriptor(command_buffer, feeds, output_tensors, None)
    }
    
    /// Encode the graph execution into a command buffer with an execution descriptor
    pub fn encode_to_command_buffer_with_descriptor(&self,
                                              command_buffer:  &CommandBuffer,
                                              feeds:  HashMap<MPSGraphTensor, MPSGraphTensorData>,
                                              output_tensors:  &[MPSGraphTensor],
                                              descriptor:  Option<&MPSGraphExecutionDescriptor>) -> MPSGraphExecutionResult {
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
            
            // Get the descriptor pointer, if provided
            let descriptor_ptr = match descriptor {
                Some(d) => d.0,
                None => std::ptr::null_mut(),
            };
            
            // Execute the graph
            let command_buffer_ptr = command_buffer.as_ptr() as *mut std::ffi::c_void;
            let results: *mut AnyObject = msg_send![self.0, encodeToCommandBuffer: command_buffer_ptr,
                feeds: feed_dict.0,
                targetTensors: output_tensors_array.0,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
                executionDescriptor: descriptor_ptr,
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
    
    /// Run the executable asynchronously with a command queue
    pub fn run_async_with_command_queue(&self,
                                  command_queue:  &CommandQueue,
                                  feeds:  HashMap<MPSGraphTensor, MPSGraphTensorData>,
                                  output_tensors:  &[MPSGraphTensor],
                                  descriptor:  Option<&MPSGraphExecutionDescriptor>) -> MPSGraphExecutionResult {
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
            
            // Get the descriptor pointer, if provided
            let descriptor_ptr = match descriptor {
                Some(d) => d.0,
                None => std::ptr::null_mut(),
            };
            
            // Execute the graph
            let command_queue_ptr = command_queue.as_ptr() as *mut std::ffi::c_void;
            let results: *mut AnyObject = msg_send![self.0, runAsyncWithMTLCommandQueue: command_queue_ptr,
                feeds: feed_dict.0,
                targetTensors: output_tensors_array.0,
                targetOperations: std::ptr::null_mut::<AnyObject>(),
                executionDescriptor: descriptor_ptr,
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

/// A wrapper for MPSGraphCompilationDescriptor objects
pub struct MPSGraphCompilationDescriptor(pub(crate) *mut AnyObject);

// Implement Send + Sync for the wrapper type
unsafe impl Send for MPSGraphCompilationDescriptor {}
unsafe impl Sync for MPSGraphCompilationDescriptor {}

impl MPSGraphCompilationDescriptor {
    /// Creates a new MPSGraphCompilationDescriptor
    pub fn new() -> Self {
        unsafe {
            // Create the descriptor
            let class_name = c"MPSGraphCompilationDescriptor";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            let obj: *mut AnyObject = msg_send![cls, alloc];
            let descriptor: *mut AnyObject = msg_send![obj, init];
            
            MPSGraphCompilationDescriptor(descriptor)
        }
    }
    
    /// Turns off type inference and relies on type inference during runtime
    pub fn disable_type_inference(&self) {
        unsafe {
            let _: () = msg_send![self.0, disableTypeInference];
        }
    }
    
    /// Sets the optimization level for the graph execution
    pub fn set_optimization_level(&self, level: MPSGraphOptimization) {
        unsafe {
            let _: () = msg_send![self.0, setOptimizationLevel: level as u64];
        }
    }
    
    /// Gets the current optimization level
    pub fn optimization_level(&self) -> MPSGraphOptimization {
        unsafe {
            let value: u64 = msg_send![self.0, optimizationLevel];
            std::mem::transmute(value)
        }
    }
    
    /// Sets whether to make the compile or specialize call blocking
    pub fn set_wait_for_compilation_completion(&self, wait: bool) {
        unsafe {
            let _: () = msg_send![self.0, setWaitForCompilationCompletion: wait];
        }
    }
    
    /// Gets the current wait for compilation completion setting
    pub fn wait_for_compilation_completion(&self) -> bool {
        unsafe {
            let value: bool = msg_send![self.0, waitForCompilationCompletion];
            value
        }
    }
    
    /// Sets the optimization profile for the graph optimization
    /// Note: Deprecated as of macOS 14.0, iOS 17.0
    pub fn set_optimization_profile(&self, profile: MPSGraphOptimizationProfile) {
        unsafe {
            let _: () = msg_send![self.0, setOptimizationProfile: profile as u64];
        }
    }
    
    /// Gets the current optimization profile
    /// Note: Deprecated as of macOS 14.0, iOS 17.0
    pub fn optimization_profile(&self) -> MPSGraphOptimizationProfile {
        unsafe {
            let value: u64 = msg_send![self.0, optimizationProfile];
            std::mem::transmute(value)
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
                // Use the NSCopying protocol to make a proper copy
                let copy: *mut AnyObject = msg_send![self.0, copy];
                MPSGraphCompilationDescriptor(copy)
            } else {
                MPSGraphCompilationDescriptor(ptr::null_mut())
            }
        }
    }
}

impl fmt::Debug for MPSGraphCompilationDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MPSGraphCompilationDescriptor")
    }
}

/// A wrapper for MPSGraphExecutionDescriptor objects
pub struct MPSGraphExecutionDescriptor(pub(crate) *mut AnyObject);

// Implement Send + Sync for the wrapper type
unsafe impl Send for MPSGraphExecutionDescriptor {}
unsafe impl Sync for MPSGraphExecutionDescriptor {}

impl MPSGraphExecutionDescriptor {
    /// Creates a new MPSGraphExecutionDescriptor
    pub fn new() -> Self {
        unsafe {
            // Create the descriptor
            let class_name = c"MPSGraphExecutionDescriptor";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            let obj: *mut AnyObject = msg_send![cls, alloc];
            let descriptor: *mut AnyObject = msg_send![obj, init];
            
            MPSGraphExecutionDescriptor(descriptor)
        }
    }
    
    /// Sets whether to block the execution call until the entire execution is complete
    pub fn set_wait_until_completed(&self, wait: bool) {
        unsafe {
            let _: () = msg_send![self.0, setWaitUntilCompleted: wait];
        }
    }
    
    /// Gets the current wait until completed setting
    pub fn wait_until_completed(&self) -> bool {
        unsafe {
            let value: bool = msg_send![self.0, waitUntilCompleted];
            value
        }
    }
    
    /// Sets the compilation descriptor for the graph
    pub fn set_compilation_descriptor(&self, descriptor: Option<&MPSGraphCompilationDescriptor>) {
        unsafe {
            let descriptor_ptr = match descriptor {
                Some(d) => d.0,
                None => ptr::null_mut(),
            };
            
            let _: () = msg_send![self.0, setCompilationDescriptor: descriptor_ptr];
        }
    }
    
    /// Executable waits on a shared event before scheduling execution
    pub fn wait_for_event(&self, event: &SharedEvent, value: u64) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut std::ffi::c_void;
            let _: () = msg_send![self.0, waitForEvent: event_ptr, value: value];
        }
    }
    
    /// Executable signals a shared event at the specified execution stage
    pub fn signal_event(&self, event: &SharedEvent, execution_stage: MPSGraphExecutionStage, value: u64) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut std::ffi::c_void;
            let _: () = msg_send![self.0, signalEvent: event_ptr, 
                             atExecutionEvent: execution_stage as u64, 
                             value: value];
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
        write!(f, "MPSGraphExecutionDescriptor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compilation_descriptor() {
        // Create a compilation descriptor
        let descriptor = MPSGraphCompilationDescriptor::new();
        
        // Set some properties
        descriptor.disable_type_inference();
        descriptor.set_optimization_level(MPSGraphOptimization::Level1);
        descriptor.set_wait_for_compilation_completion(true);
        
        // Verify properties
        assert_eq!(descriptor.optimization_level(), MPSGraphOptimization::Level1);
        assert_eq!(descriptor.wait_for_compilation_completion(), true);
    }
    
    #[test]
    fn test_execution_descriptor() {
        // Create an execution descriptor
        let descriptor = MPSGraphExecutionDescriptor::new();
        
        // Set some properties
        descriptor.set_wait_until_completed(true);
        
        // Verify properties
        assert_eq!(descriptor.wait_until_completed(), true);
        
        // Set a compilation descriptor
        let compilation_descriptor = MPSGraphCompilationDescriptor::new();
        descriptor.set_compilation_descriptor(Some(&compilation_descriptor));
    }
    
    #[test]
    fn test_compile_with_descriptor() {
        // Create a compilation descriptor
        let descriptor = MPSGraphCompilationDescriptor::new();
        descriptor.set_optimization_level(MPSGraphOptimization::Level1);
        descriptor.set_wait_for_compilation_completion(true);
        
        // Test the descriptor's properties
        assert_eq!(descriptor.optimization_level(), MPSGraphOptimization::Level1);
        assert_eq!(descriptor.wait_for_compilation_completion(), true);
        
        // Test cloning
        let descriptor_clone = descriptor.clone();
        assert_eq!(descriptor_clone.optimization_level(), MPSGraphOptimization::Level1);
    }
}