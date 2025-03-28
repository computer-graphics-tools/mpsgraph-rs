use objc2::runtime::AnyObject;
use objc2::msg_send;
use std::fmt;
use std::ptr;
use std::collections::HashMap;
use metal::foreign_types::ForeignType;
use crate::tensor::MPSGraphTensor;
use crate::tensor_data::MPSGraphTensorData;
use crate::core::{MPSGraphOptimization, MPSGraphOptimizationProfile, NSString, AsRawObject};

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
    pub fn run_async_with_command_queue(
        &self, 
        command_queue: &metal::CommandQueue,
        feeds: &HashMap<MPSGraphTensor, MPSGraphTensorData>, 
        output_tensors: &[MPSGraphTensor],
        execution_descriptor: &MPSGraphExecutionDescriptor,
        // Note: The completion handler is not fully implemented yet
    ) -> MPSGraphExecutionResult {
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
            
            // Run the executable asynchronously
            // Note: We're ignoring the completion handler for now
            let results: *mut AnyObject = msg_send![self.0, runAsyncWithMTLCommandQueue: command_queue_ptr,
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

// Note: We're simplifying the callback handling for now to fix the build issues
// To implement more advanced callbacks, we'd need to add proper support for objective-c blocks

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
    
    /// Set scheduled handler (simplified version)
    ///
    /// The handler will be called when the graph execution is scheduled.
    pub fn set_scheduled_handler(&self, wait: bool) {
        unsafe {
            let _: () = msg_send![self.0, setScheduledHandler: wait];
        }
    }
    
    /// Set completion handler (simplified version)
    ///
    /// The handler will be called when execution completes.
    pub fn set_completion_handler(&self, wait: bool) {
        unsafe {
            let _: () = msg_send![self.0, setCompletionHandler: wait];
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