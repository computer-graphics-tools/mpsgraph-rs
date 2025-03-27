use objc2::runtime::AnyObject;
use std::collections::HashMap;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, AsRawObject};

/// Gradient (Automatic Differentiation) operations for MPSGraph
impl MPSGraph {
    /// Calculates a partial derivative of primary_tensor with respect to the tensors.
    ///
    /// Returns a dictionary containing partial derivative d(primary_tensor)/d(secondary_tensor) for each tensor.
    ///
    /// # Parameters
    ///
    /// * `primary_tensor` - Tensor to be differentiated (numerator).
    /// * `tensors` - Tensors to do the differentiation with (denominator).
    /// * `name` - Optional name for the gradient operation.
    ///
    /// # Returns
    ///
    /// A HashMap mapping each input tensor to its gradient tensor.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mpsgraph_rs::prelude::*;
    /// # let graph = MPSGraph::new();
    /// # let x = graph.placeholder(&[2, 3], MPSDataType::Float32, None);
    /// # let y = graph.square(&x, None);
    /// // Calculate gradient dy/dx
    /// let grads = graph.gradient_for_primary_tensor(&y, &[x.clone()], None);
    /// let dx = grads.get(&x).unwrap();
    /// ```
    pub fn gradient_for_primary_tensor(
        &self,
        primary_tensor:  &MPSGraphTensor,
        tensors:  &[MPSGraphTensor],
        name:  Option<&str>
    ) -> HashMap<MPSGraphTensor, MPSGraphTensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Convert the Rust slice to an NSArray manually
            let tensor_ptrs: Vec<*mut AnyObject> = tensors.iter()
                .map(|tensor| tensor.0)
                .collect();
            
            // Create an NSArray from the raw pointers
            let array = crate::core::OurNSArray::from_objects(&tensor_ptrs);
            
            // Call the Objective-C method
            let dict: *mut AnyObject = msg_send![self.0, gradientForPrimaryTensor: primary_tensor.0, withTensors: array.0, name: name_obj];
            
            // We need to manually parse the NSDictionary since we can't directly use the new objc2 version
            // Convert NSDictionary to HashMap manually
            let mut result = HashMap::new();
            
            // Get the keys and values from the dictionary
            let keys: *mut AnyObject = msg_send![dict, allKeys];
            let keys_count: usize = msg_send![keys, count];
            
            for i in 0..keys_count {
                let key: *mut AnyObject = msg_send![keys, objectAtIndex: i];
                let key_retained = objc2::ffi::objc_retain(key as *mut _) as *mut AnyObject;
                let key_tensor = MPSGraphTensor(key_retained);
                
                let value: *mut AnyObject = msg_send![dict, objectForKey: key];
                let value_retained = objc2::ffi::objc_retain(value as *mut _) as *mut AnyObject;
                let value_tensor = MPSGraphTensor(value_retained);
                
                result.insert(key_tensor, value_tensor);
            }
            
            result
        }
    }
}