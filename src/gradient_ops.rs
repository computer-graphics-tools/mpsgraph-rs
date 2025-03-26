use objc::runtime::Object;
use std::collections::HashMap;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::NSString;
use crate::core::NSDictionary;

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
        primary_tensor: &MPSGraphTensor,
        tensors: &[MPSGraphTensor],
        name: Option<&str>
    ) -> HashMap<MPSGraphTensor, MPSGraphTensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            // Convert the Rust slice to an NSArray
            let tensors_array = crate::core::NSArray::from_slice(tensors);
            
            // Call the Objective-C method
            let dict: *mut Object = msg_send![self.0, 
                gradientForPrimaryTensor:primary_tensor.0
                withTensors:tensors_array.0
                name:name_obj
            ];
            
            // Convert NSDictionary to HashMap
            let dict = NSDictionary(dict);
            let mut result = HashMap::new();
            
            // Get the keys and values from the dictionary
            let keys: *mut Object = msg_send![dict.0, allKeys];
            let keys_count: usize = msg_send![keys, count];
            
            for i in 0..keys_count {
                let key: *mut Object = msg_send![keys, objectAtIndex:i];
                let key_retained: *mut Object = msg_send![key, retain];
                let key_tensor = MPSGraphTensor(key_retained);
                
                let value: *mut Object = msg_send![dict.0, objectForKey:key];
                let value_retained: *mut Object = msg_send![value, retain];
                let value_tensor = MPSGraphTensor(value_retained);
                
                result.insert(key_tensor, value_tensor);
            }
            
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MPSDataType;
    use crate::device::MPSGraphDevice;
    
    #[test]
    fn test_gradient() {
        // This test will be skipped in environments without Metal
        if crate::tests::should_skip_test("test_gradient") {
            return;
        }
        
        let graph = MPSGraph::new();
        let device = MPSGraphDevice::new();
        
        // Create tensors: f(x) = x^2
        let x = graph.placeholder_with_shape(&[2, 2], MPSDataType::Float32, Some("x"));
        let y = graph.square(&x, Some("y"));
        
        // Calculate gradient df/dx
        let grads = graph.gradient_for_primary_tensor(&y, &[x.clone()], Some("grad"));
        let dx = grads.get(&x).unwrap();
        
        // Create input data: x = [[1, 2], [3, 4]]
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let x_data = crate::tensor_data::MPSGraphTensorData::new(&input_data, &[2, 2], MPSDataType::Float32);
        
        // Expected gradient: df/dx = 2x, so dx = [[2, 4], [6, 8]]
        let expected = vec![2.0f32, 4.0, 6.0, 8.0];
        
        // Run the gradient calculation
        let feeds = HashMap::from([(x, x_data)]);
        let results = graph.run_with_feeds(&feeds, &[dx.clone()], &device, None);
        
        // Extract the results
        let dx_data = results.get(dx).unwrap();
        let dx_values: Vec<f32> = dx_data.to_vec();
        
        // Compare results with expected values
        assert_eq!(dx_values, expected);
    }
}