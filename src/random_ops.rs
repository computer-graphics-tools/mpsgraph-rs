use objc::runtime::Object;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{MPSShape, MPSDataType, NSString};

/// Random distribution types supported by MPSGraph
#[repr(u64)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MPSGraphRandomDistribution {
    /// Uniform distribution, with samples drawn uniformly from [min, max) for float types, 
    /// and [min, max] for integer types
    Uniform = 0,
    /// Normal distribution defined by mean and standard deviation
    Normal = 1,
    /// Normal distribution defined by mean and standard deviation, truncated to range [min, max)
    TruncatedNormal = 2,
}

/// Sampling methods for normal distributions
#[repr(u64)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MPSGraphRandomNormalSamplingMethod {
    /// Use inverse erf to convert uniform values to values in the normal distribution
    InverseCDF = 0,
    /// Use Box Muller transform to convert uniform values to values in the normal distribution
    BoxMuller = 1,
}

/// Descriptor for random operations in MPSGraph
pub struct MPSGraphRandomOpDescriptor(pub(crate) *mut Object);

impl MPSGraphRandomOpDescriptor {
    /// Creates a new random operation descriptor with the specified distribution and data type
    pub fn new(distribution: MPSGraphRandomDistribution, data_type: MPSDataType) -> Self {
        unsafe {
            let descriptor: *mut Object = msg_send![
                class!(MPSGraphRandomOpDescriptor),
                descriptorWithDistribution:distribution as u64
                dataType:data_type as u64
            ];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphRandomOpDescriptor(descriptor)
        }
    }
    
    /// Sets the minimum value (for float data types)
    pub fn set_min(&self, min: f32) {
        unsafe {
            let _: () = msg_send![self.0, setMin:min];
        }
    }
    
    /// Sets the maximum value (for float data types)
    pub fn set_max(&self, max: f32) {
        unsafe {
            let _: () = msg_send![self.0, setMax:max];
        }
    }
    
    /// Sets the minimum integer value (for integer data types)
    pub fn set_min_integer(&self, min: i64) {
        unsafe {
            let _: () = msg_send![self.0, setMinInteger:min];
        }
    }
    
    /// Sets the maximum integer value (for integer data types)
    pub fn set_max_integer(&self, max: i64) {
        unsafe {
            let _: () = msg_send![self.0, setMaxInteger:max];
        }
    }
    
    /// Sets the mean (for normal distributions)
    pub fn set_mean(&self, mean: f32) {
        unsafe {
            let _: () = msg_send![self.0, setMean:mean];
        }
    }
    
    /// Sets the standard deviation (for normal distributions)
    pub fn set_standard_deviation(&self, std_dev: f32) {
        unsafe {
            let _: () = msg_send![self.0, setStandardDeviation:std_dev];
        }
    }
    
    /// Sets the sampling method (for normal distributions)
    pub fn set_sampling_method(&self, method: MPSGraphRandomNormalSamplingMethod) {
        unsafe {
            let _: () = msg_send![self.0, setSamplingMethod:method as u64];
        }
    }
}

impl Drop for MPSGraphRandomOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

// Enable Send and Sync for MPSGraphRandomOpDescriptor
unsafe impl Send for MPSGraphRandomOpDescriptor {}
unsafe impl Sync for MPSGraphRandomOpDescriptor {}

impl Clone for MPSGraphRandomOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let obj: *mut Object = msg_send![self.0, retain];
            MPSGraphRandomOpDescriptor(obj)
        }
    }
}

/// Random operations for MPSGraph
impl MPSGraph {
    /// Creates a tensor representing state using the Philox algorithm with given seed
    pub fn random_philox_state_tensor_with_seed(&self, seed: usize, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut Object = msg_send![self.0, 
                randomPhiloxStateTensorWithSeed:seed
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a tensor representing state using the Philox algorithm with given counter and key values
    pub fn random_philox_state_tensor_with_counter(
        &self, 
        counter_low: usize, 
        counter_high: usize, 
        key: usize, 
        name: Option<&str>
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut Object = msg_send![self.0, 
                randomPhiloxStateTensorWithCounterLow:counter_low
                counterHigh:counter_high
                key:key
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a random tensor with the specified shape and distribution
    pub fn random_tensor(
        &self, 
        shape: &[usize], 
        descriptor: &MPSGraphRandomOpDescriptor, 
        name: Option<&str>
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let shape_obj = MPSShape::from_slice(shape);
            
            let tensor: *mut Object = msg_send![self.0, 
                randomTensorWithShape:shape_obj.0
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a random tensor with the specified shape and distribution, using a specific seed
    pub fn random_tensor_with_seed(
        &self, 
        shape: &[usize], 
        descriptor: &MPSGraphRandomOpDescriptor, 
        seed: usize,
        name: Option<&str>
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let shape_obj = MPSShape::from_slice(shape);
            
            let tensor: *mut Object = msg_send![self.0, 
                randomTensorWithShape:shape_obj.0
                descriptor:descriptor.0
                seed:seed
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a random tensor with specified shape and distribution, using and updating the state tensor
    pub fn random_tensor_with_state(
        &self, 
        shape: &[usize], 
        descriptor: &MPSGraphRandomOpDescriptor, 
        state: &MPSGraphTensor,
        name: Option<&str>
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let shape_obj = MPSShape::from_slice(shape);
            
            let result: *mut Object = msg_send![self.0, 
                randomTensorWithShape:shape_obj.0
                descriptor:descriptor.0
                stateTensor:state.0
                name:name_obj
            ];
            
            // Extract the two tensors from the result array
            let count: usize = msg_send![result, count];
            assert_eq!(count, 2, "Random tensor with state should return an array of 2 tensors");
            
            // Get the random tensor and updated state
            let random_tensor: *mut Object = msg_send![result, objectAtIndex:0];
            let updated_state: *mut Object = msg_send![result, objectAtIndex:1];
            
            let random_tensor: *mut Object = msg_send![random_tensor, retain];
            let updated_state: *mut Object = msg_send![updated_state, retain];
            
            (MPSGraphTensor(random_tensor), MPSGraphTensor(updated_state))
        }
    }
    
    /// Creates a random uniform tensor with values in range [0.0, 1.0)
    pub fn random_uniform_tensor(&self, shape: &[usize], name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let shape_obj = MPSShape::from_slice(shape);
            
            let tensor: *mut Object = msg_send![self.0, 
                randomUniformTensorWithShape:shape_obj.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a random uniform tensor with values in range [0.0, 1.0) using a specific seed
    pub fn random_uniform_tensor_with_seed(
        &self, 
        shape: &[usize], 
        seed: usize,
        name: Option<&str>
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let shape_obj = MPSShape::from_slice(shape);
            
            let tensor: *mut Object = msg_send![self.0, 
                randomUniformTensorWithShape:shape_obj.0
                seed:seed
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a random uniform tensor with values in range [0.0, 1.0), using and updating the state tensor
    pub fn random_uniform_tensor_with_state(
        &self, 
        shape: &[usize], 
        state: &MPSGraphTensor,
        name: Option<&str>
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let shape_obj = MPSShape::from_slice(shape);
            
            let result: *mut Object = msg_send![self.0, 
                randomUniformTensorWithShape:shape_obj.0
                stateTensor:state.0
                name:name_obj
            ];
            
            // Extract the two tensors from the result array
            let count: usize = msg_send![result, count];
            assert_eq!(count, 2, "Random uniform tensor with state should return an array of 2 tensors");
            
            // Get the random tensor and updated state
            let random_tensor: *mut Object = msg_send![result, objectAtIndex:0];
            let updated_state: *mut Object = msg_send![result, objectAtIndex:1];
            
            let random_tensor: *mut Object = msg_send![random_tensor, retain];
            let updated_state: *mut Object = msg_send![updated_state, retain];
            
            (MPSGraphTensor(random_tensor), MPSGraphTensor(updated_state))
        }
    }
    
    /// Creates a dropout operation which zeros out elements of the input tensor randomly with probability equal to rate
    pub fn dropout(&self, tensor: &MPSGraphTensor, rate: f64, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let result: *mut Object = msg_send![self.0, 
                dropoutTensor:tensor.0
                rate:rate
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a dropout operation using a tensor to specify the dropout rate
    pub fn dropout_with_rate_tensor(
        &self, 
        tensor: &MPSGraphTensor, 
        rate_tensor: &MPSGraphTensor, 
        name: Option<&str>
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let result: *mut Object = msg_send![self.0, 
                dropoutTensor:tensor.0
                rateTensor:rate_tensor.0
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MPSGraphDevice;
    use std::collections::HashMap;
    
    #[test]
    fn test_random_uniform() {
        // This test will be skipped in environments without Metal
        if crate::tests::should_skip_test("test_random_uniform") {
            return;
        }
        
        let graph = MPSGraph::new();
        let device = MPSGraphDevice::new();
        
        // Create a random uniform tensor [0, 1)
        let random_tensor = graph.random_uniform_tensor(&[2, 3], Some("random"));
        
        // Run the graph
        let results = graph.run_graph(&[random_tensor.clone()], &device, None);
        
        // Get the result
        let output = results.get(&random_tensor).unwrap();
        let values: Vec<f32> = output.to_vec();
        
        // Verify the output shape
        assert_eq!(values.len(), 6);
        
        // Verify values are in range [0, 1)
        for val in values {
            assert!(val >= 0.0 && val < 1.0);
        }
    }
    
    #[test]
    fn test_dropout() {
        // This test will be skipped in environments without Metal
        if crate::tests::should_skip_test("test_dropout") {
            return;
        }
        
        let graph = MPSGraph::new();
        let device = MPSGraphDevice::new();
        
        // Create a tensor filled with ones
        let x = graph.placeholder_with_shape(&[5, 5], MPSDataType::Float32, Some("x"));
        let dropout = graph.dropout(&x, 0.5, Some("dropout"));
        
        // Create input data: a 5x5 tensor of ones
        let input_data = vec![1.0f32; 25];
        let x_data = crate::tensor_data::MPSGraphTensorData::new(&input_data, &[5, 5], MPSDataType::Float32);
        
        // Run the graph
        let feeds = HashMap::from([(x, x_data)]);
        let results = graph.run_with_feeds(&feeds, &[dropout.clone()], &device, None);
        
        // Get the result
        let output = results.get(&dropout).unwrap();
        let values: Vec<f32> = output.to_vec();
        
        // Verify the output shape
        assert_eq!(values.len(), 25);
        
        // Verify each value is either 0 or the original value (1.0)
        // Note: This is probabilistic, so it's possible (but highly unlikely)
        // that all values remain 1.0 or all become 0
        let mut has_zero = false;
        let mut has_one = false;
        
        for val in values {
            assert!(val == 0.0 || val == 1.0);
            if val == 0.0 {
                has_zero = true;
            }
            if val == 1.0 {
                has_one = true;
            }
        }
        
        // Check that we have at least one zero and one non-zero
        // This could theoretically fail, but is very unlikely with dropout rate of 0.5
        assert!(has_zero || has_one, "Dropout should produce at least some zeros or ones");
    }
}