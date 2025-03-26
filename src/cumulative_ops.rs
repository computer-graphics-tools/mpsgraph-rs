use objc::runtime::{Object, YES, NO};
use objc::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::NSString;

/// Cumulative operations for MPSGraph
impl MPSGraph {
    /// Computes the cumulative sum of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension where you compute the cumulative operation
    /// * `exclusive` - If true, perform the exclusive cumulative operation, and the first element will be equal to zero
    /// * `reverse` - If true, reverse the direction of the cumulative operation along the specified axis
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object
    pub fn cumulative_sum(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        let exclusive_obj = if exclusive { YES } else { NO };
        let reverse_obj = if reverse { YES } else { NO };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                cumulativeSumWithTensor:tensor.0
                axis:axis
                exclusive:exclusive_obj
                reverse:reverse_obj
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Computes the cumulative sum of the input tensor along the specified axis using an axis tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis_tensor` - The tensor containing the axis to compute the cumulative operation on
    /// * `exclusive` - If true, perform the exclusive cumulative operation, and the first element will be equal to zero
    /// * `reverse` - If true, reverse the direction of the cumulative operation along the specified axis
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object
    pub fn cumulative_sum_with_axis_tensor(
        &self,
        tensor: &MPSGraphTensor,
        axis_tensor: &MPSGraphTensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        let exclusive_obj = if exclusive { YES } else { NO };
        let reverse_obj = if reverse { YES } else { NO };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                cumulativeSumWithTensor:tensor.0
                axisTensor:axis_tensor.0
                exclusive:exclusive_obj
                reverse:reverse_obj
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Computes the cumulative sum of the input tensor along the specified axis (simplified version).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension where you compute the cumulative operation
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object
    pub fn cumulative_sum_simple(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                cumulativeSumWithTensor:tensor.0
                axis:axis
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Computes the cumulative product of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension where you compute the cumulative operation
    /// * `exclusive` - If true, perform the exclusive cumulative operation, and the first element will be equal to one
    /// * `reverse` - If true, reverse the direction of the cumulative operation along the specified axis
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object
    pub fn cumulative_product(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        let exclusive_obj = if exclusive { YES } else { NO };
        let reverse_obj = if reverse { YES } else { NO };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                cumulativeProductWithTensor:tensor.0
                axis:axis
                exclusive:exclusive_obj
                reverse:reverse_obj
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Computes the cumulative minimum of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension where you compute the cumulative operation
    /// * `exclusive` - If true, perform the exclusive cumulative operation, and the first element will be equal to the largest value of the tensor data type
    /// * `reverse` - If true, reverse the direction of the cumulative operation along the specified axis
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object
    pub fn cumulative_minimum(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        let exclusive_obj = if exclusive { YES } else { NO };
        let reverse_obj = if reverse { YES } else { NO };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                cumulativeMinimumWithTensor:tensor.0
                axis:axis
                exclusive:exclusive_obj
                reverse:reverse_obj
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Computes the cumulative maximum of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension where you compute the cumulative operation
    /// * `exclusive` - If true, perform the exclusive cumulative operation, and the first element will be equal to the lowest value of the tensor data type
    /// * `reverse` - If true, reverse the direction of the cumulative operation along the specified axis
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object
    pub fn cumulative_maximum(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        let exclusive_obj = if exclusive { YES } else { NO };
        let reverse_obj = if reverse { YES } else { NO };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                cumulativeMaximumWithTensor:tensor.0
                axis:axis
                exclusive:exclusive_obj
                reverse:reverse_obj
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
    use crate::{MPSDataType, tests::should_skip_test};
    use crate::core::MPSShape;
    use std::collections::HashMap;
    
    #[test]
    fn test_cumulative_sum() {
        if should_skip_test("test_cumulative_sum") {
            return;
        }
        
        let graph = MPSGraph::new();
        
        // Create a simple 1D tensor
        let tensor_shape = MPSShape::from_slice(&[5]);
        let tensor_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        
        let tensor = graph.placeholder(&tensor_shape, MPSDataType::Float32, None);
        
        // Apply cumulative sum
        let cumsum = graph.cumulative_sum_simple(
            &tensor,
            0, // axis 0 (the only axis in this 1D tensor)
            Some("cumsum")
        );
        
        // Run the graph
        let mut feeds = HashMap::new();
        feeds.insert(&tensor, crate::MPSGraphTensorData::new(&tensor_data, &[5], MPSDataType::Float32));
        
        let results = graph.run(feeds, &[&cumsum]);
        
        // Get the result data - should be [1, 3, 6, 10, 15]
        let result_data = results[&cumsum].to_vec::<f32>();
        
        assert_eq!(result_data.len(), 5);
        assert_eq!(result_data[0], 1.0);
        assert_eq!(result_data[1], 3.0);
        assert_eq!(result_data[2], 6.0);
        assert_eq!(result_data[3], 10.0);
        assert_eq!(result_data[4], 15.0);
    }
    
    #[test]
    fn test_cumulative_product() {
        if should_skip_test("test_cumulative_product") {
            return;
        }
        
        let graph = MPSGraph::new();
        
        // Create a simple 1D tensor
        let tensor_shape = MPSShape::from_slice(&[4]);
        let tensor_data = vec![2.0f32, 3.0, 4.0, 5.0];
        
        let tensor = graph.placeholder(&tensor_shape, MPSDataType::Float32, None);
        
        // Apply cumulative product
        let cumprod = graph.cumulative_product(
            &tensor,
            0, // axis 0 (the only axis in this 1D tensor)
            false, // not exclusive
            false, // not reversed
            Some("cumprod")
        );
        
        // Run the graph
        let mut feeds = HashMap::new();
        feeds.insert(&tensor, crate::MPSGraphTensorData::new(&tensor_data, &[4], MPSDataType::Float32));
        
        let results = graph.run(feeds, &[&cumprod]);
        
        // Get the result data - should be [2, 6, 24, 120]
        let result_data = results[&cumprod].to_vec::<f32>();
        
        assert_eq!(result_data.len(), 4);
        assert_eq!(result_data[0], 2.0);
        assert_eq!(result_data[1], 6.0);
        assert_eq!(result_data[2], 24.0);
        assert_eq!(result_data[3], 120.0);
    }
}