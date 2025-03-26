use objc::runtime::Object;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::NSString;

/// The type of reduction applied in loss operations
#[repr(u64)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MPSGraphLossReductionType {
    /// Computes the loss without reduction
    None = 0,
    /// Reduces the loss down to a scalar with a sum operation
    Sum = 1,
    /// Reduces the loss down to a scalar with a mean operation
    Mean = 2,
}

// Alias for backward compatibility
#[allow(non_upper_case_globals)]
pub const Axis: MPSGraphLossReductionType = MPSGraphLossReductionType::None;

/// Loss operations for MPSGraph
impl MPSGraph {
    /// Creates a softmax cross-entropy loss operation and returns the result tensor.
    ///
    /// The softmax cross-entropy operation computes:
    /// ```
    /// loss = reduction(-labels * ln(softmax(source))), where
    /// softmax(source) = exp(source) / sum(exp(source))
    /// ```
    ///
    /// # Parameters
    ///
    /// * `source_tensor` - The source tensor (logits)
    /// * `labels_tensor` - The labels tensor (ground truth)
    /// * `axis` - The axis over which the operation computes softmax
    /// * `reduction_type` - The type of reduction to apply
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tensor containing the computed loss
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mpsgraph_rs::prelude::*;
    /// # use mpsgraph_rs::loss_ops::MPSGraphLossReductionType;
    /// # let graph = MPSGraph::new();
    /// # let logits = graph.placeholder(&[2, 3], MPSDataType::Float32, None);
    /// # let labels = graph.placeholder(&[2, 3], MPSDataType::Float32, None);
    /// // Calculate softmax cross entropy loss
    /// let loss = graph.softmax_cross_entropy(
    ///     &logits, 
    ///     &labels, 
    ///     1, 
    ///     MPSGraphLossReductionType::Mean, 
    ///     None
    /// );
    /// ```
    pub fn softmax_cross_entropy(
        &self,
        source_tensor: &MPSGraphTensor,
        labels_tensor: &MPSGraphTensor,
        axis: i64,
        reduction_type: MPSGraphLossReductionType,
        name: Option<&str>
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut Object = msg_send![self.0, 
                softMaxCrossEntropyWithSourceTensor:source_tensor.0
                labelsTensor:labels_tensor.0
                axis:axis
                reductionType:reduction_type as u64
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates the gradient of a softmax cross-entropy loss operation and returns the result tensor.
    ///
    /// # Parameters
    ///
    /// * `gradient_tensor` - The incoming gradient tensor (typically a constant tensor with value 1)
    /// * `source_tensor` - The original source tensor (logits)
    /// * `labels_tensor` - The original labels tensor (ground truth)
    /// * `axis` - The axis over which the operation computes softmax
    /// * `reduction_type` - The type of reduction that was applied
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tensor containing the gradient with respect to the source tensor
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mpsgraph_rs::prelude::*;
    /// # use mpsgraph_rs::loss_ops::MPSGraphLossReductionType;
    /// # let graph = MPSGraph::new();
    /// # let logits = graph.placeholder(&[2, 3], MPSDataType::Float32, None);
    /// # let labels = graph.placeholder(&[2, 3], MPSDataType::Float32, None);
    /// # let loss = graph.softmax_cross_entropy(&logits, &labels, 1, MPSGraphLossReductionType::Mean, None);
    /// // Create gradient of 1.0 for the loss (scalar)
    /// let grad_const = graph.constant_scalar(1.0, MPSDataType::Float32, None);
    /// 
    /// // Calculate gradient of loss with respect to logits
    /// let logits_grad = graph.softmax_cross_entropy_gradient(
    ///     &grad_const,
    ///     &logits,
    ///     &labels,
    ///     1,
    ///     MPSGraphLossReductionType::Mean,
    ///     None
    /// );
    /// ```
    pub fn softmax_cross_entropy_gradient(
        &self,
        gradient_tensor: &MPSGraphTensor,
        source_tensor: &MPSGraphTensor,
        labels_tensor: &MPSGraphTensor,
        axis: i64,
        reduction_type: MPSGraphLossReductionType,
        name: Option<&str>
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut Object = msg_send![self.0, 
                softMaxCrossEntropyGradientWithIncomingGradientTensor:gradient_tensor.0
                sourceTensor:source_tensor.0
                labelsTensor:labels_tensor.0
                axis:axis
                reductionType:reduction_type as u64
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MPSDataType;
    use crate::device::MPSGraphDevice;
    use std::collections::HashMap;
    
    #[test]
    fn test_softmax_cross_entropy() {
        // This test will be skipped in environments without Metal
        if crate::tests::should_skip_test("test_softmax_cross_entropy") {
            return;
        }
        
        let graph = MPSGraph::new();
        let device = MPSGraphDevice::new();
        
        // Create a simple classification problem:
        // Logits: [[1.0, 2.0, 0.1], [0.1, 1.0, 2.0]]
        // Labels: [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] (one-hot encoded)
        
        // Create tensors
        let logits = graph.placeholder_with_shape(&[2, 3], MPSDataType::Float32, Some("logits"));
        let labels = graph.placeholder_with_shape(&[2, 3], MPSDataType::Float32, Some("labels"));
        
        // Calculate loss with mean reduction
        let loss = graph.softmax_cross_entropy(
            &logits,
            &labels,
            1, // axis 1 (class dimension)
            MPSGraphLossReductionType::Mean,
            Some("loss")
        );
        
        // Create input data
        let logits_data = vec![1.0f32, 2.0, 0.1, 0.1, 1.0, 2.0];
        let labels_data = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0];
        
        let logits_tensor_data = crate::tensor_data::MPSGraphTensorData::new(
            &logits_data, 
            &[2, 3], 
            MPSDataType::Float32
        );
        
        let labels_tensor_data = crate::tensor_data::MPSGraphTensorData::new(
            &labels_data, 
            &[2, 3], 
            MPSDataType::Float32
        );
        
        // Run the graph
        let feeds = HashMap::from([
            (logits, logits_tensor_data),
            (labels, labels_tensor_data),
        ]);
        
        let results = graph.run_with_feeds(&feeds, &[loss.clone()], &device, None);
        
        // Get the loss value
        let loss_data = results.get(&loss).unwrap();
        let loss_value: Vec<f32> = loss_data.to_vec();
        
        // Expected loss calculation:
        // For first example:
        //   softmax([1.0, 2.0, 0.1]) = [0.23, 0.63, 0.13]
        //   -labels * log(softmax) = -[1.0, 0.0, 0.0] * log([0.23, 0.63, 0.13]) = -[log(0.23), 0, 0] = [1.47, 0, 0]
        //   sum = 1.47
        //
        // For second example:
        //   softmax([0.1, 1.0, 2.0]) = [0.13, 0.23, 0.63]
        //   -labels * log(softmax) = -[0.0, 0.0, 1.0] * log([0.13, 0.23, 0.63]) = -[0, 0, log(0.63)] = [0, 0, 0.46]
        //   sum = 0.46
        //
        // Mean of [1.47, 0.46] = 0.965
        
        // Allow some floating point error in comparison
        assert!((loss_value[0] - 0.965).abs() < 0.01, 
                "Expected loss close to 0.965, got {}", loss_value[0]);
    }
}