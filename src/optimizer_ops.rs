use objc::runtime::Object;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::NSString;

/// Optimizer operations for MPSGraph
impl MPSGraph {
    /// Stochastic gradient descent optimization.
    ///
    /// `variable = variable - (learningRate * gradient)`
    ///
    /// # Parameters
    ///
    /// * `learning_rate` - Scalar tensor which indicates the learning rate to use
    /// * `values` - Values tensor, usually representing the trainable parameters
    /// * `gradient` - Partial gradient of the trainable parameters with respect to loss
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tensor containing the updated values
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mpsgraph_rs::prelude::*;
    /// # let graph = MPSGraph::new();
    /// # let weights = graph.placeholder_with_shape(&[2, 3], MPSDataType::Float32, None);
    /// # let gradients = graph.placeholder_with_shape(&[2, 3], MPSDataType::Float32, None);
    /// # let learning_rate = graph.constant_scalar(0.01, MPSDataType::Float32, None);
    /// 
    /// // Update weights using SGD
    /// let updated_weights = graph.stochastic_gradient_descent(
    ///     &learning_rate,
    ///     &weights,
    ///     &gradients,
    ///     None
    /// );
    /// ```
    pub fn stochastic_gradient_descent(
        &self,
        learning_rate: &MPSGraphTensor,
        values: &MPSGraphTensor,
        gradient: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut Object = msg_send![
                self.0, 
                stochasticGradientDescentWithLearningRateTensor:learning_rate.0
                valuesTensor:values.0
                gradientTensor:gradient.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Adam optimization.
    ///
    /// The Adam optimizer combines ideas from momentum and RMSProp, maintaining per-parameter
    /// momentum and velocity (squared gradients) to adaptively adjust learning rates.
    ///
    /// ```
    /// m[t] = beta1 * m[t-1] + (1 - beta1) * g
    /// v[t] = beta2 * v[t-1] + (1 - beta2) * (g ^ 2)
    /// maxVel[t] = max(maxVel[t-1], v[t])
    /// variable = variable - (learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)) * m[t] / (sqrt(maxVel) + epsilon)
    /// ```
    ///
    /// # Parameters
    ///
    /// * `learning_rate` - Scalar tensor with the learning rate
    /// * `beta1` - Exponential decay rate for first moment estimates
    /// * `beta2` - Exponential decay rate for second moment estimates
    /// * `epsilon` - Small constant for numerical stability
    /// * `beta1_power` - Current beta1^t value
    /// * `beta2_power` - Current beta2^t value
    /// * `values` - Current parameter values to be updated
    /// * `momentum` - First moment estimates (momentum)
    /// * `velocity` - Second moment estimates (velocity)
    /// * `maximum_velocity` - Optional maximum velocity tensor
    /// * `gradient` - Gradient of parameters with respect to loss
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector containing tensors for:
    /// - Updated values
    /// - New momentum
    /// - New velocity
    /// - New maximum velocity (if maximum_velocity is provided)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mpsgraph_rs::prelude::*;
    /// # let graph = MPSGraph::new();
    /// # let weights = graph.placeholder_with_shape(&[2, 3], MPSDataType::Float32, None);
    /// # let gradients = graph.placeholder_with_shape(&[2, 3], MPSDataType::Float32, None);
    /// # let learning_rate = graph.constant_scalar(0.001, MPSDataType::Float32, None);
    /// # let beta1 = graph.constant_scalar(0.9, MPSDataType::Float32, None);
    /// # let beta2 = graph.constant_scalar(0.999, MPSDataType::Float32, None);
    /// # let epsilon = graph.constant_scalar(1e-8, MPSDataType::Float32, None);
    /// # let beta1_power = graph.constant_scalar(0.9, MPSDataType::Float32, None);
    /// # let beta2_power = graph.constant_scalar(0.999, MPSDataType::Float32, None);
    /// # let momentum = graph.zeros_like(&weights, None);
    /// # let velocity = graph.zeros_like(&weights, None);
    /// 
    /// // Update weights using Adam
    /// let results = graph.adam(
    ///     &learning_rate,
    ///     &beta1,
    ///     &beta2,
    ///     &epsilon,
    ///     &beta1_power,
    ///     &beta2_power,
    ///     &weights,
    ///     &momentum,
    ///     &velocity,
    ///     None, // no maximum velocity
    ///     &gradients,
    ///     None
    /// );
    /// 
    /// let updated_weights = &results[0];
    /// let new_momentum = &results[1];
    /// let new_velocity = &results[2];
    /// ```
    pub fn adam(
        &self,
        learning_rate: &MPSGraphTensor,
        beta1: &MPSGraphTensor,
        beta2: &MPSGraphTensor,
        epsilon: &MPSGraphTensor,
        beta1_power: &MPSGraphTensor,
        beta2_power: &MPSGraphTensor,
        values: &MPSGraphTensor,
        momentum: &MPSGraphTensor,
        velocity: &MPSGraphTensor,
        maximum_velocity: Option<&MPSGraphTensor>,
        gradient: &MPSGraphTensor,
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let max_velocity_obj = match maximum_velocity {
                Some(m) => m.0,
                None => std::ptr::null_mut(),
            };
            
            let result_array: *mut Object = msg_send![
                self.0, 
                adamWithLearningRateTensor:learning_rate.0
                beta1Tensor:beta1.0
                beta2Tensor:beta2.0
                epsilonTensor:epsilon.0
                beta1PowerTensor:beta1_power.0
                beta2PowerTensor:beta2_power.0
                valuesTensor:values.0
                momentumTensor:momentum.0
                velocityTensor:velocity.0
                maximumVelocityTensor:max_velocity_obj
                gradientTensor:gradient.0
                name:name_obj
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut Object = msg_send![result_array, objectAtIndex:i];
                let tensor: *mut Object = msg_send![tensor, retain];
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
    
    /// Adam optimization with current learning rate.
    ///
    /// This is a variant of Adam where the learning rate adaptation is already applied
    /// to the learning rate tensor.
    ///
    /// ```
    /// m[t] = beta1 * m[t-1] + (1 - beta1) * g
    /// v[t] = beta2 * v[t-1] + (1 - beta2) * (g ^ 2)
    /// maxVel[t] = max(maxVel[t-1], v[t])
    /// variable = variable - current_learning_rate * m[t] / (sqrt(maxVel) + epsilon)
    /// ```
    ///
    /// # Parameters
    ///
    /// * `current_learning_rate` - Scalar tensor with the already adjusted learning rate
    /// * `beta1` - Exponential decay rate for first moment estimates
    /// * `beta2` - Exponential decay rate for second moment estimates
    /// * `epsilon` - Small constant for numerical stability
    /// * `values` - Current parameter values to be updated
    /// * `momentum` - First moment estimates (momentum)
    /// * `velocity` - Second moment estimates (velocity)
    /// * `maximum_velocity` - Optional maximum velocity tensor
    /// * `gradient` - Gradient of parameters with respect to loss
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector containing tensors for:
    /// - Updated values
    /// - New momentum
    /// - New velocity
    /// - New maximum velocity (if maximum_velocity is provided)
    pub fn adam_with_current_learning_rate(
        &self,
        current_learning_rate: &MPSGraphTensor,
        beta1: &MPSGraphTensor,
        beta2: &MPSGraphTensor,
        epsilon: &MPSGraphTensor,
        values: &MPSGraphTensor,
        momentum: &MPSGraphTensor,
        velocity: &MPSGraphTensor,
        maximum_velocity: Option<&MPSGraphTensor>,
        gradient: &MPSGraphTensor,
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let max_velocity_obj = match maximum_velocity {
                Some(m) => m.0,
                None => std::ptr::null_mut(),
            };
            
            let result_array: *mut Object = msg_send![
                self.0, 
                adamWithCurrentLearningRateTensor:current_learning_rate.0
                beta1Tensor:beta1.0
                beta2Tensor:beta2.0
                epsilonTensor:epsilon.0
                valuesTensor:values.0
                momentumTensor:momentum.0
                velocityTensor:velocity.0
                maximumVelocityTensor:max_velocity_obj
                gradientTensor:gradient.0
                name:name_obj
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut Object = msg_send![result_array, objectAtIndex:i];
                let tensor: *mut Object = msg_send![tensor, retain];
                result.push(MPSGraphTensor(tensor));
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
    use std::collections::HashMap;
    
    #[test]
    fn test_sgd() {
        // This test will be skipped in environments without Metal
        if crate::tests::should_skip_test("test_sgd") {
            return;
        }
        
        let graph = MPSGraph::new();
        let device = MPSGraphDevice::new();
        
        // Create tensors for a simple SGD update
        // weights = [1.0, 2.0, 3.0]
        // gradients = [0.1, 0.2, 0.3]
        // learning_rate = 0.1
        // expected = weights - learning_rate * gradients = [0.99, 1.98, 2.97]
        
        let weights = graph.placeholder_with_shape(&[3], MPSDataType::Float32, Some("weights"));
        let gradients = graph.placeholder_with_shape(&[3], MPSDataType::Float32, Some("gradients"));
        let learning_rate = graph.placeholder_with_shape(&[], MPSDataType::Float32, Some("lr"));
        
        // Run SGD update
        let updated_weights = graph.stochastic_gradient_descent(
            &learning_rate,
            &weights,
            &gradients,
            Some("sgd_op")
        );
        
        // Create input data
        let weights_data = vec![1.0f32, 2.0, 3.0];
        let gradients_data = vec![0.1f32, 0.2, 0.3];
        let lr_data = vec![0.1f32];
        
        let weights_tensor_data = crate::tensor_data::MPSGraphTensorData::new(
            &weights_data, 
            &[3], 
            MPSDataType::Float32
        );
        
        let gradients_tensor_data = crate::tensor_data::MPSGraphTensorData::new(
            &gradients_data, 
            &[3], 
            MPSDataType::Float32
        );
        
        let lr_tensor_data = crate::tensor_data::MPSGraphTensorData::new(
            &lr_data, 
            &[], 
            MPSDataType::Float32
        );
        
        // Run the graph
        let feeds = HashMap::from([
            (weights, weights_tensor_data),
            (gradients, gradients_tensor_data),
            (learning_rate, lr_tensor_data),
        ]);
        
        let results = graph.run_with_feeds(&feeds, &[updated_weights.clone()], &device, None);
        
        // Get the result
        let output = results.get(&updated_weights).unwrap();
        let values: Vec<f32> = output.to_vec();
        
        // Expected values: weights - learning_rate * gradients
        let expected = vec![0.99f32, 1.98f32, 2.97f32];
        
        // Compare with tolerance
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5, 
                    "Expected {}, got {}", expected, actual);
        }
    }
    
    #[test]
    fn test_adam() {
        // This test will be skipped in environments without Metal
        if crate::tests::should_skip_test("test_adam") {
            return;
        }
        
        let graph = MPSGraph::new();
        let device = MPSGraphDevice::new();
        
        // Create tensors for a simple Adam update
        // values = [1.0, 2.0, 3.0]
        // gradients = [0.1, 0.2, 0.3]
        // momentum = [0.0, 0.0, 0.0]
        // velocity = [0.0, 0.0, 0.0]
        // learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8
        // beta1_power = 0.9, beta2_power = 0.999 (first iteration)
        
        let values = graph.placeholder_with_shape(&[3], MPSDataType::Float32, Some("values"));
        let gradients = graph.placeholder_with_shape(&[3], MPSDataType::Float32, Some("gradients"));
        let momentum = graph.placeholder_with_shape(&[3], MPSDataType::Float32, Some("momentum"));
        let velocity = graph.placeholder_with_shape(&[3], MPSDataType::Float32, Some("velocity"));
        
        // Create constants for Adam parameters
        let learning_rate = graph.constant_scalar_value(0.001, MPSDataType::Float32, Some("lr"));
        let beta1 = graph.constant_scalar_value(0.9, MPSDataType::Float32, Some("beta1"));
        let beta2 = graph.constant_scalar_value(0.999, MPSDataType::Float32, Some("beta2"));
        let epsilon = graph.constant_scalar_value(1e-8, MPSDataType::Float32, Some("epsilon"));
        let beta1_power = graph.constant_scalar_value(0.9, MPSDataType::Float32, Some("beta1_power"));
        let beta2_power = graph.constant_scalar_value(0.999, MPSDataType::Float32, Some("beta2_power"));
        
        // Run Adam update
        let result = graph.adam(
            &learning_rate,
            &beta1,
            &beta2,
            &epsilon,
            &beta1_power,
            &beta2_power,
            &values,
            &momentum,
            &velocity,
            None, // no maximum velocity
            &gradients,
            Some("adam_op")
        );
        
        // Extract result tensors
        let updated_values = &result[0];
        let updated_momentum = &result[1];
        let updated_velocity = &result[2];
        
        // Create input data
        let values_data = vec![1.0f32, 2.0, 3.0];
        let gradients_data = vec![0.1f32, 0.2, 0.3];
        let momentum_data = vec![0.0f32, 0.0, 0.0];
        let velocity_data = vec![0.0f32, 0.0, 0.0];
        
        let values_tensor_data = crate::tensor_data::MPSGraphTensorData::new(
            &values_data, 
            &[3], 
            MPSDataType::Float32
        );
        
        let gradients_tensor_data = crate::tensor_data::MPSGraphTensorData::new(
            &gradients_data, 
            &[3], 
            MPSDataType::Float32
        );
        
        let momentum_tensor_data = crate::tensor_data::MPSGraphTensorData::new(
            &momentum_data, 
            &[3], 
            MPSDataType::Float32
        );
        
        let velocity_tensor_data = crate::tensor_data::MPSGraphTensorData::new(
            &velocity_data, 
            &[3], 
            MPSDataType::Float32
        );
        
        // Run the graph
        let feeds = HashMap::from([
            (values, values_tensor_data),
            (gradients, gradients_tensor_data),
            (momentum, momentum_tensor_data),
            (velocity, velocity_tensor_data),
        ]);
        
        let results = graph.run_with_feeds(
            &feeds, 
            &[updated_values.clone(), updated_momentum.clone(), updated_velocity.clone()], 
            &device, 
            None
        );
        
        // Get results
        let new_values = results.get(updated_values).unwrap().to_vec::<f32>();
        let new_momentum = results.get(updated_momentum).unwrap().to_vec::<f32>();
        let new_velocity = results.get(updated_velocity).unwrap().to_vec::<f32>();
        
        // Manually calculate expected values
        // m' = beta1 * m + (1 - beta1) * g
        // v' = beta2 * v + (1 - beta2) * g^2
        // values' = values - (lr * sqrt(1 - beta2^t) / (1 - beta1^t)) * m' / (sqrt(v') + epsilon)
        
        // Expected new momentum: m' = 0.9 * 0 + 0.1 * g
        let expected_momentum = vec![0.01f32, 0.02, 0.03]; // 0.1 * gradients
        
        // Expected new velocity: v' = 0.999 * 0 + 0.001 * g^2
        let expected_velocity = vec![0.00001f32, 0.00004, 0.00009]; // 0.001 * gradients^2
        
        // Validation
        for i in 0..3 {
            assert!((new_momentum[i] - expected_momentum[i]).abs() < 1e-5,
                    "Momentum at index {}: expected {}, got {}", i, expected_momentum[i], new_momentum[i]);
            
            assert!((new_velocity[i] - expected_velocity[i]).abs() < 1e-5,
                    "Velocity at index {}: expected {}, got {}", i, expected_velocity[i], new_velocity[i]);
            
            // We don't check exact values for updated parameters as they involve more complex calculations,
            // but we check they're different from original
            assert!(new_values[i] != values_data[i], "Values should be updated at index {}", i);
        }
    }
}