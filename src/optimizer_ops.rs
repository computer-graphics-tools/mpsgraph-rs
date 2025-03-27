use objc2::runtime::AnyObject;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, AsRawObject};

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
        learning_rate:  &MPSGraphTensor,
        values:  &MPSGraphTensor,
        gradient:  &MPSGraphTensor,
        name:  Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![
                self.0, stochasticGradientDescentWithLearningRateTensor: learning_rate.0,
                valuesTensor: values.0,
                gradientTensor: gradient.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
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
        learning_rate:  &MPSGraphTensor,
        beta1:  &MPSGraphTensor,
        beta2:  &MPSGraphTensor,
        epsilon:  &MPSGraphTensor,
        beta1_power:  &MPSGraphTensor,
        beta2_power:  &MPSGraphTensor,
        values:  &MPSGraphTensor,
        momentum:  &MPSGraphTensor,
        velocity:  &MPSGraphTensor,
        maximum_velocity:  Option<&MPSGraphTensor>,
        gradient:  &MPSGraphTensor,
        name:  Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let max_velocity_obj = match maximum_velocity {
                Some(m) => m.0,
                None => std::ptr::null_mut(),
            };
            
            let result_array: *mut AnyObject = msg_send![
                self.0, adamWithLearningRateTensor: learning_rate.0,
                beta1Tensor: beta1.0,
                beta2Tensor: beta2.0,
                epsilonTensor: epsilon.0,
                beta1PowerTensor: beta1_power.0,
                beta2PowerTensor: beta2_power.0,
                valuesTensor: values.0,
                momentumTensor: momentum.0,
                velocityTensor: velocity.0,
                maximumVelocityTensor: max_velocity_obj,
                gradientTensor: gradient.0,
                name: name_obj,
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![result_array, objectAtIndex: i];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
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
        current_learning_rate:  &MPSGraphTensor,
        beta1:  &MPSGraphTensor,
        beta2:  &MPSGraphTensor,
        epsilon:  &MPSGraphTensor,
        values:  &MPSGraphTensor,
        momentum:  &MPSGraphTensor,
        velocity:  &MPSGraphTensor,
        maximum_velocity:  Option<&MPSGraphTensor>,
        gradient:  &MPSGraphTensor,
        name:  Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let max_velocity_obj = match maximum_velocity {
                Some(m) => m.0,
                None => std::ptr::null_mut(),
            };
            
            let result_array: *mut AnyObject = msg_send![
                self.0, adamWithCurrentLearningRateTensor: current_learning_rate.0,
                beta1Tensor: beta1.0,
                beta2Tensor: beta2.0,
                epsilonTensor: epsilon.0,
                valuesTensor: values.0,
                momentumTensor: momentum.0,
                velocityTensor: velocity.0,
                maximumVelocityTensor: max_velocity_obj,
                gradientTensor: gradient.0,
                name: name_obj,
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![result_array, objectAtIndex: i];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
}