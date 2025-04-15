use crate::core::{AsRawObject, NSString};
use crate::graph::Graph;
use crate::operation::Operation;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use std::ptr;

/// Represents a variable operation in MPSGraph.
///
/// In MPSGraph, a variable operation is a special operation that creates a tensor
/// which can be updated during the graph execution. This is particularly useful for
/// trainable parameters in machine learning models.
pub struct VariableOp(pub(crate) *mut AnyObject);

impl VariableOp {
    /// Returns the operation associated with this variable.
    pub fn operation(&self) -> Operation {
        unsafe {
            let op: *mut AnyObject = msg_send![self.0, operation];
            let op = objc2::ffi::objc_retain(op as *mut _);
            Operation(op)
        }
    }

    /// Returns the tensor associated with this variable.
    pub fn tensor(&self) -> Tensor {
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, tensor];
            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }
}

impl Drop for VariableOp {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for VariableOp {
    fn clone(&self) -> Self {
        unsafe {
            let obj = objc2::ffi::objc_retain(self.0 as *mut _);
            VariableOp(obj)
        }
    }
}

/// Optimizer operations for Graph
impl Graph {
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
    /// # use mpsgraph::prelude::*;
    /// # let graph = Graph::new();
    /// # let shape = Shape::from_slice(&[2, 3]);
    /// # let weights = graph.placeholder(&shape, MPSDataType::Float32, None);
    /// # let gradients = graph.placeholder(&shape, MPSDataType::Float32, None);
    /// # let learning_rate = graph.constant_scalar(0.01, MPSDataType::Float32);
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
        learning_rate: &Tensor,
        values: &Tensor,
        gradient: &Tensor,
        name: Option<&str>,
    ) -> Tensor {
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

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

    /// Adam optimization.
    ///
    /// The Adam optimizer combines ideas from momentum and RMSProp, maintaining per-parameter
    /// momentum and velocity (squared gradients) to adaptively adjust learning rates.
    ///
    /// ```text
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
    /// # use mpsgraph::prelude::*;
    /// # let graph = Graph::new();
    /// # let shape = Shape::from_slice(&[2, 3]);
    /// # let weights = graph.placeholder(&shape, MPSDataType::Float32, None);
    /// # let gradients = graph.placeholder(&shape, MPSDataType::Float32, None);
    /// # let learning_rate = graph.constant_scalar(0.001, MPSDataType::Float32);
    /// # let beta1 = graph.constant_scalar(0.9, MPSDataType::Float32);
    /// # let beta2 = graph.constant_scalar(0.999, MPSDataType::Float32);
    /// # let epsilon = graph.constant_scalar(1e-8, MPSDataType::Float32);
    /// # let beta1_power = graph.constant_scalar(0.9, MPSDataType::Float32);
    /// # let beta2_power = graph.constant_scalar(0.999, MPSDataType::Float32);
    /// # let dims = shape.dimensions();
    /// # let zeros = vec![0.0f32; dims.iter().product()];
    /// # let momentum = graph.constant_with_shape(&zeros, &dims, MPSDataType::Float32);
    /// # let velocity = graph.constant_with_shape(&zeros, &dims, MPSDataType::Float32);
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
        learning_rate: &Tensor,
        beta1: &Tensor,
        beta2: &Tensor,
        epsilon: &Tensor,
        beta1_power: &Tensor,
        beta2_power: &Tensor,
        values: &Tensor,
        momentum: &Tensor,
        velocity: &Tensor,
        maximum_velocity: Option<&Tensor>,
        gradient: &Tensor,
        name: Option<&str>,
    ) -> Vec<Tensor> {
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

            // Convert NSArray to Vec<Tensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![result_array, objectAtIndex: i];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _);
                result.push(Tensor(tensor));
            }

            result
        }
    }

    /// Adam optimization with current learning rate.
    ///
    /// This is a variant of Adam where the learning rate adaptation is already applied
    /// to the learning rate tensor.
    ///
    /// ```text
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
        current_learning_rate: &Tensor,
        beta1: &Tensor,
        beta2: &Tensor,
        epsilon: &Tensor,
        values: &Tensor,
        momentum: &Tensor,
        velocity: &Tensor,
        maximum_velocity: Option<&Tensor>,
        gradient: &Tensor,
        name: Option<&str>,
    ) -> Vec<Tensor> {
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

            // Convert NSArray to Vec<Tensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![result_array, objectAtIndex: i];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _);
                result.push(Tensor(tensor));
            }

            result
        }
    }

    /// Creates a variable operation for a tensor.
    ///
    /// This method creates a variable operation from a tensor, which can then be used
    /// with optimization operations that require direct manipulation of the variable.
    ///
    /// # Parameters
    ///
    /// * `tensor` - The tensor to convert to a variable
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new VariableOp wrapper
    pub fn variable_op_for_tensor(
        &self,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> VariableOp {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let variable_op: *mut AnyObject = msg_send![
                self.0, variableOpWithTensor: tensor.0,
                name: name_obj,
            ];

            let variable_op = objc2::ffi::objc_retain(variable_op as *mut _);
            VariableOp(variable_op)
        }
    }

    /// Applies stochastic gradient descent directly to a variable operation.
    ///
    /// This method updates the variable operation in-place using stochastic gradient descent.
    ///
    /// # Parameters
    ///
    /// * `learning_rate` - Scalar tensor which indicates the learning rate to use
    /// * `variable_op` - The variable operation to update
    /// * `gradient` - Partial gradient of the trainable parameters with respect to loss
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tensor representing the updated value of the variable
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mpsgraph::prelude::*;
    /// # let graph = Graph::new();
    /// # let shape = Shape::from_slice(&[2, 3]);
    /// # let weights = graph.placeholder(&shape, MPSDataType::Float32, None);
    /// # let gradients = graph.placeholder(&shape, MPSDataType::Float32, None);
    /// # let learning_rate = graph.constant_scalar(0.01, MPSDataType::Float32);
    ///
    /// // Create a variable operation for the weights
    /// let weights_var = graph.variable_op_for_tensor(&weights, Some("weights_var"));
    ///
    /// // Update the variable in-place using SGD
    /// let updated_weights = graph.apply_stochastic_gradient_descent(
    ///     &learning_rate,
    ///     &weights_var,
    ///     &gradients,
    ///     None
    /// );
    /// ```
    pub fn apply_stochastic_gradient_descent(
        &self,
        learning_rate: &Tensor,
        variable_op: &VariableOp,
        gradient: &Tensor,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![
                self.0, applyStochasticGradientDescentWithLearningRateTensor: learning_rate.0,
                variableOp: variable_op.0,
                gradientTensor: gradient.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }
}
