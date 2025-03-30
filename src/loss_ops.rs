use crate::core::{AsRawObject, NSString};
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use objc2::runtime::AnyObject;

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
    /// ```text
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
    /// # use mpsgraph::prelude::*;
    /// # use mpsgraph::loss_ops::MPSGraphLossReductionType;
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
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, softMaxCrossEntropyWithSourceTensor: source_tensor.0,
                labelsTensor: labels_tensor.0,
                axis: axis,
                reductionType: reduction_type as u64,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
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
    /// # use mpsgraph::prelude::*;
    /// # use mpsgraph::loss_ops::MPSGraphLossReductionType;
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
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, softMaxCrossEntropyGradientWithIncomingGradientTensor: gradient_tensor.0,
                sourceTensor: source_tensor.0,
                labelsTensor: labels_tensor.0,
                axis: axis,
                reductionType: reduction_type as u64,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}
