use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

use crate::{Graph, ScalarOrTensor, Tensor};

impl Graph {
    /// Applies the ReLU activation: `f(x) = max(x, 0)`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing the ReLU results.
    pub fn relu(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![self, reLUWithTensor: tensor, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Gradient of the ReLU activation.
    ///
    /// # Arguments
    ///
    /// * `gradient` – Incoming gradient (`dL/dR`).
    /// * `source` – Tensor used in the forward ReLU pass.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing `dL/dX`.
    pub fn relu_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                reLUGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the sigmoid activation.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] after sigmoid.
    pub fn sigmoid(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![self, sigmoidWithTensor: tensor, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Gradient of the sigmoid activation.
    ///
    /// # Arguments
    ///
    /// * `gradient` – Incoming gradient.
    /// * `source` – Tensor used in the forward sigmoid pass.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing `dL/dX`.
    pub fn sigmoid_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                sigmoidGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the softmax function along `axis`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `axis` – Axis along which softmax is computed.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing probabilities that sum to 1 across `axis`.
    pub fn soft_max(&self, tensor: &Tensor, axis: i64, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![self, softMaxWithTensor: tensor, axis: axis, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Gradient of the softmax function.
    ///
    /// # Arguments
    ///
    /// * `gradient` – Incoming gradient.
    /// * `source` – Tensor used in the forward softmax pass.
    /// * `axis` – Axis along which softmax was computed.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing `dL/dX`.
    pub fn soft_max_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                softMaxGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                axis: axis,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies leaky ReLU: `f(x) = max(x, α x)`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `alpha` – Slope for negative values (scalar or tensor).
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] after leaky ReLU.
    pub fn leaky_relu<'a>(
        &self,
        tensor: &Tensor,
        alpha: ScalarOrTensor<'a, f64>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match alpha {
                ScalarOrTensor::Scalar(alpha) => {
                    msg_send![self, leakyReLUWithTensor: tensor, alpha: alpha, name: name.map(NSString::from_str).as_deref()]
                }
                ScalarOrTensor::Tensor(alpha_tensor) => {
                    msg_send![self, leakyReLUWithTensor: tensor, alphaTensor: alpha_tensor, name: name.map(NSString::from_str).as_deref()]
                }
            }
        }
    }

    /// Gradient of the leaky ReLU activation.
    ///
    /// Supports broadcasting of `alpha_tensor`.
    ///
    /// # Arguments
    ///
    /// * `gradient` – Incoming gradient.
    /// * `source` – Input tensor from forward pass.
    /// * `alpha_tensor` – Alpha tensor used in the forward pass.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing `dL/dX`.
    pub fn leaky_relu_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        alpha_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                leakyReLUGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                alphaTensor: alpha_tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }
}
