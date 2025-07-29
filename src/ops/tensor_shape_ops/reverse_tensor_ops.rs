use super::AxesOrTensor;
use crate::{Graph, Tensor, ns_number_array_from_slice};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Reverses `tensor` along the provided `axes`.
    ///
    /// Semantics match TensorFlow’s *reverse* op.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Tensor to reverse.
    /// * `axes` – Axes to flip (slice or tensor; must be unique and within
    ///   range).
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] with the specified axes reversed.
    pub fn reverse_with_axes<'a>(
        &self,
        tensor: &Tensor,
        axes: AxesOrTensor<'a>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axes {
            AxesOrTensor::Axes(axes) => unsafe {
                msg_send![
                    self,
                    reverseTensor: tensor,
                    axes: &*ns_number_array_from_slice(axes),
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            AxesOrTensor::Tensor(tensor) => unsafe {
                msg_send![
                    self,
                    reverseTensor: tensor,
                    axesTensor: tensor,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Reverses `tensor` along *all* axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Tensor to reverse.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A fully reversed [`Tensor`].
    pub fn reverse(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                reverseTensor: tensor,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }
}
