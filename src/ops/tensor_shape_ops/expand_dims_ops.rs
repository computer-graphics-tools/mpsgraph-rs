use super::AxesOrTensor;
use crate::{Graph, Tensor, ns_number_array_from_slice};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Inserts a singleton dimension at `axis`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor to reshape.
    /// * `axis` – Axis index at which to insert a dimension of size 1 (negative
    ///   indices wrap around).
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A reshaped [`Tensor`] with one additional dimension.
    pub fn expand_dims(&self, tensor: &Tensor, axis: i64, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![self, expandDimsOfTensor: tensor, axis: axis, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Inserts singleton dimensions at multiple `axes`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `axes` – Axes at which to insert size-1 dimensions (slice or tensor).
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A reshaped [`Tensor`] with additional dimensions.
    pub fn expand_dims_axes<'a>(
        &self,
        tensor: &Tensor,
        axes: AxesOrTensor<'a>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axes {
            AxesOrTensor::Axes(axes) => unsafe {
                msg_send![self, expandDimsOfTensor: tensor, axes: &*ns_number_array_from_slice(axes), name: name.map(NSString::from_str).as_deref()]
            },
            AxesOrTensor::Tensor(axes) => unsafe {
                msg_send![self, expandDimsOfTensor: tensor, axesTensor: axes, name: name.map(NSString::from_str).as_deref()]
            },
        }
    }
}
