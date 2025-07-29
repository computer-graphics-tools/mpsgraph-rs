use crate::{Graph, ScalarOrTensor, Tensor};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Flattens a tensor to rank-2 by collapsing dimensions around `axis`.
    ///
    /// All dimensions before `axis` are multiplied into the first output
    /// dimension, and the remaining ones into the second.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Tensor to flatten.
    /// * `axis` – Split point between the two flattened groups (scalar or
    ///   tensor; negative indices wrap).
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A rank-2 [`Tensor`] with shape `[∏ dims<axis, ∏ dims≥axis]`.
    pub fn flatten_2d<'a>(
        &self,
        tensor: &Tensor,
        axis: ScalarOrTensor<'a, i64>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                    self,
                    flatten2DTensor: tensor,
                    axis: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    flatten2DTensor: tensor,
                    axisTensor: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
