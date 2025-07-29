use crate::{Graph, ScalarOrTensor, Tensor};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

/// GatherOps.
impl Graph {
    /// General *gather* with batch support.
    ///
    /// # Arguments
    ///
    /// * `updates_tensor` – Tensor supplying the slices.
    /// * `indices_tensor` – Indices that pick slices out of
    ///   `updates_tensor`.
    /// * `axis` – Dimension along which to gather.
    /// * `batch_dimensions` – Number of leading batch dims shared by all
    ///   tensors.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing the gathered slices.
    pub fn gather_with_updates(
        &self,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        axis: i64,
        batch_dimensions: u64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                gatherWithUpdatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                axis: axis,
                batchDimensions: batch_dimensions,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// *Gather-along-axis* variant.
    ///
    /// Gathers values from `updates_tensor` at `indices_tensor` along `axis`.
    /// The resulting tensor has the same shape as `indices_tensor`.
    /// Out-of-bounds indices produce zeros.
    ///
    /// # Arguments
    ///
    /// * `axis` – Axis to gather from (scalar or tensor; negative wraps).
    /// * `updates_tensor` – Source tensor.
    /// * `indices_tensor` – Int32/Int64 indices tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing gathered values.
    pub fn gather_along_axis<'a>(
        &self,
        axis: ScalarOrTensor<'a, i64>,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                    self,
                    gatherAlongAxis: axis,
                    withUpdatesTensor: updates_tensor,
                    indicesTensor: indices_tensor,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    gatherAlongAxisTensor: axis,
                    withUpdatesTensor: updates_tensor,
                    indicesTensor: indices_tensor,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
