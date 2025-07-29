use crate::{Graph, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Creates a one-dimensional slice along a given axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Source tensor to slice.
    /// * `dimension_index` – Axis along which to slice.
    /// * `start` – Starting index (negative values count from the end of the
    ///   dimension).
    /// * `length` – Number of elements to keep.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] representing the slice.
    pub fn slice(
        &self,
        tensor: &Tensor,
        dimension_index: u64,
        start: i64,
        length: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                sliceTensor: tensor,
                dimension: dimension_index,
                start: start,
                length: length,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Creates an N-dimensional slice using *start* and *size* tensors.
    ///
    /// The operation starts at `start_tensor` and keeps `size_tensor` elements
    /// along each dimension, following the semantics of TensorFlow’s *Slice*
    /// op.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Tensor to slice.
    /// * `start_tensor` – Per-dimension starting indices.
    /// * `size_tensor` – Per-dimension slice lengths.
    /// * `squeeze_mask` – Bitmask of dimensions to remove from the result.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing the slice.
    pub fn slice_with_tensors(
        &self,
        tensor: &Tensor,
        start_tensor: &Tensor,
        size_tensor: &Tensor,
        squeeze_mask: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                sliceTensor: tensor,
                startTensor: start_tensor,
                sizeTensor: size_tensor,
                squeezeMask: squeeze_mask,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }
}
