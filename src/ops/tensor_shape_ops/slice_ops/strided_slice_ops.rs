use super::StartEndStrideScalarsOrTensors;
use crate::{Graph, Tensor, ns_number_array_from_slice};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Creates a strided-slice operation.
    ///
    /// Equivalent to TensorFlow’s *StridedSlice* op.
    /// The slice starts at `starts`, ends before `ends`, and moves `strides`
    /// steps at a time.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Tensor to slice.
    /// * `starts` – Per-dimension start indices.
    /// * `ends` – Per-dimension end indices.
    /// * `strides` – Per-dimension strides.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] representing the slice.
    pub fn strided_slice(
        &self,
        tensor: &Tensor,
        starts: &[u64],
        ends: &[u64],
        strides: &[u64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                sliceTensor: tensor,
                starts: &*ns_number_array_from_slice(starts),
                ends: &*ns_number_array_from_slice(ends),
                strides: &*ns_number_array_from_slice(strides),
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Creates a strided-slice operation with explicit *mask* control.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Tensor to slice.
    /// * `start_end_stride` – Slice parameters as scalars or tensors (see
    ///   [`StartEndStrideScalarsOrTensors`]).
    /// * `start_mask` – Bitmask of dimensions whose `starts` values are
    ///   ignored.
    /// * `end_mask` – Bitmask of dimensions whose `ends` values are ignored.
    /// * `squeeze_mask` – Bitmask of dimensions to remove from the result.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] representing the slice.
    pub fn strided_slice_with_masks<'a>(
        &self,
        tensor: &Tensor,
        start_end_stride: StartEndStrideScalarsOrTensors<'a>,
        start_mask: u32,
        end_mask: u32,
        squeeze_mask: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match start_end_stride {
            StartEndStrideScalarsOrTensors::Scalars {
                starts,
                ends,
                strides,
            } => unsafe {
                msg_send![
                self,
                sliceTensor: tensor,
                starts: &*ns_number_array_from_slice(starts),
                ends: &*ns_number_array_from_slice(ends),
                strides: &*ns_number_array_from_slice(strides),
                startMask: start_mask,
                endMask: end_mask,
                squeezeMask: squeeze_mask,
                name: name.map(NSString::from_str).as_deref(),
                    ]
            },
            StartEndStrideScalarsOrTensors::Tensors {
                start_tensor,
                end_tensor,
                stride_tensor,
            } => unsafe {
                msg_send![
                    self,
                    sliceTensor: tensor,
                    startTensor: start_tensor,
                    endTensor: end_tensor,
                    strideTensor: stride_tensor,
                    startMask: start_mask,
                    endMask: end_mask,
                    squeezeMask: squeeze_mask,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
