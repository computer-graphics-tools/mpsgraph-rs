use super::StartEndStrideScalarsOrTensors;
use crate::{Graph, Tensor, ns_number_array_from_slice};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Creates a strided-slice gradient operation.
    ///
    /// # Arguments
    ///
    /// * `input_gradient_tensor` – Gradient flowing into the slice op
    ///   (`dL/dR`).
    /// * `fwd_in_shape_tensor` – Shape of the forward-pass input (defines the
    ///   output shape of this gradient op).
    /// * `starts` – Per-dimension start indices.
    /// * `ends` – Per-dimension end indices.
    /// * `strides` – Per-dimension strides.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing `dL/dS` where `S` was the sliced tensor.
    pub fn slice_gradient(
        &self,
        input_gradient_tensor: &Tensor,
        fwd_in_shape_tensor: &Tensor,
        starts: &[u64],
        ends: &[u64],
        strides: &[u64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                sliceGradientTensor: input_gradient_tensor,
                fwdInShapeTensor: fwd_in_shape_tensor,
                starts: &*ns_number_array_from_slice(starts),
                ends: &*ns_number_array_from_slice(ends),
                strides: &*ns_number_array_from_slice(strides),
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Creates a strided-slice gradient operation with *mask* support.
    ///
    /// # Arguments
    ///
    /// * `input_gradient_tensor` – Gradient flowing into the slice op.
    /// * `fwd_in_shape_tensor` – Shape of the forward-pass input.
    /// * `start_end_stride` – Slice parameters provided either as scalars or
    ///   tensors (see [`StartEndStrideScalarsOrTensors`]).
    /// * `start_mask` – Bitmask of dimensions whose `starts` values are
    ///   ignored.
    /// * `end_mask` – Bitmask of dimensions whose `ends` values are ignored.
    /// * `squeeze_mask` – Bitmask of dimensions to squeeze out of the result.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing the gradient result.
    pub fn slice_gradient_with_masks<'a>(
        &self,
        input_gradient_tensor: &Tensor,
        fwd_in_shape_tensor: &Tensor,
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
                    sliceGradientTensor: input_gradient_tensor,
                    fwdInShapeTensor: fwd_in_shape_tensor,
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
                    sliceGradientTensor: input_gradient_tensor,
                    fwdInShapeTensor: fwd_in_shape_tensor,
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

    /// Creates a slice gradient operation that uses *start* and *size* tensors.
    ///
    /// # Arguments
    ///
    /// * `input_gradient_tensor` – Gradient flowing into the slice op.
    /// * `fwd_in_shape_tensor` – Shape of the forward-pass input.
    /// * `start_tensor` – Tensor defining per-dimension start indices.
    /// * `size_tensor` – Tensor defining the size of the forward result per
    ///   dimension.
    /// * `squeeze_mask` – Bitmask of dimensions to squeeze from the result.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing the gradient result.
    pub fn slice_gradient_start_tensor_size_tensor_squeeze_mask(
        &self,
        input_gradient_tensor: &Tensor,
        fwd_in_shape_tensor: &Tensor,
        start_tensor: &Tensor,
        size_tensor: &Tensor,
        squeeze_mask: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                sliceGradientTensor: input_gradient_tensor,
                fwdInShapeTensor: fwd_in_shape_tensor,
                startTensor: start_tensor,
                sizeTensor: size_tensor,
                squeezeMask: squeeze_mask,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }
}
