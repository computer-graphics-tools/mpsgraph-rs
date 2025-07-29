use super::StartEndStrideScalarsOrTensors;
use crate::{Graph, Tensor, ns_number_array_from_slice};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Creates a strided-slice update operation.
    ///
    /// Replaces a slice of `data_tensor` with `update_tensor` using *start*,
    /// *end*, and *stride* parameters. All masks are implicitly set to zero.
    ///
    /// # Arguments
    ///
    /// * `data_tensor` – Tensor to be updated.
    /// * `update_tensor` – Tensor providing replacement values.
    /// * `start_end_stride` – Slice parameters provided as scalars or tensors
    ///   (see [`StartEndStrideScalarsOrTensors`]).
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] representing the updated data.
    pub fn slice_update<'a>(
        &self,
        data_tensor: &Tensor,
        update_tensor: &Tensor,
        start_end_stride: StartEndStrideScalarsOrTensors<'a>,
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
                    sliceUpdateDataTensor: data_tensor,
                    updateTensor: update_tensor,
                    starts: &*ns_number_array_from_slice(starts),
                    ends: &*ns_number_array_from_slice(ends),
                    strides: &*ns_number_array_from_slice(strides),
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
                    sliceUpdateDataTensor: data_tensor,
                    updateTensor: update_tensor,
                    startsTensor: start_tensor,
                    endsTensor: end_tensor,
                    stridesTensor: stride_tensor,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Creates a strided-slice update operation with explicit *mask* control.
    ///
    /// # Arguments
    ///
    /// * `data_tensor` – Tensor to be updated.
    /// * `update_tensor` – Tensor providing replacement values.
    /// * `start_end_stride` – Slice parameters provided as scalars or tensors.
    /// * `start_mask` – Bitmask of dimensions whose `starts` values are ignored.
    /// * `end_mask` – Bitmask of dimensions whose `ends` values are ignored.
    /// * `squeeze_mask` – Bitmask of dimensions to remove from the result.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] representing the updated data.
    pub fn slice_update_with_masks<'a>(
        &self,
        data_tensor: &Tensor,
        update_tensor: &Tensor,
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
                    sliceUpdateDataTensor: data_tensor,
                    updateTensor: update_tensor,
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
                    sliceUpdateDataTensor: data_tensor,
                    updateTensor: update_tensor,
                    startsTensor: start_tensor,
                    endsTensor: end_tensor,
                    stridesTensor: stride_tensor,
                    startMask: start_mask,
                    endMask: end_mask,
                    squeezeMask: squeeze_mask,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
