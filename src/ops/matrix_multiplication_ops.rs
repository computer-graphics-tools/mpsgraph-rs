use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

use crate::{DataType, Graph, Tensor};

/// MPSGraphMatrixMultiplicationOps.
impl Graph {
    /// Computes the matrix multiplication of 2 input tensors with support for broadcasting.
    ///
    /// # Arguments
    ///
    /// * `primary_tensor` - The left-hand side [`Tensor`].
    /// * `secondary_tensor` - The right-hand side [`Tensor`].
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] containing the product of the input matrices.
    pub fn matrix_multiplication(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                matrixMultiplicationWithPrimaryTensor: primary_tensor,
                secondaryTensor: secondary_tensor,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Computes the hamming distance of two input tensors with support for broadcasting.
    ///
    /// The hamming distance is computed between 2 sets of vectors and the last dimension(s) of each
    /// input tensor is considered a vector.
    ///
    /// # Arguments
    ///
    /// * `primary_tensor` - The first input [`Tensor`].
    /// * `secondary_tensor` - The second input [`Tensor`].
    /// * `result_data_type` - The [`DataType`] of the result tensor. Must be `DataType::UInt32` or `DataType::UInt16`.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] containing the Hamming distance between the input tensors.
    pub fn hamming_distance(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        result_data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                HammingDistanceWithPrimaryTensor: primary_tensor,
                secondaryTensor: secondary_tensor,
                resultDataType: result_data_type,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Creates a scaled dot product attention (SDPA) operation and returns the result tensor.
    ///
    /// SDPA Op computes attention by computing softmax(scale * QK^T + M)V.
    /// queryTensor Q with shape [B, Hq, Nq, F] and keyTensor K with shape [B, Hq, Nkv, F],
    /// with Q's H dimension expandable to satisfy matmul QK^T. maskTensor M's shape
    /// should be broadcast compatible to satisfy (QK^T + M). valueTensor V with shape
    /// [B, Hv, Nkv, F] should satisfy the matmul (QK^T + M)V.
    ///
    /// # Arguments
    ///
    /// * `query_tensor` - A [`Tensor`] representing the query projection.
    /// * `key_tensor` - A [`Tensor`] representing the key projection.
    /// * `value_tensor` - A [`Tensor`] representing the value projection.
    /// * `mask_tensor` - Optional [`Tensor`] mask applied to the scaled `QK^T` matrix.
    /// * `scale` - Scale applied to the `QK^T` product before softmax.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] containing the SDPA result.
    pub fn sdpa_with_mask(
        &self,
        query_tensor: &Tensor,
        key_tensor: &Tensor,
        value_tensor: &Tensor,
        mask_tensor: Option<&Tensor>,
        scale: f64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                scaledDotProductAttentionWithQueryTensor: query_tensor,
                keyTensor: key_tensor,
                valueTensor: value_tensor,
                maskTensor: mask_tensor,
                scale: scale,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Creates a scaled dot product attention (SDPA) operation (without a mask) and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `query_tensor` - A [`Tensor`] representing the query projection.
    /// * `key_tensor` - A [`Tensor`] representing the key projection.
    /// * `value_tensor` - A [`Tensor`] representing the value projection.
    /// * `scale` - Scale applied to the `QK^T` product before softmax.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] containing the SDPA result.
    pub fn sdpa(
        &self,
        query_tensor: &Tensor,
        key_tensor: &Tensor,
        value_tensor: &Tensor,
        scale: f64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                scaledDotProductAttentionWithQueryTensor: query_tensor,
                keyTensor: key_tensor,
                valueTensor: value_tensor,
                scale: scale,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }
}
