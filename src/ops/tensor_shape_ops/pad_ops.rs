use crate::{Graph, PaddingMode, Shape, Tensor, ns_number_array_from_slice};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Pads `tensor` according to `padding_mode`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `padding_mode` – Padding strategy (constant, reflection, etc.).
    /// * `left_padding` – Padding sizes before each dimension (`rank` elements).
    /// * `right_padding` – Padding sizes after each dimension (`rank` elements).
    /// * `constant_value` – Value used when `padding_mode` is constant.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A padded [`Tensor`].
    pub fn pad(
        &self,
        tensor: &Tensor,
        padding_mode: PaddingMode,
        left_padding: &[i64],
        right_padding: &[i64],
        constant_value: f64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                padTensor: tensor,
                withPaddingMode: padding_mode,
                leftPadding: &*ns_number_array_from_slice(left_padding),
                rightPadding: &*ns_number_array_from_slice(right_padding),
                constantValue: constant_value,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Computes the gradient for a padding operation.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient_tensor` – Gradient flowing from subsequent ops (`dL/dP`).
    /// * `source_tensor` – Original unpadded tensor from the forward pass.
    /// * `padding_mode` – Padding strategy used in the forward op.
    /// * `left_padding` – Padding sizes before each dimension.
    /// * `right_padding` – Padding sizes after each dimension.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing the gradient with respect to the unpadded input.
    pub fn pad_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        padding_mode: PaddingMode,
        left_padding: &[i64],
        right_padding: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                padGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                sourceTensor: source_tensor,
                paddingMode: padding_mode,
                leftPadding: &*ns_number_array_from_slice(left_padding),
                rightPadding: &*ns_number_array_from_slice(right_padding),
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }
}
