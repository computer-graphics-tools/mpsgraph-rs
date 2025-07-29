use super::StartEndStrideScalarsOrTensors;
use crate::{Graph, Tensor, ns_number_array_from_slice};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Tiles `tensor` according to `multiplier`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `multiplier` – Number of repeats for each dimension (`rank` values).
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing repeated copies of `tensor`.
    pub fn tile(
        &self,
        tensor: &Tensor,
        multiplier: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                tileTensor: tensor,
                withMultiplier: &*ns_number_array_from_slice(multiplier),
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Computes the gradient for a tile operation.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient_tensor` – Gradient flowing from subsequent ops (`dL/dT`).
    /// * `source_tensor` – Original tensor fed to the forward *tile* op.
    /// * `multiplier` – Same multiplier used in the forward op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing the gradient with respect to `source_tensor`.
    pub fn tile_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        multiplier: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                tileGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                sourceTensor: source_tensor,
                withMultiplier: &*ns_number_array_from_slice(multiplier),
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }
}
