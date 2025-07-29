use crate::{Graph, ShapeOrTensor, Tensor, ns_number_array_from_slice};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Broadcasts `tensor` to a new shape.
    ///
    /// Semantics match NumPy-style broadcasting: starting from the trailing
    /// dimensions, singleton axes are repeated to match `shape`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Tensor to broadcast.
    /// * `shape` – Desired output shape specified either as a slice or as a
    ///   [`Tensor`] (see [`ShapeOrTensor`]).
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A broadcasted [`Tensor`] with the requested shape.
    pub fn broadcast<'a>(
        &self,
        tensor: &Tensor,
        shape: ShapeOrTensor<'a>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match shape {
            ShapeOrTensor::Shape(shape) => unsafe {
                msg_send![
                    self,
                    broadcastTensor: tensor,
                    toShape: &*ns_number_array_from_slice(shape),
                    name: name.map(NSString::from_str).as_deref()
                ]
            },
            ShapeOrTensor::Tensor(shape) => unsafe {
                msg_send![
                    self,
                    broadcastTensor: tensor,
                    toShapeTensor: shape,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
