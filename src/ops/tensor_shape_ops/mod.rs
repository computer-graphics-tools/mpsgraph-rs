mod axes_or_tensor;
mod broadcast_ops;
mod concat_ops;
mod coordinate_ops;
mod expand_dims_ops;
mod flatten_ops;
mod pad_ops;
mod reverse_tensor_ops;
mod slice_ops;
mod space_to_batch;
mod space_to_depth_ops;
mod split_ops;
mod squeeze_ops;
mod tile_ops;

pub use axes_or_tensor::*;
pub use broadcast_ops::*;
pub use concat_ops::*;
pub use coordinate_ops::*;
pub use expand_dims_ops::*;
pub use flatten_ops::*;
pub use pad_ops::*;
pub use reverse_tensor_ops::*;
pub use slice_ops::*;
pub use space_to_batch::*;
pub use space_to_depth_ops::*;
pub use split_ops::*;
pub use squeeze_ops::*;
pub use tile_ops::*;

/// Tensor shape-manipulation helpers (reshape, transpose, stack, etc.).
///
/// All functions live as extension methods on [`Graph`] and mirror their
/// Objective-C counterparts while using idiomatic Rust types.
///
use crate::{DataType, Graph, ShapeOrTensor, Tensor, ns_number_array_from_slice};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSString};

impl Graph {
    /// Reshapes `tensor` to `shape`.
    ///
    /// The total number of elements must not change. A value of `-1` in `shape`
    /// denotes a dynamic dimension that will be inferred.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Tensor to reshape.
    /// * `shape` – Desired shape (slice or tensor).
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] with the requested shape.
    pub fn reshape<'a>(
        &self,
        tensor: &Tensor,
        shape: ShapeOrTensor<'a>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match shape {
            ShapeOrTensor::Shape(shape) => unsafe {
                msg_send![
                    self,
                    reshapeTensor: tensor,
                    withShape: &*ns_number_array_from_slice(shape),
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ShapeOrTensor::Tensor(shape_tensor) => unsafe {
                msg_send![
                    self,
                    reshapeTensor: tensor,
                    withShapeTensor: shape_tensor,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Permutes the axes of `tensor`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Tensor to transpose.
    /// * `permutation` – Permutation vector (`len == rank`).
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] with permuted axes.
    pub fn transpose(
        &self,
        tensor: &Tensor,
        permutation: &[u64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                transposeTensor: tensor,
                permutation: &*ns_number_array_from_slice(permutation),
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Returns the static shape of `tensor` as a rank-1 int32 tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A rank-1 [`Tensor`] containing the shape.
    pub fn shape_of(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![self, shapeOfTensor: tensor, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Casts `tensor` to `type`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `type` – Destination [`DataType`].
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A tensor with the same data reinterpreted as `type`.
    pub fn cast(&self, tensor: &Tensor, r#type: DataType, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![self, castTensor: tensor, toType: r#type, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Reinterprets the underlying bytes of `tensor` as a different element type.
    ///
    /// The total byte size is preserved; the last dimension is scaled by
    /// `sizeof(old_type) / sizeof(type)`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `type` – Target [`DataType`].
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A tensor sharing the same buffer but with a new element type.
    pub fn reinterpret_cast(
        &self,
        tensor: &Tensor,
        r#type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![self, reinterpretCastTensor: tensor, toType: r#type, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Stacks `input_tensors` along a new `axis` (rank increases by 1).
    ///
    /// All tensors must be broadcast compatible along the existing dimensions
    /// and share the same datatype.
    ///
    /// # Arguments
    ///
    /// * `input_tensors` – Slice of tensors to stack.
    /// * `axis` – Axis index for the new dimension (supports negative indexing).
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A stacked [`Tensor`].
    pub fn stack(
        &self,
        input_tensors: &[&Tensor],
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let input_tensors_array = NSArray::from_slice(input_tensors);
        unsafe {
            msg_send![
                self,
                stackTensors: &*input_tensors_array,
                axis: axis,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }
}
