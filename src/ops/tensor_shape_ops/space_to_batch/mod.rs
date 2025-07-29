mod scalars_or_tensors;

pub use scalars_or_tensors::SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors;

use crate::{Graph, Tensor, ns_number_array_from_slice};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Creates a *space-to-batch* operation.
    ///
    /// Values from the `spatial_axes` dimensions are packed into blocks of
    /// `block_dimensions` and moved to the `batch_axis` dimension.
    /// Setting `use_pixel_shuffle_order` controls whether the blocks are stored
    /// contiguously (pixel-shuffle order) or interleaved.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `spatial_axes_batch_axis_block_dimensions` – Tuple of spatial axes,
    ///   destination batch axis, and per-axis block sizes. Accepts scalars or
    ///   tensors via [`SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors`].
    /// * `use_pixel_shuffle_order` – If `true`, blocks are laid out
    ///   contiguously along the batch axis.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] with spatial data moved into the batch dimension.
    pub fn space_to_batch<'a>(
        &self,
        tensor: &Tensor,
        spatial_axes_batch_axis_block_dimensions: SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors<'a>,
        use_pixel_shuffle_order: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match spatial_axes_batch_axis_block_dimensions {
            SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors::Scalars {
                spatial_axes,
                batch_axis,
                block_dimensions,
            } => unsafe {
                msg_send![
                    self,
                    spaceToBatchTensor: tensor,
                    spatialAxes: &*ns_number_array_from_slice(spatial_axes),
                    batchAxis: batch_axis,
                    blockDimensions: &*ns_number_array_from_slice(block_dimensions),
                    usePixelShuffleOrder: use_pixel_shuffle_order,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors::Tensors {
                spatial_axes_tensor,
                batch_axis_tensor,
                block_dimensions_tensor,
            } => unsafe {
                msg_send![
                    self,
                    spaceToBatchTensor: tensor,
                    spatialAxesTensor: spatial_axes_tensor,
                    batchAxisTensor: batch_axis_tensor,
                    blockDimensionsTensor: block_dimensions_tensor,
                    usePixelShuffleOrder: use_pixel_shuffle_order,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Creates a *batch-to-space* operation (inverse of space-to-batch).
    ///
    /// Values from the `batch_axis` are unpacked into spatial blocks of
    /// `block_dimensions` and distributed across the `spatial_axes`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `spatial_axes_batch_axis_block_dimensions` – Tuple of spatial axes,
    ///   source batch axis, and per-axis block sizes.
    /// * `use_pixel_shuffle_order` – If `true`, expects contiguous
    ///   pixel-shuffle layout in the batch dimension.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] with batch data moved back into spatial dimensions.
    pub fn batch_to_space<'a>(
        &self,
        tensor: &Tensor,
        spatial_axes_batch_axis_block_dimensions: SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors<'a>,
        use_pixel_shuffle_order: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match spatial_axes_batch_axis_block_dimensions {
            SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors::Scalars {
                spatial_axes,
                batch_axis,
                block_dimensions,
            } => unsafe {
                msg_send![
                    self,
                    batchToSpaceTensor: tensor,
                    spatialAxes: &*ns_number_array_from_slice(spatial_axes),
                    batchAxis: batch_axis,
                    blockDimensions: &*ns_number_array_from_slice(block_dimensions),
                    usePixelShuffleOrder: use_pixel_shuffle_order,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors::Tensors {
                spatial_axes_tensor,
                batch_axis_tensor,
                block_dimensions_tensor,
            } => unsafe {
                msg_send![
                    self,
                    batchToSpaceTensor: tensor,
                    spatialAxesTensor: spatial_axes_tensor,
                    batchAxisTensor: batch_axis_tensor,
                    blockDimensionsTensor: block_dimensions_tensor,
                    usePixelShuffleOrder: use_pixel_shuffle_order,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
