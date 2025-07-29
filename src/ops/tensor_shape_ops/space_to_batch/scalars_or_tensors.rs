use crate::Tensor;

/// Helper enum for specifying *spatial axes*, *batch axis*, and *block
/// dimensions* either as plain Rust slices or as graph [`Tensor`] inputs.
///
/// This polymorphism allows `space_to_batch` / `batch_to_space` to accept
/// compile-time constants as well as values computed dynamically during graph
/// construction.
///
pub enum SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors<'a> {
    Scalars {
        /// The axes that define the dimensions containing the spatial blocks.
        spatial_axes: &'a [u64],
        /// The axis that defines the destination dimension, where to copy the blocks.
        batch_axis: i64,
        /// An array of numbers that defines the size of the rectangular spatial sub-block.
        block_dimensions: &'a [u64],
    },
    Tensors {
        /// A tensor that contains the axes that define the dimensions containing the spatial blocks.
        spatial_axes_tensor: &'a Tensor,
        /// A tensor that contains the axis that defines the destination dimension, where to copy the blocks.
        batch_axis_tensor: &'a Tensor,
        /// A tensor that defines the size of the rectangular spatial sub-block.
        block_dimensions_tensor: &'a Tensor,
    },
}
