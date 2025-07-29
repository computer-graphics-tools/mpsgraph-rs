use crate::Tensor;

/// Helper enum for specifying the three axis indices required by the
/// *space-to-depth* / *depth-to-space* 2-D transforms.
///
/// The axes can be provided either as compile-time scalar values (`u64`) or as
/// graph [`Tensor`] inputs, allowing dynamic graph construction.
///
pub enum WidthHeightDepthAxisScalarsOrTensors<'a> {
    Scalars {
        /// The axis that defines the fastest running dimension within the block.
        width_axis: u64,
        /// The axis that defines the 2nd fastest running dimension within the block.
        height_axis: u64,
        /// The axis that defines the destination dimension, where to copy the blocks.
        depth_axis: u64,
    },
    Tensors {
        /// A scalar tensor that contains the axis that defines the fastest running dimension within the block.
        width_axis_tensor: &'a Tensor,
        /// A scalar tensor that contains the axis that defines the 2nd fastest running dimension within the block.
        height_axis_tensor: &'a Tensor,
        /// A scalar tensor that contains the axis that defines the destination dimension, where to copy the blocks.
        depth_axis_tensor: &'a Tensor,
    },
}
