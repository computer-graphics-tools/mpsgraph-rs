//! Space-to-depth and depth-to-space helper operations.
//!
//! These are 2-D specializations that pack or unpack spatial blocks between
//! the H/W axes and the depth axis, optionally using *pixel-shuffle* ordering
//! (contiguous vs interleaved).
//!

mod scalars_or_tensors;

pub use scalars_or_tensors::WidthHeightDepthAxisScalarsOrTensors;

use crate::{Graph, Tensor};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Creates a *space-to-depth 2-D* operation.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `axis` – Tuple specifying width-, height-, and depth-axis indices (see
    ///   [`WidthHeightDepthAxisScalarsOrTensors`]).
    /// * `block_size` – Size of the square spatial block.
    /// * `use_pixel_shuffle_order` – If `true`, blocks are stored contiguously
    ///   along the depth axis (pixel-shuffle order).
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] where H/W spatial blocks are packed into the depth axis.
    pub fn space_to_depth_2d<'a>(
        &self,
        tensor: &Tensor,
        axis: WidthHeightDepthAxisScalarsOrTensors<'a>,
        block_size: u64,
        use_pixel_shuffle_order: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            WidthHeightDepthAxisScalarsOrTensors::Scalars {
                width_axis,
                height_axis,
                depth_axis,
            } => unsafe {
                msg_send![
                    self,
                    spaceToDepth2DTensor: tensor,
                    widthAxis: width_axis,
                    heightAxis: height_axis,
                    depthAxis: depth_axis,
                    blockSize: block_size,
                    usePixelShuffleOrder: use_pixel_shuffle_order,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            WidthHeightDepthAxisScalarsOrTensors::Tensors {
                width_axis_tensor,
                height_axis_tensor,
                depth_axis_tensor,
            } => unsafe {
                msg_send![
                    self,
                    spaceToDepth2DTensor: tensor,
                    widthAxisTensor: width_axis_tensor,
                    heightAxisTensor: height_axis_tensor,
                    depthAxisTensor: depth_axis_tensor,
                    blockSize: block_size,
                    usePixelShuffleOrder: use_pixel_shuffle_order,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Creates a *depth-to-space 2-D* operation (inverse of space-to-depth).
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `axis` – Tuple specifying width-, height-, and depth-axis indices.
    /// * `block_size` – Size of the square spatial block.
    /// * `use_pixel_shuffle_order` – If `true`, expects contiguous pixel-shuffle
    ///   layout in the depth axis.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] where data from the depth axis is unpacked into H/W axes.
    pub fn depth_to_space_2d<'a>(
        &self,
        tensor: &Tensor,
        axis: WidthHeightDepthAxisScalarsOrTensors<'a>,
        block_size: u64,
        use_pixel_shuffle_order: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            WidthHeightDepthAxisScalarsOrTensors::Scalars {
                width_axis,
                height_axis,
                depth_axis,
            } => unsafe {
                msg_send![
                    self,
                    depthToSpace2DTensor: tensor,
                    widthAxis: width_axis,
                    heightAxis: height_axis,
                    depthAxis: depth_axis,
                    blockSize: block_size,
                    usePixelShuffleOrder: use_pixel_shuffle_order,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            WidthHeightDepthAxisScalarsOrTensors::Tensors {
                width_axis_tensor,
                height_axis_tensor,
                depth_axis_tensor,
            } => unsafe {
                msg_send![
                    self,
                    depthToSpace2DTensor: tensor,
                    widthAxisTensor: width_axis_tensor,
                    heightAxisTensor: height_axis_tensor,
                    depthAxisTensor: depth_axis_tensor,
                    blockSize: block_size,
                    usePixelShuffleOrder: use_pixel_shuffle_order,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
