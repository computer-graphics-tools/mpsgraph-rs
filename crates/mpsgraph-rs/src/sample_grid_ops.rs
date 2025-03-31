use objc2::runtime::AnyObject;
// In objc2, use false as NO and true as YES
const NO: bool = false;
const YES: bool = true;
use crate::convolution_transpose_ops::TensorNamedDataLayout;
use crate::core::{AsRawObject, NSString};
use crate::graph::MPSGraph;
use crate::resize_ops::{MPSGraphResizeMode, MPSGraphResizeNearestRoundingMode};
use crate::tensor::MPSGraphTensor;
use objc2::msg_send;

/// Padding modes for MPSGraph operations
#[repr(i64)]
#[derive(Debug, Copy, Clone)]
pub enum MPSGraphPaddingMode {
    /// Constant padding
    Constant = 0,
    /// Reflect padding
    Reflect = 1,
    /// Symmetric padding
    Symmetric = 2,
    /// Clamp to edge padding (PyTorch ReplicationPad)
    ClampToEdge = 3,
    /// Zero padding
    Zero = 4,
    /// Periodic padding (x[-2] -> x[L-3], where L is size of x)
    Periodic = 5,
    /// Anti-periodic padding (x[-2] -> -x[L-3])
    AntiPeriodic = 6,
}

/// Sample Grid operations for MPSGraph
impl MPSGraph {
    /// Samples a tensor using the coordinates provided.
    ///
    /// Given an input tensor (N, H1, W1, C) or (N, C, H1, W1) and coordinates tensor (N, H2, W2, 2)
    /// this operation outputs a tensor of size (N, H2, W2, C) or (N, C, H2, W2) by sampling the
    /// input tensor at the coordinates provided by the coordinates tensor.
    ///
    /// - Parameters:
    ///   - source: Tensor containing source data
    ///   - coordinates: a tensor (N, Hout, Wout, 2) that contains the coordinates of the samples in the source tensor
    ///   - layout: Specifies what layout the provided tensor is in
    ///   - normalize_coordinates: If true, coordinates are within [-1, 1] x [-1, 1] otherwise they are in pixels
    ///   - relative_coordinates: If true, coordinates are relative to the position of the pixel in the output tensor
    ///   - align_corners: If true, coordinate extrema are equal to the center of edge pixels
    ///   - padding_mode: determines how samples outside the inputTensor are evaluated
    ///   - sampling_mode: Can be either MPSGraphResizeNearest or MPSGraphResizeBilinear
    ///   - constant_value: If paddingMode is Constant, then this constant is used for samples outside the input tensor
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn sample_grid(
        &self,
        source: &MPSGraphTensor,
        coordinates: &MPSGraphTensor,
        layout: TensorNamedDataLayout,
        normalize_coordinates: bool,
        relative_coordinates: bool,
        align_corners: bool,
        padding_mode: MPSGraphPaddingMode,
        sampling_mode: MPSGraphResizeMode,
        constant_value: f64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let normalize_coordinates_val = if normalize_coordinates { YES } else { NO };
            let relative_coordinates_val = if relative_coordinates { YES } else { NO };
            let align_corners_val = if align_corners { YES } else { NO };

            let result: *mut AnyObject = msg_send![self.0, sampleGridWithSourceTensor: source.0,
                coordinateTensor: coordinates.0,
                layout: layout as u64,
                normalizeCoordinates: normalize_coordinates_val,
                relativeCoordinates: relative_coordinates_val,
                alignCorners: align_corners_val,
                paddingMode: padding_mode as i64,
                samplingMode: sampling_mode as u64,
                constantValue: constant_value,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            MPSGraphTensor(result)
        }
    }

    /// Samples a tensor using the coordinates provided, using nearest neighbor sampling with specified rounding mode.
    ///
    /// - Parameters:
    ///   - source: Tensor containing source data
    ///   - coordinates: a tensor (N, Hout, Wout, 2) that contains the coordinates of the samples in the source tensor
    ///   - layout: Specifies what layout the provided tensor is in
    ///   - normalize_coordinates: If true, coordinates are within [-1, 1] x [-1, 1] otherwise they are in pixels
    ///   - relative_coordinates: If true, coordinates are relative to the position of the pixel in the output tensor
    ///   - align_corners: If true, coordinate extrema are equal to the center of edge pixels
    ///   - padding_mode: determines how samples outside the inputTensor are evaluated
    ///   - nearest_rounding_mode: The rounding mode to use for determining the nearest neighbor
    ///   - constant_value: If paddingMode is Constant, then this constant is used for samples outside the input tensor
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn sample_grid_nearest(
        &self,
        source: &MPSGraphTensor,
        coordinates: &MPSGraphTensor,
        layout: TensorNamedDataLayout,
        normalize_coordinates: bool,
        relative_coordinates: bool,
        align_corners: bool,
        padding_mode: MPSGraphPaddingMode,
        nearest_rounding_mode: MPSGraphResizeNearestRoundingMode,
        constant_value: f64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let normalize_coordinates_val = if normalize_coordinates { YES } else { NO };
            let relative_coordinates_val = if relative_coordinates { YES } else { NO };
            let align_corners_val = if align_corners { YES } else { NO };

            let result: *mut AnyObject = msg_send![self.0, sampleGridWithSourceTensor: source.0,
                coordinateTensor: coordinates.0,
                layout: layout as u64,
                normalizeCoordinates: normalize_coordinates_val,
                relativeCoordinates: relative_coordinates_val,
                alignCorners: align_corners_val,
                paddingMode: padding_mode as i64,
                nearestRoundingMode: nearest_rounding_mode as u64,
                constantValue: constant_value,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            MPSGraphTensor(result)
        }
    }
}
