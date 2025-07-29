use crate::{DataType, Tensor};

/// Arguments controlling how a floating-point tensor is quantized into an
/// integer representation.
///
/// All variants follow the same conceptual formula but differ in whether the
/// scale and zero-point are provided as scalars or tensors and whether the
/// scaling is applied per-axis.
pub enum QuantizationArguments<'a> {
    /// Quantize using scalar `scale` / `zero_point`.
    ///
    /// # Details
    ///
    /// Formula: `result = (tensor / scale) + zero_point`.
    ScaleZeroPointDataType {
        /// Scale scalar parameter
        scale: f64,
        /// Bias scalar parameter (converted to dataType of resultTensor)
        zero_point: i64,
        /// Integer data type of the result tensor.
        data_type: DataType,
    },
    /// Quantize using a per-axis `scale_tensor` and scalar `zero_point`.
    ///
    /// # Details
    ///
    /// Formula: `result = (tensor / scale_tensor) + zero_point`.
    ScaleTensorZeroPointDataTypeAxis {
        /// Scale 1D Tensor parameter with size == tensor.shape[axis]
        scale_tensor: &'a Tensor,
        /// Bias scalar parameter (converted to dataType of resultTensor)
        zero_point: f64,
        /// Integer data type of the result tensor.
        data_type: DataType,
        /// Axis on which the scale 1D value is being broadcasted
        axis: i64,
    },
    /// Quantize using per-axis `scale_tensor` and `zero_point_tensor`.
    ///
    /// # Details
    ///
    /// Formula: `result = (tensor / scale_tensor) + zero_point_tensor`.
    ScaleTensorZeroPointTensorDataTypeAxis {
        /// Scale scalar or 1D Tensor parameter with size == tensor.shape[axis]
        scale_tensor: &'a Tensor,
        /// Bias scalar or 1D Tensor parameter with size == tensor.shape[axis]
        zero_point_tensor: &'a Tensor,
        /// Integer data type of the result tensor.
        data_type: DataType,
        /// Axis on which the scale 1D value is being broadcasted
        axis: i64,
    },
}

/// Arguments controlling how an integer tensor is dequantized back to
/// floating-point.
///
/// Each variant specifies the combination of scale / zero-point / LUT inputs
/// and whether they are scalars or tensors.
pub enum DequantizationArguments<'a> {
    /// Dequantize using scalar `scale` / `zero_point`.
    ///
    /// # Details
    ///
    /// Formula: `result = scale * (tensor - zero_point)`.
    ScaleZeroPointDataType {
        /// Scale scalar parameter
        scale: f64,
        /// Bias scalar parameter (converted to dataType of resultTensor)
        zero_point: i64,
        /// Integer data type of the result tensor.
        data_type: DataType,
    },
    /// Dequantize using per-axis `scale_tensor` and scalar `zero_point`.
    ///
    /// # Details
    ///
    /// Formula: `result = scale_tensor * (tensor - zero_point)`.
    ScaleTensorZeroPointDataTypeAxis {
        /// Scale 1D Tensor parameter with size == tensor.shape[axis]
        scale_tensor: &'a Tensor,
        /// Bias scalar parameter (converted to dataType of resultTensor)
        zero_point: f64,
        /// Integer data type of the result tensor.
        data_type: DataType,
        /// Axis on which the scale 1D value is being broadcasted
        axis: i64,
    },
    /// Dequantize using per-axis `scale_tensor` and `zero_point_tensor`.
    ///
    /// # Details
    ///
    /// Formula: `result = scale_tensor * (tensor - zero_point_tensor)`.
    ScaleTensorZeroPointTensorDataTypeAxis {
        /// Scale scalar or 1D Tensor parameter with size == tensor.shape[axis]
        scale_tensor: &'a Tensor,
        /// Bias scalar or 1D Tensor parameter with size == tensor.shape[axis]
        zero_point_tensor: &'a Tensor,
        /// Integer data type of the result tensor.
        data_type: DataType,
        /// Axis on which the scale 1D value is being broadcasted
        axis: i64,
    },
    /// Dequantize using grouped `scale_tensor` / `zero_point_tensor` (no axis).
    ///
    /// # Details
    ///
    /// Formula: `result = scale_tensor * (tensor - zero_point_tensor)`.
    ScaleTensorZeroPointTensorDataType {
        /// The scale tensor with groups support.
        scale_tensor: &'a Tensor,
        /// The bias tensor with groups support.
        zero_point_tensor: &'a Tensor,
        /// Float data type of the result tensor.
        data_type: DataType,
    },
    /// Dequantize using grouped `scale_tensor` (no zero-point).
    ///
    /// # Details
    ///
    /// Formula: `result = scale_tensor * tensor`.
    ScaleTensorDataType {
        /// Scale Tensor parameter with groups support.
        scale_tensor: &'a Tensor,
        /// Float data type of the result tensor.
        data_type: DataType,
    },
    /// Dequantize via lookup-table (`lut_tensor`).
    ///
    /// # Details
    ///
    /// Formula: `result[i,…] = lut_tensor[..., tensor[i,…]]`.
    LutTensor {
        /// The lookup table to use - for u4 the last dimension should have 16 elements, and for u8 256 elements.
        lut_tensor: &'a Tensor,
    },
    /// Dequantize via lookup-table (`lut_tensor`) with an explicit output axis.
    ///
    /// # Details
    ///
    /// Formula: `result = lut_tensor[..., tensor[...], axis]`.
    LutTensorAxis {
        /// The lookup table to use - for u4 the second to last dimension should have 16 elements, and for u8 256 elements.
        lut_tensor: &'a Tensor,
        /// Axis on which the scale 1D value is being broadcasted.
        axis: i64,
    },
}
