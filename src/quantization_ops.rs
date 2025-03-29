use crate::core::{AsRawObject, MPSDataType, NSString};
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;

/// Quantization operations for MPSGraph
impl MPSGraph {
    /// Creates a Quantize operation and returns the result tensor.
    ///
    /// Convert the float `tensor` to an i8 or u8 tensor by applying a scale + bias transform:
    /// result = (tensor / scale) + zeroPoint
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be quantized
    /// * `scale` - Scale scalar parameter
    /// * `zero_point` - Bias scalar parameter (converted to dataType of resultTensor)
    /// * `data_type` - Integer data type of the result tensor
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor of datatype `data_type`
    pub fn quantize(
        &self,
        tensor: &MPSGraphTensor,
        scale: f64,
        zero_point: f64,
        data_type: MPSDataType,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, quantizeTensor: tensor.0,
                scale: scale,
                zeroPoint: zero_point,
                dataType: data_type as u64,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a Dequantize operation and returns the result tensor.
    ///
    /// Convert the i8 or u8 `tensor` to a float tensor by applying a scale + bias transform:
    /// result = scale(tensor - zeroPoint)
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be dequantized
    /// * `scale` - Scale scalar parameter
    /// * `zero_point` - Bias scalar parameter (converted to dataType of tensor)
    /// * `data_type` - Float data type of the result tensor
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor of datatype `data_type`
    pub fn dequantize(
        &self,
        tensor: &MPSGraphTensor,
        scale: f64,
        zero_point: f64,
        data_type: MPSDataType,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, dequantizeTensor: tensor.0,
                scale: scale,
                zeroPoint: zero_point,
                dataType: data_type as u64,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a Quantize operation with scale tensor and returns the result tensor.
    ///
    /// Convert the float `tensor` to an i8 or u8 tensor by applying a scale + bias transform:
    /// result = (tensor / scaleTensor) + zeroPoint
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be quantized
    /// * `scale_tensor` - Scale 1D Tensor parameter with size == tensor.shape[axis]
    /// * `zero_point` - Bias scalar parameter (converted to dataType of resultTensor)
    /// * `data_type` - Integer data type of the result tensor
    /// * `axis` - Axis on which the scale 1D value is being broadcasted
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor of datatype `data_type`
    pub fn quantize_with_scale_tensor(
        &self,
        tensor: &MPSGraphTensor,
        scale_tensor: &MPSGraphTensor,
        zero_point: f64,
        data_type: MPSDataType,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, quantizeTensor: tensor.0,
                scaleTensor: scale_tensor.0,
                zeroPoint: zero_point,
                dataType: data_type as u64,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a Dequantize operation with scale tensor and returns the result tensor.
    ///
    /// Convert the i8 or u8 `tensor` to a float tensor by applying a scale + bias transform:
    /// result = scaleTensor(tensor - zeroPoint)
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be dequantized
    /// * `scale_tensor` - Scale 1D Tensor parameter with size == tensor.shape[axis]
    /// * `zero_point` - Bias scalar parameter (converted to dataType of tensor)
    /// * `data_type` - Float data type of the result tensor
    /// * `axis` - Axis on which the scale 1D value is being broadcasted
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor of datatype `data_type`
    pub fn dequantize_with_scale_tensor(
        &self,
        tensor: &MPSGraphTensor,
        scale_tensor: &MPSGraphTensor,
        zero_point: f64,
        data_type: MPSDataType,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, dequantizeTensor: tensor.0,
                scaleTensor: scale_tensor.0,
                zeroPoint: zero_point,
                dataType: data_type as u64,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a Quantize operation with scale and zero point tensors and returns the result tensor.
    ///
    /// Convert the float `tensor` to an i8 or u8 tensor by applying a scale + bias transform:
    /// result = (tensor / scaleTensor) + zeroPointTensor
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be quantized
    /// * `scale_tensor` - Scale 1D Tensor parameter with size == tensor.shape[axis]
    /// * `zero_point_tensor` - Bias 1D Tensor parameter with size == tensor.shape[axis]
    /// * `data_type` - Integer data type of the result tensor
    /// * `axis` - Axis on which the scale 1D value is being broadcasted
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor of datatype `data_type`
    pub fn quantize_with_tensors(
        &self,
        tensor: &MPSGraphTensor,
        scale_tensor: &MPSGraphTensor,
        zero_point_tensor: &MPSGraphTensor,
        data_type: MPSDataType,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, quantizeTensor: tensor.0,
                scaleTensor: scale_tensor.0,
                zeroPointTensor: zero_point_tensor.0,
                dataType: data_type as u64,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a Dequantize operation with scale and zero point tensors and returns the result tensor.
    ///
    /// Convert the i8 or u8 `tensor` to a float tensor by applying a scale + bias transform:
    /// result = scaleTensor(tensor - zeroPointTensor)
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be dequantized
    /// * `scale_tensor` - Scale 1D Tensor parameter with size == tensor.shape[axis]
    /// * `zero_point_tensor` - Bias 1D Tensor parameter with size == tensor.shape[axis]
    /// * `data_type` - Float data type of the result tensor
    /// * `axis` - Axis on which the scale 1D value is being broadcasted
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor of datatype `data_type`
    pub fn dequantize_with_tensors(
        &self,
        tensor: &MPSGraphTensor,
        scale_tensor: &MPSGraphTensor,
        zero_point_tensor: &MPSGraphTensor,
        data_type: MPSDataType,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, dequantizeTensor: tensor.0,
                scaleTensor: scale_tensor.0,
                zeroPointTensor: zero_point_tensor.0,
                dataType: data_type as u64,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a lookup-table based dequantization operation and returns the result tensor.
    ///
    /// Converts a u8 or u4 `tensor` to a float tensor by applying a lookup operation:
    /// result[i1,...,in] = LUTTensor[i1',...,in',tensor[i1,...,in]].
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be dequantized
    /// * `lut_tensor` - The lookup table to use - for u4 the last dimension should have 16 elements, and for u8 256 elements
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object
    pub fn dequantize_with_lut(
        &self,
        tensor: &MPSGraphTensor,
        lut_tensor: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, dequantizeTensor: tensor.0,
                LUTTensor: lut_tensor.0,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a vector lookup-table based dequantization operation and returns the result tensor.
    ///
    /// Converts a u8 or u4 `tensor` to a float tensor by applying a lookup operation.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be dequantized
    /// * `lut_tensor` - The lookup table to use - for u4 the second to last dimension should have 16 elements, and for u8 256 elements
    /// * `axis` - Axis on which the scale 1D value is being broadcasted
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object
    pub fn dequantize_with_lut_axis(
        &self,
        tensor: &MPSGraphTensor,
        lut_tensor: &MPSGraphTensor,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, dequantizeTensor: tensor.0,
                LUTTensor: lut_tensor.0,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
}
