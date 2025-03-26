use objc::runtime::Object;
use objc::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, MPSDataType};

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
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                quantizeTensor:tensor.0
                scale:scale
                zeroPoint:zero_point
                dataType:data_type as u64
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
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
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                dequantizeTensor:tensor.0
                scale:scale
                zeroPoint:zero_point
                dataType:data_type as u64
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
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
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                quantizeTensor:tensor.0
                scaleTensor:scale_tensor.0
                zeroPoint:zero_point
                dataType:data_type as u64
                axis:axis
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
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
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                dequantizeTensor:tensor.0
                scaleTensor:scale_tensor.0
                zeroPoint:zero_point
                dataType:data_type as u64
                axis:axis
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
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
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                quantizeTensor:tensor.0
                scaleTensor:scale_tensor.0
                zeroPointTensor:zero_point_tensor.0
                dataType:data_type as u64
                axis:axis
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
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
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                dequantizeTensor:tensor.0
                scaleTensor:scale_tensor.0
                zeroPointTensor:zero_point_tensor.0
                dataType:data_type as u64
                axis:axis
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
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
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                dequantizeTensor:tensor.0
                LUTTensor:lut_tensor.0
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
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
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                dequantizeTensor:tensor.0
                LUTTensor:lut_tensor.0
                axis:axis
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MPSDataType, tests::should_skip_test};
    use crate::core::MPSShape;
    use std::collections::HashMap;
    
    #[test]
    fn test_quantize_dequantize() {
        if should_skip_test("test_quantize_dequantize") {
            return;
        }
        
        let graph = MPSGraph::new();
        
        // Create a float tensor
        let input_shape = MPSShape::from_slice(&[2, 2]);
        let input_data = vec![0.0f32, 1.0, 2.0, 3.0];
        
        let input_tensor = graph.placeholder(&input_shape, MPSDataType::Float32, None);
        
        // Quantize the tensor to INT8 with scale=0.5 and zero_point=0
        let quantized = graph.quantize(
            &input_tensor,
            0.5, // scale
            0.0, // zero_point
            MPSDataType::Int8,
            Some("quantize")
        );
        
        // Dequantize back to float
        let dequantized = graph.dequantize(
            &quantized,
            0.5, // scale
            0.0, // zero_point
            MPSDataType::Float32,
            Some("dequantize")
        );
        
        // Run the graph
        let mut feeds = HashMap::new();
        feeds.insert(&input_tensor, crate::MPSGraphTensorData::new(&input_data, &[2, 2], MPSDataType::Float32));
        
        let results = graph.run(feeds, &[&dequantized]);
        
        // Get the result data
        let result_data = results[&dequantized].to_vec::<f32>();
        
        // Due to quantization and dequantization, values might not be exactly the same as input
        // but they should be reasonably close
        for i in 0..input_data.len() {
            let diff = (input_data[i] - result_data[i]).abs();
            assert!(diff < 0.5, "Difference too large: {} vs {}", input_data[i], result_data[i]);
        }
    }
}