use objc::runtime::Object;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, MPSDataType};

/// OneHot operations for MPSGraph
impl MPSGraph {
    /// Creates a oneHot operation and returns the result tensor.
    /// 
    /// Creates a tensor of rank equal to the indicesTensor rank + 1. 
    /// Inserts a new axis at the axis specified, or the minor axis if axis is -1.
    /// The values at the indices in the indicesTensor will have the onValue, 
    /// and all other values will be set to the offValue.
    /// 
    /// - Parameters:
    ///   - indices_tensor: Tensor of indices for on values
    ///   - depth: Depth of the oneHot vector along the axis
    ///   - axis: The axis to insert the new oneHot vector at
    ///   - data_type: MPSDataType of the result tensor
    ///   - on_value: The value for indices designated by the indicesTensor
    ///   - off_value: The value for indices not designated by the indicesTensor
    ///   - name: Name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn one_hot(&self,
                  indices_tensor: &MPSGraphTensor,
                  depth: usize,
                  axis: usize,
                  data_type: MPSDataType,
                  on_value: f64,
                  off_value: f64,
                  name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let result: *mut Object = msg_send![self.0, 
                oneHotWithIndicesTensor:indices_tensor.0
                depth:depth
                axis:axis
                dataType:data_type as u64
                onValue:on_value
                offValue:off_value
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a oneHot operation with default axis (the minor dimension).
    /// 
    /// - Parameters:
    ///   - indices_tensor: Tensor of indices for on values
    ///   - depth: Depth of the oneHot vector along the axis
    ///   - data_type: MPSDataType of the result tensor
    ///   - on_value: The value for indices designated by the indicesTensor
    ///   - off_value: The value for indices not designated by the indicesTensor
    ///   - name: Name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn one_hot_default_axis(&self,
                               indices_tensor: &MPSGraphTensor,
                               depth: usize,
                               data_type: MPSDataType,
                               on_value: f64,
                               off_value: f64,
                               name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let result: *mut Object = msg_send![self.0, 
                oneHotWithIndicesTensor:indices_tensor.0
                depth:depth
                dataType:data_type as u64
                onValue:on_value
                offValue:off_value
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a oneHot operation with default on/off values (1.0 and 0.0).
    /// 
    /// - Parameters:
    ///   - indices_tensor: Tensor of indices for on values
    ///   - depth: Depth of the oneHot vector along the axis
    ///   - axis: The axis to insert the new oneHot vector at
    ///   - data_type: MPSDataType of the result tensor
    ///   - name: Name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn one_hot_default_values(&self,
                                 indices_tensor: &MPSGraphTensor,
                                 depth: usize,
                                 axis: usize,
                                 data_type: MPSDataType,
                                 name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let result: *mut Object = msg_send![self.0, 
                oneHotWithIndicesTensor:indices_tensor.0
                depth:depth
                axis:axis
                dataType:data_type as u64
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a oneHot operation with default axis and float32 data type (simplest version).
    /// 
    /// - Parameters:
    ///   - indices_tensor: Tensor of indices for on values
    ///   - depth: Depth of the oneHot vector along the axis
    ///   - name: Name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn one_hot_simple(&self,
                         indices_tensor: &MPSGraphTensor,
                         depth: usize,
                         name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let result: *mut Object = msg_send![self.0, 
                oneHotWithIndicesTensor:indices_tensor.0
                depth:depth
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a oneHot operation with default axis and default values.
    /// 
    /// - Parameters:
    ///   - indices_tensor: Tensor of indices for on values
    ///   - depth: Depth of the oneHot vector along the axis
    ///   - data_type: MPSDataType of the result tensor
    ///   - name: Name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn one_hot_default_axis_values(&self,
                                      indices_tensor: &MPSGraphTensor,
                                      depth: usize,
                                      data_type: MPSDataType,
                                      name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let result: *mut Object = msg_send![self.0, 
                oneHotWithIndicesTensor:indices_tensor.0
                depth:depth
                dataType:data_type as u64
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a oneHot operation with default data type (Float32) and default values.
    /// 
    /// - Parameters:
    ///   - indices_tensor: Tensor of indices for on values
    ///   - depth: Depth of the oneHot vector along the axis
    ///   - axis: The axis to insert the new oneHot vector at
    ///   - name: Name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn one_hot_default_type_values(&self,
                                      indices_tensor: &MPSGraphTensor,
                                      depth: usize,
                                      axis: usize,
                                      name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let result: *mut Object = msg_send![self.0, 
                oneHotWithIndicesTensor:indices_tensor.0
                depth:depth
                axis:axis
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
}