use objc2::runtime::AnyObject;
use objc2::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, AsRawObject};
use crate::shape::MPSShape;

/// Scatter operation mode
#[repr(i64)]
#[derive(Debug, Copy, Clone)]
pub enum MPSGraphScatterMode {
    /// Add values
    Add = 0,
    /// Subtract values
    Sub = 1, 
    /// Multiply values
    Mul = 2,
    /// Divide values
    Div = 3,
    /// Take minimum value
    Min = 4,
    /// Take maximum value
    Max = 5,
    /// Set value (overwrite)
    Set = 6,
}

/// ScatterND operations for MPSGraph
impl MPSGraph {
    /// Creates a ScatterND operation and returns the result tensor.
    /// 
    /// Scatters the slices in updates_tensor to the result tensor along the indices in indices_tensor.
    /// 
    /// - Parameters:
    ///   - updates_tensor: Tensor containing slices to be inserted into the result tensor
    ///   - indices_tensor: Tensor containing the result indices to insert slices at
    ///   - shape: The shape of the result tensor
    ///   - batch_dimensions: The number of batch dimensions
    ///   - mode: The type of update to use on the destination
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn scatter_nd(&self,
                     updates_tensor:  &MPSGraphTensor,
                     indices_tensor:  &MPSGraphTensor,
                     shape:  &MPSShape,
                     batch_dimensions:  usize,
                     mode:  MPSGraphScatterMode,
                     name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let result: *mut AnyObject = msg_send![self.0, scatterNDWithUpdatesTensor: updates_tensor.0,
                indicesTensor: indices_tensor.0,
                shape: shape.0,
                batchDimensions: batch_dimensions,
                mode: mode as i64,
                name: name_obj
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a ScatterND operation with Add mode.
    /// 
    /// - Parameters:
    ///   - updates_tensor: Tensor containing slices to be inserted into the result tensor
    ///   - indices_tensor: Tensor containing the result indices to insert slices at
    ///   - shape: The shape of the result tensor
    ///   - batch_dimensions: The number of batch dimensions
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn scatter_nd_add(&self,
                         updates_tensor:  &MPSGraphTensor,
                         indices_tensor:  &MPSGraphTensor,
                         shape:  &MPSShape,
                         batch_dimensions:  usize,
                         name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let result: *mut AnyObject = msg_send![self.0, scatterNDWithUpdatesTensor: updates_tensor.0,
                indicesTensor: indices_tensor.0,
                shape: shape.0,
                batchDimensions: batch_dimensions,
                name: name_obj,
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a ScatterND operation with a data tensor as base.
    /// 
    /// Scatters the slices in updates_tensor to the result tensor along the indices in indices_tensor, on top of data_tensor.
    /// 
    /// - Parameters:
    ///   - data_tensor: Tensor containing initial values of same shape as result tensor
    ///   - updates_tensor: Tensor containing slices to be inserted into the result tensor
    ///   - indices_tensor: Tensor containing the result indices to insert slices at
    ///   - batch_dimensions: The number of batch dimensions
    ///   - mode: The type of update to use on the destination
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn scatter_nd_with_data(&self,
                               data_tensor:  &MPSGraphTensor,
                               updates_tensor:  &MPSGraphTensor,
                               indices_tensor:  &MPSGraphTensor,
                               batch_dimensions:  usize,
                               mode:  MPSGraphScatterMode,
                               name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let result: *mut AnyObject = msg_send![self.0, scatterNDWithDataTensor: data_tensor.0,
                updatesTensor: updates_tensor.0,
                indicesTensor: indices_tensor.0,
                batchDimensions: batch_dimensions,
                mode: mode as i64,
                name: name_obj
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a Scatter operation and returns the result tensor.
    /// 
    /// Scatters the values in updates_tensor to the result tensor along the indices in indices_tensor.
    /// 
    /// - Parameters:
    ///   - updates_tensor: Tensor containing values to be inserted into the result tensor
    ///   - indices_tensor: Tensor containing the result indices to insert values at
    ///   - shape: The shape of the result tensor
    ///   - axis: The axis of the result tensor to scatter values along
    ///   - mode: The type of update to use on the destination
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn scatter(&self,
                  updates_tensor:  &MPSGraphTensor,
                  indices_tensor:  &MPSGraphTensor,
                  shape:  &MPSShape,
                  axis:  isize,
                  mode:  MPSGraphScatterMode,
                  name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let result: *mut AnyObject = msg_send![self.0, scatterWithUpdatesTensor: updates_tensor.0,
                indicesTensor: indices_tensor.0,
                shape: shape.0,
                axis: axis,
                mode: mode as i64,
                name: name_obj
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a Scatter operation with a data tensor as base.
    /// 
    /// Scatters the values in updates_tensor to the result tensor along the indices in indices_tensor, on top of data_tensor.
    /// 
    /// - Parameters:
    ///   - data_tensor: Tensor containing initial values of same shape as result tensor
    ///   - updates_tensor: Tensor containing values to be inserted into the result tensor
    ///   - indices_tensor: Tensor containing the result indices to insert values at
    ///   - axis: The axis of the result tensor to scatter values along
    ///   - mode: The type of update to use on the destination
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn scatter_with_data(&self,
                            data_tensor:  &MPSGraphTensor,
                            updates_tensor:  &MPSGraphTensor,
                            indices_tensor:  &MPSGraphTensor,
                            axis:  isize,
                            mode:  MPSGraphScatterMode,
                            name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let result: *mut AnyObject = msg_send![self.0, scatterWithDataTensor: data_tensor.0,
                updatesTensor: updates_tensor.0,
                indicesTensor: indices_tensor.0,
                axis: axis,
                mode: mode as i64,
                name: name_obj
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a ScatterAlongAxis operation.
    /// 
    /// Scatter values from `updates_tensor` along the specified `axis` at indices in `indices_tensor` into a result tensor.
    /// 
    /// - Parameters:
    ///   - axis: The axis to scatter to
    ///   - updates_tensor: The input tensor to scatter values from
    ///   - indices_tensor: Int32 or Int64 tensor used to index the result tensor
    ///   - shape: The shape of the result tensor
    ///   - mode: The type of update to use
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn scatter_along_axis(&self,
                             axis:  isize,
                             updates_tensor:  &MPSGraphTensor,
                             indices_tensor:  &MPSGraphTensor,
                             shape:  &MPSShape,
                             mode:  MPSGraphScatterMode,
                             name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let result: *mut AnyObject = msg_send![self.0, scatterAlongAxis: axis,
                withUpdatesTensor: updates_tensor.0,
                indicesTensor: indices_tensor.0,
                shape: shape.0,
                mode: mode as i64,
                name: name_obj
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a ScatterAlongAxis operation with a data tensor as base.
    /// 
    /// Scatter values from `updates_tensor` along the specified `axis` at indices in `indices_tensor` onto `data_tensor`.
    /// 
    /// - Parameters:
    ///   - axis: The axis to scatter to
    ///   - data_tensor: The input tensor to scatter values onto
    ///   - updates_tensor: The input tensor to scatter values from
    ///   - indices_tensor: Int32 or Int64 tensor used to index the result tensor
    ///   - mode: The type of update to use
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn scatter_along_axis_with_data(&self,
                                       axis:  isize,
                                       data_tensor:  &MPSGraphTensor,
                                       updates_tensor:  &MPSGraphTensor,
                                       indices_tensor:  &MPSGraphTensor,
                                       mode:  MPSGraphScatterMode,
                                       name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let result: *mut AnyObject = msg_send![self.0, scatterAlongAxis: axis,
                withDataTensor: data_tensor.0,
                updatesTensor: updates_tensor.0,
                indicesTensor: indices_tensor.0,
                mode: mode as i64,
                name: name_obj
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
}