use objc2::runtime::AnyObject;
// In objc2, use false as NO and true as YES
const NO: bool = false;
const YES: bool = true;
use objc2::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, AsRawObject};

/// Sort operations for MPSGraph
impl MPSGraph {
    /// Sorts the elements of the input tensor along the specified axis.
    /// 
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis: The tensor dimension over which to sort the tensor
    ///   - descending: If true, reverse the sort direction
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn sort(&self, 
                tensor:  &MPSGraphTensor,
                axis:  isize,
                descending:  bool,
                name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let descending_val = if descending { YES } else { NO };
            
            let result: *mut AnyObject = msg_send![self.0, sortWithTensor: tensor.0,
                axis: axis,
                descending: descending_val,
                name: name_obj,
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
    
    /// Sorts the elements of the input tensor along the specified axis (in ascending order).
    /// 
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis: The tensor dimension over which to sort the tensor
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn sort_ascending(&self, 
                          tensor:  &MPSGraphTensor,
                          axis:  isize,
                          name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let result: *mut AnyObject = msg_send![self.0, sortWithTensor: tensor.0,
                axis: axis,
                name: name_obj,
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
    
    /// Sorts the elements of the input tensor along an axis specified by a tensor.
    /// 
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis_tensor: A scalar tensor that specifies the dimension over which to sort
    ///   - descending: If true, reverse the sort direction
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn sort_with_axis_tensor(&self, 
                                 tensor:  &MPSGraphTensor,
                                 axis_tensor:  &MPSGraphTensor,
                                 descending:  bool,
                                 name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let descending_val = if descending { YES } else { NO };
            
            let result: *mut AnyObject = msg_send![self.0, sortWithTensor: tensor.0,
                axisTensor: axis_tensor.0,
                descending: descending_val,
                name: name_obj,
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
    
    /// Sorts the elements of the input tensor along an axis specified by a tensor (in ascending order).
    /// 
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis_tensor: A scalar tensor that specifies the dimension over which to sort
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn sort_ascending_with_axis_tensor(&self, 
                                          tensor:  &MPSGraphTensor,
                                          axis_tensor:  &MPSGraphTensor,
                                          name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let result: *mut AnyObject = msg_send![self.0, sortWithTensor: tensor.0,
                axisTensor: axis_tensor.0,
                name: name_obj,
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
    
    /// Computes the indices that sort the elements of the input tensor along the specified axis.
    /// 
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis: The tensor dimension over which to sort the tensor
    ///   - descending: If true, reverse the sort direction
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object with 32-bit integer data type
    pub fn arg_sort(&self, 
                   tensor:  &MPSGraphTensor,
                   axis:  isize,
                   descending:  bool,
                   name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let descending_val = if descending { YES } else { NO };
            
            let result: *mut AnyObject = msg_send![self.0, argSortWithTensor: tensor.0,
                axis: axis,
                descending: descending_val,
                name: name_obj,
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
    
    /// Computes the indices that sort the elements of the input tensor along the specified axis (in ascending order).
    /// 
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis: The tensor dimension over which to sort the tensor
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object with 32-bit integer data type
    pub fn arg_sort_ascending(&self, 
                             tensor:  &MPSGraphTensor,
                             axis:  isize,
                             name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let result: *mut AnyObject = msg_send![self.0, argSortWithTensor: tensor.0,
                axis: axis,
                name: name_obj,
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
    
    /// Computes the indices that sort the elements of the input tensor along an axis specified by a tensor.
    /// 
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis_tensor: A scalar tensor that specifies the dimension over which to sort
    ///   - descending: If true, reverse the sort direction
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object with 32-bit integer data type
    pub fn arg_sort_with_axis_tensor(&self, 
                                    tensor:  &MPSGraphTensor,
                                    axis_tensor:  &MPSGraphTensor,
                                    descending:  bool,
                                    name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let descending_val = if descending { YES } else { NO };
            
            let result: *mut AnyObject = msg_send![self.0, argSortWithTensor: tensor.0,
                axisTensor: axis_tensor.0,
                descending: descending_val,
                name: name_obj,
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
    
    /// Computes the indices that sort the elements of the input tensor along an axis specified by a tensor (in ascending order).
    /// 
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis_tensor: A scalar tensor that specifies the dimension over which to sort
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object with 32-bit integer data type
    pub fn arg_sort_ascending_with_axis_tensor(&self, 
                                              tensor:  &MPSGraphTensor,
                                              axis_tensor:  &MPSGraphTensor,
                                              name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let result: *mut AnyObject = msg_send![self.0, argSortWithTensor: tensor.0,
                axisTensor: axis_tensor.0,
                name: name_obj,
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
}