use objc::runtime::Object;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::NSString;

/// Gather operations for MPSGraph
impl MPSGraph {
    /// Creates a GatherND operation and returns the result tensor.
    ///
    /// Gathers the slices in updatesTensor to the result tensor along the indices in indicesTensor. 
    ///
    /// - Parameters:
    ///   - updates_tensor: Tensor containing slices to be inserted into the result tensor.
    ///   - indices_tensor: Tensor containing the updates indices to read slices from
    ///   - batch_dimensions: The number of batch dimensions
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object 
    pub fn gather_nd(&self, 
                     updates_tensor: &MPSGraphTensor,
                     indices_tensor: &MPSGraphTensor,
                     batch_dimensions: usize,
                     name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut Object = msg_send![self.0, 
                gatherNDWithUpdatesTensor:updates_tensor.0
                indicesTensor:indices_tensor.0
                batchDimensions:batch_dimensions
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a Gather operation and returns the result tensor.
    ///
    /// Gathers the values in updatesTensor to the result tensor along the indices in indicesTensor. 
    ///
    /// - Parameters:
    ///   - updates_tensor: Tensor containing slices to be inserted into the result tensor.
    ///   - indices_tensor: Tensor containing the updates indices to read slices from
    ///   - axis: The dimension on which to perform the gather
    ///   - batch_dimensions: The number of batch dimensions
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object 
    pub fn gather(&self, 
                  updates_tensor: &MPSGraphTensor,
                  indices_tensor: &MPSGraphTensor,
                  axis: usize,
                  batch_dimensions: usize,
                  name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut Object = msg_send![self.0, 
                gatherWithUpdatesTensor:updates_tensor.0
                indicesTensor:indices_tensor.0
                axis:axis
                batchDimensions:batch_dimensions
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a GatherAlongAxis operation and returns the result tensor.
    ///
    /// Gather values from `updates_tensor` along the specified `axis` at indices in `indices_tensor`. 
    /// The shape of `updates_tensor` and `indices_tensor` must match except at `axis`. 
    /// The shape of the result tensor is equal to the shape of `indices_tensor`. 
    /// If an index is out of bounds of the `updates_tensor` along `axis` a 0 is inserted.
    ///
    /// - Parameters:
    ///   - axis: The axis to gather from. Negative values wrap around
    ///   - updates_tensor: The input tensor to gather values from
    ///   - indices_tensor: Int32 or Int64 tensor used to index `updates_tensor`
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object 
    pub fn gather_along_axis(&self,
                            axis: isize,
                            updates_tensor: &MPSGraphTensor,
                            indices_tensor: &MPSGraphTensor,
                            name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut Object = msg_send![self.0, 
                gatherAlongAxis:axis
                withUpdatesTensor:updates_tensor.0
                indicesTensor:indices_tensor.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a GatherAlongAxis operation using an axis tensor and returns the result tensor.
    ///
    /// Gather values from `updates_tensor` along the specified `axis_tensor` at indices in `indices_tensor`.
    /// The shape of `updates_tensor` and `indices_tensor` must match except at the axis specified.
    /// The shape of the result tensor is equal to the shape of `indices_tensor`.
    /// If an index is out of bounds of the `updates_tensor` along axis a 0 is inserted.
    /// 
    /// - Parameters:
    ///   - axis_tensor: Scalar Int32 tensor. The axis to gather from. Negative values wrap around
    ///   - updates_tensor: The input tensor to gather values from
    ///   - indices_tensor: Int32 or Int64 tensor used to index `updates_tensor`
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object 
    pub fn gather_along_axis_tensor(&self,
                                  axis_tensor: &MPSGraphTensor,
                                  updates_tensor: &MPSGraphTensor,
                                  indices_tensor: &MPSGraphTensor,
                                  name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut Object = msg_send![self.0, 
                gatherAlongAxisTensor:axis_tensor.0
                withUpdatesTensor:updates_tensor.0
                indicesTensor:indices_tensor.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
}