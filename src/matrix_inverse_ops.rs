use objc2::runtime::AnyObject;
use objc2::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use objc2_foundation::NSString;
use crate::core::AsRawObject;
use std::ptr;

/// Matrix inverse operations for MPSGraph
impl MPSGraph {
    /// Creates a matrix inverse operation
    ///
    /// This function computes the matrix inverse of a tensor.
    /// The input tensor must be a square matrix or a batch of square matrices.
    /// If the input is a batch of matrices, the inverse operation is applied to each matrix independently.
    ///
    /// # Parameters
    /// - `x`: The input tensor containing matrix/matrices to invert
    /// - `name`: Optional name for the operation
    ///
    /// # Returns
    /// A new tensor containing the inverted matrix/matrices
    pub fn inverse(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, inverseOfTensor: x.0, name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}