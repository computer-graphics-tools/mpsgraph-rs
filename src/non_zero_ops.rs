use objc2::runtime::AnyObject;
use objc2::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, AsRawObject};

/// NonZero operations for MPSGraph
impl MPSGraph {
    /// Computes the indices of the non-zero elements of the input tensor.
    /// 
    /// The indices are returned as a two-dimensional tensor of size `[number_of_nonzeros, input_rank]`.
    /// Each row in the result contains indices of a nonzero elements in input.
    /// For example:
    /// ```
    /// tensor = [[ 1,  0, 3],
    ///           [ 0, 10, 0]]
    /// indices = [[ 0, 0],
    ///            [ 0, 2],
    ///            [ 1, 1]]
    /// ```
    /// 
    /// - Parameters:
    ///   - tensor: An MPSGraphTensor of which to compute the non-zero indices.
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor containing indices in signed int32 data type.
    pub fn non_zero_indices(&self, 
                           tensor:  &MPSGraphTensor,
                           name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let result: *mut AnyObject = msg_send![self.0, nonZeroIndicesOfTensor: tensor.0,
                name: name_obj,
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
}