use crate::core::{AsRawObject, NSString};
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;

/// TopK operations for MPSGraph
impl MPSGraph {
    /// Finds the k largest values along the minor dimension of the input.
    ///
    /// The source must have at least k elements along its minor dimension.
    /// Returns a tuple of tensors: (values, indices) where:
    /// - values: The top k values
    /// - indices: The indices of those values
    ///
    /// - Parameters:
    ///   - source: Tensor containing source data
    ///   - k: The number of largest values to return
    ///   - name: The name for the operation
    /// - Returns: A tuple (values, indices) of MPSGraphTensor objects
    pub fn top_k(
        &self,
        source: &MPSGraphTensor,
        k: usize,
        name: Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result_array: *mut AnyObject = msg_send![self.0, topKWithSourceTensor: source.0,
                k: k,
                name: name_obj,
            ];

            // Get the two tensors from the NSArray
            let values: *mut AnyObject = msg_send![result_array, objectAtIndex: 0];
            let indices: *mut AnyObject = msg_send![result_array, objectAtIndex: 1];

            // Retain the tensors
            let values = objc2::ffi::objc_retain(values as *mut _);
            let indices = objc2::ffi::objc_retain(indices as *mut _);

            (MPSGraphTensor(values), MPSGraphTensor(indices))
        }
    }

    /// Finds the k largest values along the specified axis of the input.
    ///
    /// The source must have at least k elements along the specified axis.
    ///
    /// - Parameters:
    ///   - source: Tensor containing source data
    ///   - axis: The dimension along which to compute the TopK values
    ///   - k: The number of largest values to return
    ///   - name: The name for the operation
    /// - Returns: A tuple (values, indices) of MPSGraphTensor objects
    pub fn top_k_axis(
        &self,
        source: &MPSGraphTensor,
        axis: isize,
        k: usize,
        name: Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result_array: *mut AnyObject = msg_send![self.0, topKWithSourceTensor: source.0,
                axis: axis,
                k: k,
                name: name_obj,
            ];

            // Get the two tensors from the NSArray
            let values: *mut AnyObject = msg_send![result_array, objectAtIndex: 0];
            let indices: *mut AnyObject = msg_send![result_array, objectAtIndex: 1];

            // Retain the tensors
            let values = objc2::ffi::objc_retain(values as *mut _);
            let indices = objc2::ffi::objc_retain(indices as *mut _);

            (MPSGraphTensor(values), MPSGraphTensor(indices))
        }
    }

    /// Finds the k smallest values along the specified axis of the input.
    ///
    /// The source must have at least k elements along the specified axis.
    ///
    /// - Parameters:
    ///   - source: Tensor containing source data
    ///   - axis: The dimension along which to compute the BottomK values
    ///   - k: The number of smallest values to return
    ///   - name: The name for the operation
    /// - Returns: A tuple (values, indices) of MPSGraphTensor objects
    pub fn bottom_k_axis(
        &self,
        source: &MPSGraphTensor,
        axis: isize,
        k: usize,
        name: Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result_array: *mut AnyObject = msg_send![self.0, bottomKWithSourceTensor: source.0,
                axis: axis,
                k: k,
                name: name_obj,
            ];

            // Get the two tensors from the NSArray
            let values: *mut AnyObject = msg_send![result_array, objectAtIndex: 0];
            let indices: *mut AnyObject = msg_send![result_array, objectAtIndex: 1];

            // Retain the tensors
            let values = objc2::ffi::objc_retain(values as *mut _);
            let indices = objc2::ffi::objc_retain(indices as *mut _);

            (MPSGraphTensor(values), MPSGraphTensor(indices))
        }
    }

    /// Finds the k largest values using tensors for parameters.
    ///
    /// - Parameters:
    ///   - source: Tensor containing source data
    ///   - k_tensor: Tensor containing the value of k
    ///   - name: The name for the operation
    /// - Returns: A tuple (values, indices) of MPSGraphTensor objects
    pub fn top_k_with_tensor(
        &self,
        source: &MPSGraphTensor,
        k_tensor: &MPSGraphTensor,
        name: Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result_array: *mut AnyObject = msg_send![self.0, topKWithSourceTensor: source.0,
                kTensor: k_tensor.0,
                name: name_obj,
            ];

            // Get the two tensors from the NSArray
            let values: *mut AnyObject = msg_send![result_array, objectAtIndex: 0];
            let indices: *mut AnyObject = msg_send![result_array, objectAtIndex: 1];

            // Retain the tensors
            let values = objc2::ffi::objc_retain(values as *mut _);
            let indices = objc2::ffi::objc_retain(indices as *mut _);

            (MPSGraphTensor(values), MPSGraphTensor(indices))
        }
    }

    /// Finds the k largest values using tensors for axis and k parameters.
    ///
    /// - Parameters:
    ///   - source: Tensor containing source data
    ///   - axis_tensor: Tensor containing the axis along which to compute TopK
    ///   - k_tensor: Tensor containing the value of k
    ///   - name: The name for the operation
    /// - Returns: A tuple (values, indices) of MPSGraphTensor objects
    pub fn top_k_with_axis_tensor(
        &self,
        source: &MPSGraphTensor,
        axis_tensor: &MPSGraphTensor,
        k_tensor: &MPSGraphTensor,
        name: Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result_array: *mut AnyObject = msg_send![self.0, topKWithSourceTensor: source.0,
                axisTensor: axis_tensor.0,
                kTensor: k_tensor.0,
                name: name_obj,
            ];

            // Get the two tensors from the NSArray
            let values: *mut AnyObject = msg_send![result_array, objectAtIndex: 0];
            let indices: *mut AnyObject = msg_send![result_array, objectAtIndex: 1];

            // Retain the tensors
            let values = objc2::ffi::objc_retain(values as *mut _);
            let indices = objc2::ffi::objc_retain(indices as *mut _);

            (MPSGraphTensor(values), MPSGraphTensor(indices))
        }
    }

    /// Finds the k smallest values using tensors for axis and k parameters.
    ///
    /// - Parameters:
    ///   - source: Tensor containing source data
    ///   - axis_tensor: Tensor containing the axis along which to compute BottomK
    ///   - k_tensor: Tensor containing the value of k
    ///   - name: The name for the operation
    /// - Returns: A tuple (values, indices) of MPSGraphTensor objects
    pub fn bottom_k_with_axis_tensor(
        &self,
        source: &MPSGraphTensor,
        axis_tensor: &MPSGraphTensor,
        k_tensor: &MPSGraphTensor,
        name: Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result_array: *mut AnyObject = msg_send![self.0, bottomKWithSourceTensor: source.0,
                axisTensor: axis_tensor.0,
                kTensor: k_tensor.0,
                name: name_obj,
            ];

            // Get the two tensors from the NSArray
            let values: *mut AnyObject = msg_send![result_array, objectAtIndex: 0];
            let indices: *mut AnyObject = msg_send![result_array, objectAtIndex: 1];

            // Retain the tensors
            let values = objc2::ffi::objc_retain(values as *mut _);
            let indices = objc2::ffi::objc_retain(indices as *mut _);

            (MPSGraphTensor(values), MPSGraphTensor(indices))
        }
    }

    /// Computes the gradient for a TopK operation.
    ///
    /// - Parameters:
    ///   - gradient: Tensor containing the incoming gradient
    ///   - source: Tensor containing source data
    ///   - k: The number of largest values used in the forward pass
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn top_k_gradient(
        &self,
        gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        k: usize,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0, topKWithGradientTensor: gradient.0,
                source: source.0,
                k: k,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            MPSGraphTensor(result)
        }
    }

    /// Computes the gradient for a TopK operation with specified axis.
    ///
    /// - Parameters:
    ///   - gradient: Tensor containing the incoming gradient
    ///   - source: Tensor containing source data
    ///   - axis: The dimension along which TopK was computed
    ///   - k: The number of largest values used in the forward pass
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn top_k_gradient_axis(
        &self,
        gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        axis: isize,
        k: usize,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0, topKWithGradientTensor: gradient.0,
                source: source.0,
                axis: axis,
                k: k,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            MPSGraphTensor(result)
        }
    }

    /// Computes the gradient for a BottomK operation with specified axis.
    ///
    /// - Parameters:
    ///   - gradient: Tensor containing the incoming gradient
    ///   - source: Tensor containing source data
    ///   - axis: The dimension along which BottomK was computed
    ///   - k: The number of smallest values used in the forward pass
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn bottom_k_gradient_axis(
        &self,
        gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        axis: isize,
        k: usize,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0, bottomKWithGradientTensor: gradient.0,
                source: source.0,
                axis: axis,
                k: k,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            MPSGraphTensor(result)
        }
    }
}
