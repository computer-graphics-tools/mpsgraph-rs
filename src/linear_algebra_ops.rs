use objc2::runtime::AnyObject;
use objc2::msg_send;
use crate::tensor::MPSGraphTensor;
use crate::graph::MPSGraph;
use crate::core::{NSString, AsRawObject};

/// Linear algebra operations for MPSGraph
impl MPSGraph {
    /// Creates a matrix multiplication operation.
    ///
    /// # Arguments
    ///
    /// * `primary` - First tensor input
    /// * `secondary` - Second tensor input
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object.
    pub fn matmul(
        &self,
        primary: &MPSGraphTensor,
        secondary: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, matrixMultiplicationWithPrimaryTensor: primary.0,
                secondaryTensor: secondary.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a matrix multiplication operation with transposed operands.
    ///
    /// # Arguments
    ///
    /// * `primary` - First tensor input
    /// * `primary_transpose` - Whether to transpose the first tensor
    /// * `secondary` - Second tensor input
    /// * `secondary_transpose` - Whether to transpose the second tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object.
    pub fn matmul_with_transpose(
        &self,
        primary: &MPSGraphTensor,
        primary_transpose: bool,
        secondary: &MPSGraphTensor,
        secondary_transpose: bool,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, matrixMultiplicationWithPrimaryTensor: primary.0,
                transposePrimary: primary_transpose,
                secondaryTensor: secondary.0,
                transposeSecondary: secondary_transpose,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a vector inner product operation.
    ///
    /// # Arguments
    ///
    /// * `primary` - First vector tensor
    /// * `secondary` - Second vector tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object.
    pub fn inner_product(
        &self,
        primary: &MPSGraphTensor,
        secondary: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, innerProductWithPrimaryTensor: primary.0,
                secondaryTensor: secondary.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a vector outer product operation.
    ///
    /// # Arguments
    ///
    /// * `primary` - First vector tensor
    /// * `secondary` - Second vector tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object.
    pub fn outer_product(
        &self,
        primary: &MPSGraphTensor,
        secondary: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, outerProductWithPrimaryTensor: primary.0,
                secondaryTensor: secondary.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a batch matrix multiplication operation.
    ///
    /// # Arguments
    ///
    /// * `primary` - First tensor input
    /// * `secondary` - Second tensor input
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object.
    pub fn batch_matmul(
        &self,
        primary: &MPSGraphTensor,
        secondary: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, batchMatrixMultiplicationWithPrimaryTensor: primary.0,
                secondaryTensor: secondary.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a batch matrix multiplication operation with transposed operands.
    ///
    /// # Arguments
    ///
    /// * `primary` - First tensor input
    /// * `primary_transpose` - Whether to transpose the first tensor
    /// * `secondary` - Second tensor input
    /// * `secondary_transpose` - Whether to transpose the second tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object.
    pub fn batch_matmul_with_transpose(
        &self,
        primary: &MPSGraphTensor,
        primary_transpose: bool,
        secondary: &MPSGraphTensor,
        secondary_transpose: bool,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, 
                batchMatrixMultiplicationWithPrimaryTensor: primary.0,
                transposePrimary: primary_transpose,
                secondaryTensor: secondary.0,
                transposeSecondary: secondary_transpose,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}