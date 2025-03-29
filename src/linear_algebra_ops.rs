use crate::core::{AsRawObject, NSString};
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;

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

    /// Creates a tensor with a band part extracted from the input tensor.
    ///
    /// This operation extracts a band of rows and columns from the input 2D tensor,
    /// where values outside the band are set to zero. The lower and upper diagonals
    /// specify the distance below and above the main diagonal that should be retained.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor (at least rank 2)
    /// * `num_lower` - The lower diagonal index value. If negative, retain the entire lower triangle.
    /// * `num_upper` - The upper diagonal index value. If negative, retain the entire upper triangle.
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object with the band part extracted.
    pub fn band_part(
        &self,
        input: &MPSGraphTensor,
        num_lower: &MPSGraphTensor,
        num_upper: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                bandPartWithTensor: input.0,
                numLower: num_lower.0,
                numUpper: num_upper.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a tensor with a band part extracted from the input tensor using scalar values.
    ///
    /// This operation extracts a band of rows and columns from the input 2D tensor,
    /// where values outside the band are set to zero. The lower and upper diagonals
    /// specify the distance below and above the main diagonal that should be retained.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor (at least rank 2)
    /// * `num_lower` - The lower diagonal scalar value. If negative, retain the entire lower triangle.
    /// * `num_upper` - The upper diagonal scalar value. If negative, retain the entire upper triangle.
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object with the band part extracted.
    pub fn band_part_with_scalars(
        &self,
        input: &MPSGraphTensor,
        num_lower: i64,
        num_upper: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                bandPartWithTensor: input.0,
                numLowerScalar: num_lower,
                numUpperScalar: num_upper,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Calculates the Hamming distance between two tensors.
    ///
    /// The Hamming distance between two tensors is the number of positions at which the corresponding elements
    /// are different. This operation computes the Hamming distance along the innermost dimension.
    ///
    /// # Arguments
    ///
    /// * `primary` - The first input tensor
    /// * `secondary` - The second input tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor with the Hamming distance between the inputs.
    pub fn hamming_distance(
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
                self.0,
                hammingDistanceWithPrimaryTensor: primary.0,
                secondaryTensor: secondary.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs scaled dot-product attention on the input tensors.
    ///
    /// Scaled dot-product attention is a key operation in transformer architectures. This operation
    /// computes attention weights by scaling the dot product of query and key, then applying softmax,
    /// and finally multiplying with value.
    ///
    /// The formula is: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    ///
    /// # Arguments
    ///
    /// * `query` - The query tensor
    /// * `key` - The key tensor
    /// * `value` - The value tensor
    /// * `scale` - The scaling factor tensor (usually 1/sqrt(d_k))
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the attention operation
    pub fn scaled_dot_product_attention(
        &self,
        query: &MPSGraphTensor,
        key: &MPSGraphTensor,
        value: &MPSGraphTensor,
        scale: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                scaledDotProductAttentionWithQueryTensor: query.0,
                keyTensor: key.0,
                valueTensor: value.0,
                scaleTensor: scale.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs scaled dot-product attention with a scalar scaling factor.
    ///
    /// This is a variant of scaled dot-product attention where the scaling factor is a scalar value
    /// instead of a tensor.
    ///
    /// # Arguments
    ///
    /// * `query` - The query tensor
    /// * `key` - The key tensor
    /// * `value` - The value tensor
    /// * `scale` - The scalar scaling factor (usually 1/sqrt(d_k))
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the attention operation
    pub fn scaled_dot_product_attention_with_scalar(
        &self,
        query: &MPSGraphTensor,
        key: &MPSGraphTensor,
        value: &MPSGraphTensor,
        scale: f32,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                scaledDotProductAttentionWithQueryTensor: query.0,
                keyTensor: key.0,
                valueTensor: value.0,
                scaleScalar: scale as f64,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs masked scaled dot-product attention.
    ///
    /// This is a variant of scaled dot-product attention where certain attention weights
    /// can be masked out (e.g., for causal attention).
    ///
    /// # Arguments
    ///
    /// * `query` - The query tensor
    /// * `key` - The key tensor
    /// * `value` - The value tensor
    /// * `mask` - The mask tensor (often used for causal attention)
    /// * `scale` - The scaling factor tensor (usually 1/sqrt(d_k))
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the attention operation
    pub fn masked_scaled_dot_product_attention(
        &self,
        query: &MPSGraphTensor,
        key: &MPSGraphTensor,
        value: &MPSGraphTensor,
        mask: &MPSGraphTensor,
        scale: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                scaledDotProductAttentionWithQueryTensor: query.0,
                keyTensor: key.0,
                valueTensor: value.0,
                maskTensor: mask.0,
                scaleTensor: scale.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs masked scaled dot-product attention with a scalar scaling factor.
    ///
    /// This is a variant of masked scaled dot-product attention where the scaling factor is a scalar value
    /// instead of a tensor.
    ///
    /// # Arguments
    ///
    /// * `query` - The query tensor
    /// * `key` - The key tensor
    /// * `value` - The value tensor
    /// * `mask` - The mask tensor (often used for causal attention)
    /// * `scale` - The scalar scaling factor (usually 1/sqrt(d_k))
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the attention operation
    pub fn masked_scaled_dot_product_attention_with_scalar(
        &self,
        query: &MPSGraphTensor,
        key: &MPSGraphTensor,
        value: &MPSGraphTensor,
        mask: &MPSGraphTensor,
        scale: f32,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                scaledDotProductAttentionWithQueryTensor: query.0,
                keyTensor: key.0,
                valueTensor: value.0,
                maskTensor: mask.0,
                scaleScalar: scale as f64,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}
