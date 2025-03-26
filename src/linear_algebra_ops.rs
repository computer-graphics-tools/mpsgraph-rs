use crate::{MPSGraph, MPSGraphTensor, core::NSString};
use objc::msg_send;
use std::os::raw::c_int;

impl MPSGraph {
    /// Computes the band part of an input tensor.
    ///
    /// This operation copies a diagonal band of values from input tensor to a result tensor of the same size.
    /// A coordinate `[..., i, j]` is in the band if 
    /// ```
    /// (num_lower < 0 || (i-j) <= num_lower) && (num_upper < 0 || (j-i) <= num_upper)
    /// ```
    /// The values outside of the band are set to 0.
    ///
    /// # Arguments
    ///
    /// * `input_tensor` - The source tensor to copy
    /// * `num_lower` - The number of diagonals in the lower triangle to keep. If -1, keep all sub-diagonals
    /// * `num_upper` - The number of diagonals in the upper triangle to keep. If -1, keep all super-diagonals
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new `MPSGraphTensor` containing the result
    pub fn band_part(
        &self,
        input_tensor: &MPSGraphTensor,
        num_lower: isize,
        num_upper: isize,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };

        let result = unsafe {
            let raw_tensor: *mut objc::runtime::Object = msg_send![
                self.0,
                bandPartWithTensor:input_tensor.0
                numLower:num_lower as c_int
                numUpper:num_upper as c_int
                name:name_obj
            ];
            MPSGraphTensor(raw_tensor)
        };

        result
    }

    /// Computes the band part of an input tensor with parameters specified as tensors.
    ///
    /// # Arguments
    ///
    /// * `input_tensor` - The source tensor to copy
    /// * `num_lower_tensor` - Scalar Int32 tensor. The number of diagonals in the lower triangle to keep. If -1, keep all
    /// * `num_upper_tensor` - Scalar Int32 tensor. The number of diagonals in the upper triangle to keep. If -1, keep all
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new `MPSGraphTensor` containing the result
    pub fn band_part_with_tensors(
        &self,
        input_tensor: &MPSGraphTensor,
        num_lower_tensor: &MPSGraphTensor,
        num_upper_tensor: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };

        let result = unsafe {
            let raw_tensor: *mut objc::runtime::Object = msg_send![
                self.0,
                bandPartWithTensor:input_tensor.0
                numLowerTensor:num_lower_tensor.0
                numUpperTensor:num_upper_tensor.0
                name:name_obj
            ];
            MPSGraphTensor(raw_tensor)
        };

        result
    }

    /// Computes the hamming distance of two input tensors with support for broadcasting.
    ///
    /// The hamming distance is computed between 2 sets of vectors and the last dimension(s) of each 
    /// input tensor is considered a vector.
    ///
    /// # Arguments
    ///
    /// * `primary_tensor` - The first input tensor
    /// * `secondary_tensor` - The second input tensor
    /// * `result_data_type` - The datatype of the return tensor. Must be either `MPSDataType::UInt32` or `MPSDataType::UInt16`
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new `MPSGraphTensor` containing the hamming distance between the input tensors
    pub fn hamming_distance(
        &self,
        primary_tensor: &MPSGraphTensor,
        secondary_tensor: &MPSGraphTensor,
        result_data_type: crate::MPSDataType,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };

        let result = unsafe {
            let raw_tensor: *mut objc::runtime::Object = msg_send![
                self.0,
                HammingDistanceWithPrimaryTensor:primary_tensor.0
                secondaryTensor:secondary_tensor.0
                resultDataType:result_data_type as u64
                name:name_obj
            ];
            MPSGraphTensor(raw_tensor)
        };

        result
    }

    /// Creates a scaled dot product attention (SDPA) operation and returns the result tensor.
    ///
    /// SDPA Op computes attention by computing softmax(scale * QK^T + M)V.
    ///
    /// # Arguments
    ///
    /// * `query_tensor` - A tensor that represents the query projection
    /// * `key_tensor` - A tensor that represents the key projection
    /// * `value_tensor` - A tensor that represents the value projection
    /// * `mask_tensor` - An optional tensor that contains a mask applied to the scaled matrix multiplied query and value matrices
    /// * `scale` - A scale that is applied to the result of query and value matrix multiply
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new `MPSGraphTensor` containing the result of the attention operation
    pub fn scaled_dot_product_attention(
        &self,
        query_tensor: &MPSGraphTensor,
        key_tensor: &MPSGraphTensor,
        value_tensor: &MPSGraphTensor,
        mask_tensor: Option<&MPSGraphTensor>,
        scale: f32,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };

        let mask_tensor_ptr = match mask_tensor {
            Some(tensor) => tensor.0,
            None => std::ptr::null_mut(),
        };

        let result = unsafe {
            let raw_tensor: *mut objc::runtime::Object = msg_send![
                self.0,
                scaledDotProductAttentionWithQueryTensor:query_tensor.0
                keyTensor:key_tensor.0
                valueTensor:value_tensor.0
                maskTensor:mask_tensor_ptr
                scale:scale
                name:name_obj
            ];
            MPSGraphTensor(raw_tensor)
        };

        result
    }

    /// Computes the inverse of an input tensor.
    ///
    /// The framework computes the inverse of a square matrix by calling LU decomposition and LU solver.
    /// All dimensions after the first 2 are treated as batch dimensions and the inverse for each batch is computed.
    /// Results are undefined for ill-conditioned matrices.
    ///
    /// # Arguments
    ///
    /// * `input_tensor` - The input tensor containing square matrices to invert
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new `MPSGraphTensor` containing the inverse of the input tensor
    pub fn matrix_inverse(
        &self,
        input_tensor: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };

        let result = unsafe {
            let raw_tensor: *mut objc::runtime::Object = msg_send![
                self.0,
                inverseOfTensor:input_tensor.0
                name:name_obj
            ];
            MPSGraphTensor(raw_tensor)
        };

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MPSGraphTensorData, MPSDataType, MPSShape};
    use std::collections::HashMap;
    
    #[test]
    fn test_band_part() {
        use crate::tests::should_skip_test;
        if should_skip_test("test_band_part") {
            return;
        }
        
        let graph = MPSGraph::new();
        
        // Create a 3x3 input tensor with values 1-9
        let shape = MPSShape::from_slice(&[3, 3]);
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let input_tensor = graph.placeholder(&shape, MPSDataType::Float32, None);
        
        // Test with num_lower=1, num_upper=1 (only main diagonal and first super/sub diagonal)
        let result_tensor = graph.band_part(&input_tensor, 1, 1, Some("band_part"));
        
        // Create feed dictionary
        let mut feeds = HashMap::new();
        feeds.insert(
            &input_tensor, 
            MPSGraphTensorData::new(&input_data, &[3, 3], MPSDataType::Float32)
        );
        
        // Execute the graph
        let results = graph.run(feeds, &[&result_tensor]);
        
        // Get the data from the result tensor
        let result_data = results[&result_tensor].to_vec::<f32>();
        
        // Expected result: a 3x3 tensor with only the main diagonal, first super-diagonal, 
        // and first sub-diagonal preserved; all other values set to 0
        let expected = vec![1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0];
        assert_eq!(result_data, expected);
    }

    #[test]
    fn test_matrix_inverse() {
        use crate::tests::should_skip_test;
        if should_skip_test("test_matrix_inverse") {
            return;
        }
        
        let graph = MPSGraph::new();
        
        // Create a 2x2 input tensor representing a simple invertible matrix
        let shape = MPSShape::from_slice(&[2, 2]);
        let input_data = vec![4.0f32, 7.0, 2.0, 6.0]; // [[4, 7], [2, 6]]
        let input_tensor = graph.placeholder(&shape, MPSDataType::Float32, None);
        
        // Get the inverse of the matrix
        let result_tensor = graph.matrix_inverse(&input_tensor, Some("inverse"));
        
        // Create feed dictionary
        let mut feeds = HashMap::new();
        feeds.insert(
            &input_tensor, 
            MPSGraphTensorData::new(&input_data, &[2, 2], MPSDataType::Float32)
        );
        
        // Execute the graph
        let results = graph.run(feeds, &[&result_tensor]);
        
        // Get the data from the result tensor
        let result_data = results[&result_tensor].to_vec::<f32>();
        
        // Expected inverse of [[4, 7], [2, 6]] is [[0.6, -0.7], [-0.2, 0.4]]
        // with small floating point differences
        assert!((result_data[0] - 0.6).abs() < 1e-5);
        assert!((result_data[1] - (-0.7)).abs() < 1e-5);
        assert!((result_data[2] - (-0.2)).abs() < 1e-5);
        assert!((result_data[3] - 0.4).abs() < 1e-5);
    }
}