use objc::runtime::Object;
use objc::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::operation::MPSGraphOperation;
use crate::core::{NSString, MPSShape, MPSDataType};

/// Memory operations for MPSGraph
impl MPSGraph {
    // The placeholder and constant_scalar methods are already implemented in graph.rs
    
    /// Creates a complex constant with the realPart and imaginaryPart values and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `real_part` - The real part of the complex scalar to fill the entire tensor values with.
    /// * `imaginary_part` - The imaginary part of the complex scalar to fill the entire tensor values with.
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object of type ComplexFloat32.
    pub fn complex_constant(
        &self,
        real_part: f64,
        imaginary_part: f64,
    ) -> MPSGraphTensor {
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                constantWithRealPart:real_part 
                imaginaryPart:imaginary_part
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a complex constant with the specified data type and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `real_part` - The real part of the complex scalar to fill the entire tensor values with.
    /// * `imaginary_part` - The imaginary part of the complex scalar to fill the entire tensor values with.
    /// * `data_type` - The complex data type of the constant tensor.
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object of complex type.
    pub fn complex_constant_with_type(
        &self,
        real_part: f64,
        imaginary_part: f64,
        data_type: MPSDataType,
    ) -> MPSGraphTensor {
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                constantWithRealPart:real_part 
                imaginaryPart:imaginary_part
                dataType:data_type as u64
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a complex constant with shape and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `real_part` - The real part of the complex scalar to fill the entire tensor values with.
    /// * `imaginary_part` - The imaginary part of the complex scalar to fill the entire tensor values with.
    /// * `shape` - The shape of the output tensor.
    /// * `data_type` - The complex data type of the constant tensor.
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object of complex type.
    pub fn complex_constant_with_shape(
        &self,
        real_part: f64,
        imaginary_part: f64,
        shape: &MPSShape,
        data_type: MPSDataType,
    ) -> MPSGraphTensor {
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                constantWithRealPart:real_part 
                imaginaryPart:imaginary_part
                shape:shape.0
                dataType:data_type as u64
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a variable operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `data` - The data for the tensor.
    /// * `shape` - The shape of the output tensor. This has to be statically shaped.
    /// * `data_type` - The dataType of the variable tensor.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object.
    pub fn variable<T: Copy>(
        &self,
        data: &[T],
        shape: &MPSShape,
        data_type: MPSDataType,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            // Create NSData from slice
            let cls = objc::runtime::Class::get("NSData").unwrap();
            let bytes_ptr = data.as_ptr() as *const std::ffi::c_void;
            let bytes_len = data.len() * std::mem::size_of::<T>();
            
            let data_obj: *mut Object = msg_send![
                cls,
                dataWithBytes:bytes_ptr 
                length:bytes_len
            ];
            
            let result: *mut Object = msg_send![
                self.0,
                variableWithData:data_obj
                shape:shape.0
                dataType:data_type as u64
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a variable from an input tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor from which to form the variable.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object representing the variable.
    pub fn variable_from_tensor(
        &self,
        tensor: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                variableFromTensorWithTensor:tensor.0
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a read op which reads at this point of execution of the graph.
    ///
    /// # Arguments
    ///
    /// * `variable` - The variable resource tensor to read from.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object.
    pub fn read_variable(
        &self,
        variable: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                readVariable:variable.0
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates an assign operation which writes at this point of execution of the graph.
    ///
    /// # Arguments
    ///
    /// * `variable` - The variable resource tensor to assign to.
    /// * `tensor` - The tensor to assign to the variable.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid MPSGraphOperation object.
    pub fn assign_variable(
        &self,
        variable: &MPSGraphTensor,
        tensor: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphOperation {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                assignVariable:variable.0
                withValueOfTensor:tensor.0
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphOperation(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::should_skip_test;
    use std::collections::HashMap;
    
    #[test]
    fn test_complex_constant() {
        if should_skip_test("test_complex_constant") {
            return;
        }
        
        let graph = MPSGraph::new();
        
        // Create a complex constant tensor
        let _complex_tensor = graph.complex_constant(
            1.0,  // real part
            2.0,  // imaginary part
        );
        
        // Test succeeds if it doesn't crash
    }
    
    #[test]
    fn test_variable_read() {
        if should_skip_test("test_variable_read") {
            return;
        }
        
        let graph = MPSGraph::new();
        
        // Create a variable tensor
        let shape = MPSShape::from_slice(&[2, 2]);
        let initial_data = vec![1.0f32, 2.0, 3.0, 4.0];
        
        let variable = graph.variable(
            &initial_data, 
            &shape, 
            MPSDataType::Float32, 
            Some("test_variable")
        );
        
        // Read the variable
        let read_var = graph.read_variable(&variable, Some("read_var"));
        
        // Run the graph
        let feeds = HashMap::new();
        let results = graph.run(feeds, &[&read_var]);
        
        // Get the result data - should be the initial data
        let result_data = results[&read_var].to_vec::<f32>();
        
        assert_eq!(result_data.len(), 4);
        assert_eq!(result_data[0], 1.0);
        assert_eq!(result_data[1], 2.0);
        assert_eq!(result_data[2], 3.0);
        assert_eq!(result_data[3], 4.0);
    }
}