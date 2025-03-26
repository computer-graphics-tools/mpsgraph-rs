use objc::runtime::Object;
use objc::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::NSString;

/// Call operations for MPSGraph
impl MPSGraph {
    /// Creates an operation which invokes another executable.
    ///
    /// # Arguments
    ///
    /// * `symbol_name` - The unique identifier used to find the executable in the MPSGraphCompilationDescriptor.callables directory
    /// * `input_tensors` - The tensors which are passed as inputs to the executable being invoked
    /// * `output_types` - The expected return types of the executable being invoked
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// An array of MPSGraphTensor objects representing the return tensors of the invoked executable
    pub fn call(
        &self,
        symbol_name: &str,
        input_tensors: &[&MPSGraphTensor],
        output_types: &[*mut Object], // MPSGraphType objects
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        let symbol_name_obj = NSString::from_str(symbol_name).0;
        
        // Create NSArray of input tensors
        let input_tensors_array = unsafe {
            let cls = objc::runtime::Class::get("NSArray").unwrap();
            let mut ns_objects: Vec<*mut Object> = Vec::with_capacity(input_tensors.len());
            
            for tensor in input_tensors {
                ns_objects.push(tensor.0);
            }
            
            let count = ns_objects.len();
            let ns_array: *mut Object = msg_send![
                cls,
                arrayWithObjects:ns_objects.as_ptr()
                count:count
            ];
            ns_array
        };
        
        // Create NSArray of output types
        let output_types_array = unsafe {
            let cls = objc::runtime::Class::get("NSArray").unwrap();
            let count = output_types.len();
            let ns_array: *mut Object = msg_send![
                cls,
                arrayWithObjects:output_types.as_ptr()
                count:count
            ];
            ns_array
        };
        
        // Call the Objective-C method and get the result array
        let result_array = unsafe {
            let result: *mut Object = msg_send![
                self.0,
                callSymbolName:symbol_name_obj
                inputTensors:input_tensors_array
                outputTypes:output_types_array
                name:name_obj
            ];
            result
        };
        
        // Convert the result array to a Vec of MPSGraphTensor
        let count = unsafe {
            let count: usize = msg_send![result_array, count];
            count
        };
        
        let mut results = Vec::with_capacity(count);
        
        for i in 0..count {
            unsafe {
                let tensor: *mut Object = msg_send![result_array, objectAtIndex:i];
                let tensor: *mut Object = msg_send![tensor, retain];
                results.push(MPSGraphTensor(tensor));
            }
        }
        
        results
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::should_skip_test;
    
    // We can't easily test the call operation without creating a proper compilation descriptor
    // with callables, so we'll just stub the test to make sure the code compiles
    #[test]
    fn test_call_stub() {
        if should_skip_test("test_call_stub") {
            return;
        }
        
        // This is just a compile-time check since we can't easily test the call operation
    }
}