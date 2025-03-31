use crate::core::AsRawObject;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use objc2_foundation::{NSArray, NSString};

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
        output_types: &[&crate::data_types::MPSGraphShapedType],
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        let symbol_name_obj = NSString::from_str(symbol_name).as_raw_object();

        // Create NSArray of input tensors using objc2_foundation
        let input_tensors_array = unsafe {
            // Convert to slice of references to AnyObject
            let refs: Vec<&objc2::runtime::AnyObject> = input_tensors
                .iter()
                .map(|tensor| &*tensor.0.cast::<objc2::runtime::AnyObject>())
                .collect();

            // Create NSArray from references
            let array = NSArray::from_slice(&refs);
            let ns_array: *mut AnyObject =
                array.as_ref() as *const objc2_foundation::NSArray as *mut AnyObject;
            ns_array
        };

        // Create NSArray of output types using objc2_foundation
        let output_types_array = unsafe {
            // Convert to slice of references to AnyObject
            let refs: Vec<&objc2::runtime::AnyObject> = output_types
                .iter()
                .map(|type_obj| &*type_obj.0.cast::<objc2::runtime::AnyObject>())
                .collect();

            // Create NSArray from references
            let array = NSArray::from_slice(&refs);
            let ns_array: *mut AnyObject =
                array.as_ref() as *const objc2_foundation::NSArray as *mut AnyObject;
            ns_array
        };

        // Call the Objective-C method and get the result array
        let result_array = unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, callSymbolName: symbol_name_obj,
                inputTensors: input_tensors_array,
                outputTypes: output_types_array,
                name: name_obj,
            ];
            result
        };

        // Convert the result array to a Vec of MPSGraphTensor using objc2_foundation
        unsafe {
            // Convert to NSArray
            let array_ref: &NSArray<objc2::runtime::AnyObject> =
                &*(result_array as *const objc2_foundation::NSArray);
            let count = array_ref.len();

            let mut results = Vec::with_capacity(count);

            for i in 0..count {
                // NSArray in objc2-foundation may have different methods in different versions
                // Directly get object at index
                if i < count {
                    let obj: &objc2::runtime::AnyObject = msg_send![array_ref, objectAtIndex: i,];
                    // Get the object and convert it to a raw pointer
                    let tensor_ptr: *mut AnyObject =
                        obj as *const objc2::runtime::AnyObject as *mut AnyObject;
                    let tensor = objc2::ffi::objc_retain(tensor_ptr as *mut _);
                    results.push(MPSGraphTensor(tensor));
                }
            }

            results
        }
    }
}
