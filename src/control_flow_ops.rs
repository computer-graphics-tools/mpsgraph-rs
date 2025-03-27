use objc2::runtime::AnyObject;
use std::ops::Fn;
use std::ffi::c_void;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::operation::MPSGraphOperation;
use crate::core::NSString;

/// Control flow operations for MPSGraph
///
/// These operations allow for dynamic control flow within the graph, including:
/// - Control dependencies
/// - Conditional execution (if-then-else)
/// - While loops
/// - For loops
impl MPSGraph {
    /// Creates a control dependency between operations.
    ///
    /// This ensures that operations in the dependent_block are executed
    /// only after all operations in the operations list have been executed.
    ///
    /// # Parameters
    ///
    /// * `operations` - Operations that must be completed before the dependent block executes
    /// * `dependent_block` - A closure that returns tensors that depend on the operations
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector of tensors that are the result of the dependent_block
    pub fn control_dependency<F>(&self, 
                              operations:  &[&MPSGraphOperation],
                              dependent_block:  F,
                              name:  Option<&str>) -> Vec<MPSGraphTensor>
    where
        F:  Fn() -> Vec<MPSGraphTensor>
    {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Convert operations to NSArray
            let operations_array = crate::core::NSArray::from_objects(
                &operations.iter().map(|op| op.0).collect::<Vec<_>>()
            );
            
            // Create the block callback
            let block = ConcreteBlock::new(move || {
                // Call the user's block
                let tensors = dependent_block();
                // Convert Vec<MPSGraphTensor> to NSArray of pointers
                let tensor_ptrs: Vec<*mut Object> = tensors.iter().map(|t| t.0).collect();
                let tensor_array = crate::core::NSArray::from_objects(&tensor_ptrs);
                // Return the NSArray pointer
                tensor_array.0
            }).copy();
            
            // Call the Objective-C method
            let result_array: *mut AnyObject = msg_send![
                self.0, controlDependencyWithOperations: operations_array.0,
                dependentBlock: &*block,
                name: name_obj,
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![result_array, objectAtIndex: i,,,,,,,];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
    
    /// Creates an if-then-else operation.
    ///
    /// This allows for conditional execution of operations based on a predicate tensor.
    ///
    /// # Parameters
    ///
    /// * `predicate` - A scalar tensor that determines which branch to execute
    /// * `then_block` - A closure that returns tensors for the "then" branch
    /// * `else_block` - An optional closure that returns tensors for the "else" branch
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector of tensors that are the result of either the then_block or else_block,
    /// depending on the value of the predicate tensor
    pub fn if_op<F, G>(&self,
                    predicate:  &MPSGraphTensor,
                    then_block:  F,
                    else_block:  Option<G>,
                    name:  Option<&str>) -> Vec<MPSGraphTensor>
    where
        F:  Fn() -> Vec<MPSGraphTensor>,
        G:  Fn() -> Vec<MPSGraphTensor>
    {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Create the then block callback
            let then_callback = ConcreteBlock::new(move || {
                // Call the user's block
                let tensors = then_block();
                // Convert Vec<MPSGraphTensor> to NSArray of pointers
                let tensor_ptrs: Vec<*mut Object> = tensors.iter().map(|t| t.0).collect();
                let tensor_array = crate::core::NSArray::from_objects(&tensor_ptrs);
                // Return the NSArray pointer
                tensor_array.0
            }).copy();
            
            // Create the else block callback if provided
            let else_callback = if let Some(else_fn) = else_block {
                let block = ConcreteBlock::new(move || {
                    // Call the user's block
                    let tensors = else_fn();
                    // Convert Vec<MPSGraphTensor> to NSArray of pointers
                    let tensor_ptrs: Vec<*mut Object> = tensors.iter().map(|t| t.0).collect();
                    let tensor_array = crate::core::NSArray::from_objects(&tensor_ptrs);
                    // Return the NSArray pointer
                    tensor_array.0
                }).copy();
                &*block as *const _ as *mut c_void
            } else {
                std::ptr::null_mut()
            };
            
            // Call the Objective-C method
            let result_array: *mut AnyObject = msg_send![
                self.0, ifWithPredicateTensor: predicate.0,
                thenBlock: &*then_callback,
                elseBlock: else_callback,
                name: name_obj,
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![result_array, objectAtIndex: i,,];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
    
    /// Creates a while loop operation.
    ///
    /// This allows for iterative execution of operations until a condition is met.
    ///
    /// # Parameters
    ///
    /// * `initial_inputs` - Initial tensors passed to the loop
    /// * `before_block` - A closure that evaluates the condition and produces intermediate tensors
    /// * `after_block` - A closure that executes the loop body
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector of tensors that are the final result of the while loop
    pub fn while_loop<F, G>(&self,
                         initial_inputs:  &[&MPSGraphTensor],
                         before_block:  F,
                         after_block:  G,
                         name:  Option<&str>) -> Vec<MPSGraphTensor>
    where
        F:  Fn(&[MPSGraphTensor], &mut Vec<MPSGraphTensor>) -> MPSGraphTensor,
        G:  Fn(&[MPSGraphTensor]) -> Vec<MPSGraphTensor>
    {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Convert initial inputs to NSArray
            let initial_inputs_array = crate::core::NSArray::from_objects(
                &initial_inputs.iter().map(|t| t.0).collect::<Vec<_>>()
            );
            
            // Create the before block callback
            let before_callback = ConcreteBlock::new(move |inputs: *mut AnyObject, results: *mut AnyObject| -> *mut AnyObject {
                // Convert NSArray to Vec<MPSGraphTensor>
                let inputs_count: usize = msg_send![inputs, count];
                let mut input_tensors = Vec::with_capacity(inputs_count);
                for i in 0..inputs_count {
                    let tensor: *mut AnyObject = msg_send![inputs, objectAtIndex: i,];
                    input_tensors.push(MPSGraphTensor(tensor));
                }
                
                // Create a mutable Vec to hold result tensors
                let mut result_tensors = Vec::new();
                
                // Call the user's block
                let condition = before_block(&input_tensors, &mut result_tensors);
                
                // Set results array
                if !result_tensors.is_empty() {
                    let result_ptrs: Vec<*mut Object> = result_tensors.iter().map(|t| t.0).collect();
                    for i in 0..result_ptrs.len() {
                        let _: () = msg_send![results, addObject: result_ptrs[i],];
                    }
                }
                
                // Return the condition tensor
                condition.0
            }).copy();
            
            // Create the after block callback
            let after_callback = ConcreteBlock::new(move |body_args: *mut AnyObject| -> *mut AnyObject {
                // Convert NSArray to Vec<MPSGraphTensor>
                let args_count: usize = msg_send![body_args, count];
                let mut body_arguments = Vec::with_capacity(args_count);
                for i in 0..args_count {
                    let tensor: *mut AnyObject = msg_send![body_args, objectAtIndex: i,,];
                    body_arguments.push(MPSGraphTensor(tensor));
                }
                
                // Call the user's block
                let result_tensors = after_block(&body_arguments);
                
                // Convert Vec<MPSGraphTensor> to NSArray
                let result_ptrs: Vec<*mut Object> = result_tensors.iter().map(|t| t.0).collect();
                let result_array = crate::core::NSArray::from_objects(&result_ptrs);
                
                // Return the NSArray
                result_array.0
            }).copy();
            
            // Call the Objective-C method
            let result_array: *mut AnyObject = msg_send![
                self.0, whileWithInitialInputs: initial_inputs_array.0,
                before: &*before_callback,
                after: &*after_callback,
                name: name_obj,
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![result_array, objectAtIndex: i,,,,,,,];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
    
    /// Creates a for loop operation.
    ///
    /// This allows for iterative execution of operations for a specified range.
    ///
    /// # Parameters
    ///
    /// * `lower_bound` - Lower bound value of the loop (inclusive)
    /// * `upper_bound` - Upper bound value of the loop (exclusive)
    /// * `step` - Step value of the loop (must be positive)
    /// * `initial_body_arguments` - Initial tensors passed to the loop body
    /// * `body_block` - A closure that executes the loop body for each iteration
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector of tensors that are the final result of the for loop
    pub fn for_loop<F>(&self, 
                    lower_bound:  &MPSGraphTensor,
                    upper_bound:  &MPSGraphTensor,
                    step:  &MPSGraphTensor,
                    initial_body_arguments:  &[&MPSGraphTensor],
                    body_block:  F,
                    name:  Option<&str>) -> Vec<MPSGraphTensor>
    where
        F:  Fn(&MPSGraphTensor, &[MPSGraphTensor]) -> Vec<MPSGraphTensor>
    {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Convert initial body arguments to NSArray
            let initial_args_array = crate::core::NSArray::from_objects(
                &initial_body_arguments.iter().map(|t| t.0).collect::<Vec<_>>()
            );
            
            // Create the body block callback
            let body_callback = ConcreteBlock::new(move |index: *mut AnyObject, args: *mut AnyObject| -> *mut AnyObject {
                // Convert to MPSGraphTensor
                let index_tensor = MPSGraphTensor(index);
                
                // Convert NSArray to Vec<MPSGraphTensor>
                let args_count: usize = msg_send![args, count];
                let mut body_arguments = Vec::with_capacity(args_count);
                for i in 0..args_count {
                    let tensor: *mut AnyObject = msg_send![args, objectAtIndex: i,,,];
                    body_arguments.push(MPSGraphTensor(tensor));
                }
                
                // Call the user's block
                let result_tensors = body_block(&index_tensor, &body_arguments);
                
                // Convert Vec<MPSGraphTensor> to NSArray
                let result_ptrs: Vec<*mut Object> = result_tensors.iter().map(|t| t.0).collect();
                let result_array = crate::core::NSArray::from_objects(&result_ptrs);
                
                // Return the NSArray
                result_array.0
            }).copy();
            
            // Call the Objective-C method
            let result_array: *mut AnyObject = msg_send![
                self.0, forLoopWithLowerBound: lower_bound.0,
                upperBound: upper_bound.0,
                step: step.0,
                initialBodyArguments: initial_args_array.0,
                body: &*body_callback,
                name: name_obj,
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![result_array, objectAtIndex: i,,];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
    
    /// Creates a for loop operation with a specific number of iterations.
    ///
    /// This is a more direct version of the for loop that just specifies the total number of iterations.
    ///
    /// # Parameters
    ///
    /// * `num_iterations` - Number of iterations for the loop
    /// * `initial_body_arguments` - Initial tensors passed to the loop body
    /// * `body_block` - A closure that executes the loop body for each iteration
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector of tensors that are the final result of the for loop
    pub fn for_loop_with_iterations<F>(&self,
                                    num_iterations:  &MPSGraphTensor,
                                    initial_body_arguments:  &[&MPSGraphTensor],
                                    body_block:  F,
                                    name:  Option<&str>) -> Vec<MPSGraphTensor>
    where
        F:  Fn(&MPSGraphTensor, &[MPSGraphTensor]) -> Vec<MPSGraphTensor>
    {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Convert initial body arguments to NSArray
            let initial_args_array = crate::core::NSArray::from_objects(
                &initial_body_arguments.iter().map(|t| t.0).collect::<Vec<_>>()
            );
            
            // Create the body block callback
            let body_callback = ConcreteBlock::new(move |index: *mut AnyObject, args: *mut AnyObject| -> *mut AnyObject {
                // Convert to MPSGraphTensor
                let index_tensor = MPSGraphTensor(index);
                
                // Convert NSArray to Vec<MPSGraphTensor>
                let args_count: usize = msg_send![args, count];
                let mut body_arguments = Vec::with_capacity(args_count);
                for i in 0..args_count {
                    let tensor: *mut AnyObject = msg_send![args, objectAtIndex: i,];
                    body_arguments.push(MPSGraphTensor(tensor));
                }
                
                // Call the user's block
                let result_tensors = body_block(&index_tensor, &body_arguments);
                
                // Convert Vec<MPSGraphTensor> to NSArray
                let result_ptrs: Vec<*mut Object> = result_tensors.iter().map(|t| t.0).collect();
                let result_array = crate::core::NSArray::from_objects(&result_ptrs);
                
                // Return the NSArray
                result_array.0
            }).copy();
            
            // Call the Objective-C method
            let result_array: *mut AnyObject = msg_send![
                self.0, forLoopWithNumberOfIterations: num_iterations.0,
                initialBodyArguments: initial_args_array.0,
                body: &*body_callback,
                name: name_obj,
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![result_array, objectAtIndex: i,,,,,,,];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
}

/// A concrete implementation of an Objective-C block
#[repr(C)]
struct ConcreteBlock<F> {
    isa:  *const c_void,
    flags:  usize,
    reserved:  usize,
    invoke:  *const c_void,
    descriptor:  *const BlockDescriptor,
    closure:  F,
}

/// Block descriptor containing size and copy/dispose functions
#[repr(C)]
struct BlockDescriptor {
    reserved:  usize,
    size:  usize,
    copy_helper:  *const c_void,
    dispose_helper:  *const c_void,
}

impl<F> ConcreteBlock<F> {
    /// Creates a new ConcreteBlock with the given closure
    fn new(closure: F) -> Self {
        ConcreteBlock {
            isa:  0 as *const _,
            flags:  0,
            reserved:  0,
            invoke:  0 as *const _,
            descriptor:  0 as *const _,
            closure,
        }
    }
    
    /// Creates a copy of the block that can be passed to Objective-C
    fn copy(self) -> *mut Self {
        unsafe {
            // Try to use the system Block_copy function first
            #[cfg(target_os = "macos")]
            {
                let copied = Block_copy(&self as *const _ as *const c_void);
                if !copied.is_null() {
                    return copied as *mut Self;
                }
            }
            
            // If that fails, fall back to our alternative implementation
            let copied = block_copy(&self as *const _ as *const c_void);
            copied as *mut Self
        }
    }
}

// Allow Block_copy to be used
extern "C" {
    fn Block_copy(block: *const c_void) -> *mut c_void;
}

// Alternative implementation using objc_msgSend if Block_copy isn't found
fn block_copy(block: *const c_void) -> *mut c_void {
    unsafe {
        let cls = objc2::runtime::AnyClass::get(std::ffi::CStr::from_bytes_with_nul(b("NSBlock").unwrap_or_else(|| {
            objc2::runtime::AnyClass::get(std::ffi::CStr::from_bytes_with_nul(b("NSObject").unwrap()
        });
        let block_obj = block as *mut objc::runtime::Object;
        let copied: *mut objc::runtime::Object = msg_send![block_obj, copy];
        copied as *mut c_void
    }
}

    use super::*;
    use crate::core::MPSDataType;
    use crate::device::MPSGraphDevice;
    use std::collections::HashMap;
    
    #[test]
    fn test_if_op() {
        // This test will be skipped in environments without Metal
        if crate::tests::should_skip_test("test_if_op") {
            return;
        }
        
        let graph = MPSGraph::new();
        let device = MPSGraphDevice::new();
        
        // Create a predicate tensor (true)
        let predicate = graph.constant_scalar_value(1, MPSDataType::Bool, Some("predicate"));
        
        // Create tensors for the then branch
        let then_value = graph.constant_scalar_value(42.0f32, MPSDataType::Float32, Some("then_value"));
        
        // Create tensors for the else branch
        let else_value = graph.constant_scalar_value(-42.0f32, MPSDataType::Float32, Some("else_value"));
        
        // Create the if-then-else operation
        let result = graph.if_op(
            &predicate,
            || vec![then_value.clone()],
            Some(|| vec![else_value.clone()]),
            Some("if_op")
        );
        
        // Verify that we got one result
        assert_eq!(result.len(), 1, "If operation should return 1 tensor");
        
        // Run the graph
        let results = graph.run_graph(&result, &device, None);
        
        // Get the result
        let output = results.get(&result[0]).unwrap().to_vec::<f32>();
        
        // Since predicate is true, the result should be the then_value
        assert_eq!(output[0], 42.0f32, "Result should be the then_value");
    }
    
    #[test]
    fn test_for_loop() {
        // This test will be skipped in environments without Metal
        if crate::tests::should_skip_test("test_for_loop") {
            return;
        }
        
        let graph = MPSGraph::new();
        let device = MPSGraphDevice::new();
        
        // Create bounds and step for the loop
        let lower_bound = graph.constant_scalar_value(0i32, MPSDataType::Int32, Some("lower_bound"));
        let upper_bound = graph.constant_scalar_value(5i32, MPSDataType::Int32, Some("upper_bound"));
        let step = graph.constant_scalar_value(1i32, MPSDataType::Int32, Some("step"));
        
        // Create initial value for accumulator (0.0)
        let initial_value = graph.constant_scalar_value(0.0f32, MPSDataType::Float32, Some("initial"));
        
        // Create the for loop to sum numbers from 0 to 4
        let result = graph.for_loop(
            &lower_bound,
            &upper_bound,
            &step,
            &[&initial_value],
            |index, args| {
                // Convert index to float
                let index_float = graph.cast(index, MPSDataType::Float32, None);
                
                // Add the current index to the accumulator
                let new_acc = graph.add(&args[0], &index_float, None);
                
                // Return the updated accumulator
                vec![new_acc]
            },
            Some("for_loop")
        );
        
        // Verify that we got one result
        assert_eq!(result.len(), 1, "For loop should return 1 tensor");
        
        // Run the graph
        let results = graph.run_graph(&result, &device, None);
        
        // Get the result
        let output = results.get(&result[0]).unwrap().to_vec::<f32>();
        
        // The sum of 0 + 1 + 2 + 3 + 4 = 10
        assert_eq!(output[0], 10.0f32, "Result should be the sum 0+1+2+3+4=10");
    }
}