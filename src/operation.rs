use objc::runtime::Object;
use std::fmt;
use crate::tensor::MPSGraphTensor;
use crate::graph::MPSGraph;

/// A wrapper for MPSGraphOperation objects
pub struct MPSGraphOperation(pub(crate) *mut Object);

impl MPSGraphOperation {
    /// Returns the input tensors of this operation
    pub fn input_tensors(&self) -> Vec<MPSGraphTensor> {
        unsafe {
            let input_tensors: *mut Object = msg_send![self.0, inputTensors];
            let count: usize = msg_send![input_tensors, count];
            let mut result = Vec::with_capacity(count);
            
            for i in 0..count {
                let tensor: *mut Object = msg_send![input_tensors, objectAtIndex:i];
                let tensor: *mut Object = msg_send![tensor, retain];
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
    
    /// Returns the output tensors of this operation
    pub fn output_tensors(&self) -> Vec<MPSGraphTensor> {
        unsafe {
            let output_tensors: *mut Object = msg_send![self.0, outputTensors];
            let count: usize = msg_send![output_tensors, count];
            let mut result = Vec::with_capacity(count);
            
            for i in 0..count {
                let tensor: *mut Object = msg_send![output_tensors, objectAtIndex:i];
                let tensor: *mut Object = msg_send![tensor, retain];
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
    
    /// Returns the graph this operation belongs to
    pub fn graph(&self) -> MPSGraph {
        unsafe {
            let graph: *mut Object = msg_send![self.0, graph];
            let graph: *mut Object = msg_send![graph, retain];
            MPSGraph(graph)
        }
    }
    
    /// Returns the name of this operation
    pub fn name(&self) -> String {
        unsafe {
            let name: *mut Object = msg_send![self.0, name];
            let utf8: *const i8 = msg_send![name, UTF8String];
            std::ffi::CStr::from_ptr(utf8).to_string_lossy().to_string()
        }
    }
    
    /// Returns the control dependencies of this operation
    pub fn control_dependencies(&self) -> Vec<MPSGraphOperation> {
        unsafe {
            let dependencies: *mut Object = msg_send![self.0, controlDependencies];
            let count: usize = msg_send![dependencies, count];
            let mut result = Vec::with_capacity(count);
            
            for i in 0..count {
                let op: *mut Object = msg_send![dependencies, objectAtIndex:i];
                let op: *mut Object = msg_send![op, retain];
                result.push(MPSGraphOperation(op));
            }
            
            result
        }
    }
}

impl Drop for MPSGraphOperation {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

impl Clone for MPSGraphOperation {
    fn clone(&self) -> Self {
        unsafe {
            let obj: *mut Object = msg_send![self.0, retain];
            MPSGraphOperation(obj)
        }
    }
}

impl fmt::Debug for MPSGraphOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MPSGraphOperation")
            .field("name", &self.name())
            .finish()
    }
}