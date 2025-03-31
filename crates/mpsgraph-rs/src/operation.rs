use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use std::fmt;
use std::ptr;

/// A wrapper for MPSGraphOperation objects
pub struct MPSGraphOperation(pub(crate) *mut AnyObject);

impl MPSGraphOperation {
    /// Returns the input tensors of this operation
    pub fn input_tensors(&self) -> Vec<MPSGraphTensor> {
        unsafe {
            let input_tensors: *mut AnyObject = msg_send![self.0, inputTensors];
            let count: usize = msg_send![input_tensors, count];
            let mut result = Vec::with_capacity(count);

            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![input_tensors, objectAtIndex: i,];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _);
                result.push(MPSGraphTensor(tensor));
            }

            result
        }
    }

    /// Returns the output tensors of this operation
    pub fn output_tensors(&self) -> Vec<MPSGraphTensor> {
        unsafe {
            let output_tensors: *mut AnyObject = msg_send![self.0, outputTensors];
            let count: usize = msg_send![output_tensors, count];
            let mut result = Vec::with_capacity(count);

            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![output_tensors, objectAtIndex: i,];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _);
                result.push(MPSGraphTensor(tensor));
            }

            result
        }
    }

    /// Returns the graph this operation belongs to
    pub fn graph(&self) -> MPSGraph {
        unsafe {
            let graph: *mut AnyObject = msg_send![self.0, graph];
            let graph = objc2::ffi::objc_retain(graph as *mut _);
            MPSGraph(graph)
        }
    }

    /// Returns the name of this operation
    pub fn name(&self) -> String {
        unsafe {
            let name: *mut AnyObject = msg_send![self.0, name];
            let utf8: *const i8 = msg_send![name, UTF8String];
            std::ffi::CStr::from_ptr(utf8).to_string_lossy().to_string()
        }
    }

    /// Returns the control dependencies of this operation
    pub fn control_dependencies(&self) -> Vec<MPSGraphOperation> {
        unsafe {
            let dependencies: *mut AnyObject = msg_send![self.0, controlDependencies];
            let count: usize = msg_send![dependencies, count];
            let mut result = Vec::with_capacity(count);

            for i in 0..count {
                let op: *mut AnyObject = msg_send![dependencies, objectAtIndex: i,];
                let op = objc2::ffi::objc_retain(op as *mut _);
                result.push(MPSGraphOperation(op));
            }

            result
        }
    }
}

impl Drop for MPSGraphOperation {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraphOperation {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _);
                MPSGraphOperation(obj)
            } else {
                MPSGraphOperation(ptr::null_mut())
            }
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
