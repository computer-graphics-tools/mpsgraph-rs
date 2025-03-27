use objc2::runtime::AnyObject;
use objc2::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{OurNSString, AsRawObject};

/// Activation operations for MPSGraph
impl MPSGraph {
    /// Creates a Leaky ReLU operation
    pub fn leaky_relu(&self, x: &MPSGraphTensor, alpha: f32, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => OurNSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, leakyReLUWithTensor: x.0,
                alpha: alpha as f64,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a parametric ReLU operation
    pub fn prelu(&self, x: &MPSGraphTensor, alpha: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => OurNSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, PReLUWithTensor: x.0,
                alpha: alpha.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a GELU (Gaussian Error Linear Unit) operation
    pub fn gelu(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => OurNSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, gELUWithTensor: x.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Hard Sigmoid operation
    pub fn hard_sigmoid(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => OurNSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, hardSigmoidWithTensor: x.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Softplus operation
    pub fn softplus(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => OurNSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, softPlusWithTensor: x.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a log softmax operation
    pub fn log_softmax(&self, x: &MPSGraphTensor, axis: i64, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => OurNSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, logSoftMaxWithTensor: x.0,
                axis: axis,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}