use crate::core::AsRawObject;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use objc2_foundation::NSString;

/// Activation operations for MPSGraph
impl MPSGraph {
    /// Creates a ReLU operation
    pub fn relu(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![
                self.0,
                reLUWithTensor: x.0,
                name: name_obj
            ];

            if !tensor.is_null() {
                MPSGraphTensor(tensor)
            } else {
                MPSGraphTensor(std::ptr::null_mut())
            }
        }
    }

    /// Creates a ReLU gradient operation
    pub fn relu_gradient_with_incoming_gradient(
        &self,
        gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![
                self.0,
                reLUGradientWithIncomingGradient: gradient.0,
                sourceTensor: source.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a Sigmoid operation
    pub fn sigmoid(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            // Call the Sigmoid operation with null name
            let tensor: *mut AnyObject = msg_send![self.0,
                sigmoidWithTensor: x.0,
                name: name_obj
            ];

            if !tensor.is_null() {
                MPSGraphTensor(tensor)
            } else {
                // Return null tensor if the operation failed
                MPSGraphTensor(std::ptr::null_mut())
            }
        }
    }

    /// Creates a Sigmoid gradient operation
    pub fn sigmoid_gradient_with_incoming_gradient(
        &self,
        gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, sigmoidGradientWithIncomingGradient: gradient.0,
                sourceTensor: source.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a Tanh operation
    pub fn tanh(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, tanhWithTensor: x.0, name: name_obj];

            if !tensor.is_null() {
                MPSGraphTensor(tensor)
            } else {
                MPSGraphTensor(std::ptr::null_mut())
            }
        }
    }

    /// Creates a SoftMax operation
    pub fn softmax(&self, x: &MPSGraphTensor, axis: i64, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            // Call the SoftMax operation with null name
            let tensor: *mut AnyObject = msg_send![self.0, softMaxWithTensor: x.0,
                axis: axis,
                name: name_obj
            ];

            if !tensor.is_null() {
                MPSGraphTensor(tensor)
            } else {
                MPSGraphTensor(std::ptr::null_mut())
            }
        }
    }

    /// Creates a SoftMax gradient operation
    pub fn softmax_gradient_with_incoming_gradient(
        &self,
        gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, softMaxGradientWithIncomingGradient: gradient.0,
                sourceTensor: source.0,
                axis: axis,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a Leaky ReLU operation
    pub fn leaky_relu(&self, x: &MPSGraphTensor, alpha: f32, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, leakyReLUWithTensor: x.0,
                alpha: alpha as f64,
                name: name_obj
            ];

            if !tensor.is_null() {
                MPSGraphTensor(tensor)
            } else {
                MPSGraphTensor(std::ptr::null_mut())
            }
        }
    }

    /// Creates a Leaky ReLU with alpha tensor
    pub fn leaky_relu_with_alpha_tensor(
        &self,
        x: &MPSGraphTensor,
        alpha_tensor: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, leakyReLUWithTensor: x.0,
                alphaTensor: alpha_tensor.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a Leaky ReLU gradient operation
    pub fn leaky_relu_gradient_with_incoming_gradient(
        &self,
        gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        alpha: f32,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![
                self.0,
                leakyReLUGradientWithIncomingGradient: gradient.0,
                sourceTensor: source.0,
                alpha: alpha as f64,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }
}
