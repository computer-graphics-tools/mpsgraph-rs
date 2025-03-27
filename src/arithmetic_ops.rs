use objc2::runtime::AnyObject;
use objc2::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, AsRawObject};

/// Arithmetic operations for MPSGraph
impl MPSGraph {
    // MARK: - Unary Operations

    /// Creates an identity operation
    pub fn identity(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, identityWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns e raised to the power of the input tensor
    pub fn exp(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, exponentWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns 2 raised to the power of the input tensor
    pub fn exp2(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, exponentBase2WithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns 10 raised to the power of the input tensor
    pub fn exp10(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, exponentBase10WithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the natural logarithm of the input tensor
    pub fn log(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, logarithmWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the base-2 logarithm of the input tensor
    pub fn log2(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, logarithmBase2WithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the base-10 logarithm of the input tensor
    pub fn log10(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, logarithmBase10WithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the square of the input tensor
    pub fn square(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, squareWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the square root of the input tensor
    pub fn sqrt(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, squareRootWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the reciprocal square root of the input tensor
    pub fn rsqrt(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, reciprocalSquareRootWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the reciprocal of the input tensor
    pub fn reciprocal(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, reciprocalWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the absolute value of the input tensor
    pub fn abs(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, absoluteWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the absolute square of the input tensor (equivalent to squaring the absolute value)
    pub fn abs_square(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, absoluteSquareWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the negation of the input tensor
    pub fn negative(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, negativeWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns a tensor with the sign of each element in the input tensor
    /// -1 for negative, 0 for zero, 1 for positive
    pub fn sign(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, signWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns a tensor with 1 where sign bit is set, 0 otherwise
    pub fn signbit(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, signbitWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the ceiling of the input tensor
    pub fn ceil(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, ceilWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the floor of the input tensor
    pub fn floor(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, floorWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the rounded value of the input tensor
    pub fn round(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, roundWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the rounded to nearest integral value of the input tensor using current rounding mode
    pub fn rint(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, rintWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the sine of the input tensor
    pub fn sin(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, sinWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the cosine of the input tensor
    pub fn cos(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, cosWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the tangent of the input tensor
    pub fn tan(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, tanWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the hyperbolic sine of the input tensor
    pub fn sinh(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, sinhWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the hyperbolic cosine of the input tensor
    pub fn cosh(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, coshWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the hyperbolic tangent of the input tensor (arithmetic version)
    /// 
    /// Note: This operation is also available in activation_ops.rs.
    /// This is provided for completeness of the arithmetic operations module.
    pub fn tanh_arithmetic(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, tanhWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the inverse sine (arcsine) of the input tensor
    pub fn asin(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, asinWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the inverse cosine (arccosine) of the input tensor
    pub fn acos(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, acosWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the inverse tangent (arctangent) of the input tensor
    pub fn atan(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, atanWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the inverse hyperbolic sine of the input tensor
    pub fn asinh(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, asinhWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the inverse hyperbolic cosine of the input tensor
    pub fn acosh(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, acoshWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the inverse hyperbolic tangent of the input tensor
    pub fn atanh(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, atanhWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the error function of the input tensor
    pub fn erf(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, erfWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns a tensor that is 1 if the input is infinite, 0 otherwise
    pub fn is_infinite(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, isInfiniteWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns a tensor that is 1 if the input is finite, 0 otherwise
    pub fn is_finite(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, isFiniteWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns a tensor that is 1 if the input is NaN, 0 otherwise
    pub fn is_nan(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, isNaNWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Truncates the input tensor value
    pub fn truncate(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, truncateWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs bitwise NOT on the input tensor
    pub fn bitwise_not(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, bitwiseNOTWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Counts the number of 1 bits in each element of the input tensor
    pub fn bitwise_population_count(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, bitwisePopulationCountWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the logical NOT of the input tensor
    pub fn logical_not(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, logicalNOTWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    // MARK: - Binary Operations

    /// Performs addition of two tensors
    pub fn add(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, additionWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs subtraction of two tensors
    pub fn subtract(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, subtractionWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs multiplication of two tensors
    pub fn multiply(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, multiplicationWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs division of two tensors
    pub fn divide(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, divisionWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs division of two tensors but returns 0 if secondary is 0
    pub fn division_no_nan(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, divisionNoNaNWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs modulo operation between two tensors
    pub fn modulo(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, moduloWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs floor modulo operation
    pub fn floor_modulo(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, floorModuloWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Raises primary tensor to the power of secondary tensor
    pub fn power(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, powerWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the smaller of two tensors
    pub fn minimum(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, minimumWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the greater of two tensors
    pub fn maximum(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, maximumWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the smaller of two tensors, propagating NaNs
    pub fn minimum_with_nan_propagation(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, minimumWithNaNPropagationWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the greater of two tensors, propagating NaNs
    pub fn maximum_with_nan_propagation(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, maximumWithNaNPropagationWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns element-wise equal comparison
    pub fn equal(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, equalWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns element-wise not equal comparison
    pub fn not_equal(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, notEqualWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns element-wise less than comparison
    pub fn less_than(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, lessThanWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns element-wise less than or equal comparison
    pub fn less_than_or_equal_to(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, lessThanOrEqualWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns element-wise greater than comparison
    pub fn greater_than(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, greaterThanWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns element-wise greater than or equal comparison
    pub fn greater_than_or_equal_to(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, greaterThanOrEqualWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs logical AND of two tensors
    pub fn logical_and(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, logicalANDWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs logical OR of two tensors
    pub fn logical_or(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, logicalORWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs logical NAND of two tensors
    pub fn logical_nand(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, logicalNANDWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs logical NOR of two tensors
    pub fn logical_nor(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, logicalNORWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs logical XOR of two tensors
    pub fn logical_xor(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, logicalXORWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs logical XNOR of two tensors
    pub fn logical_xnor(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, logicalXNORWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs atan2 operation (arctangent of y/x)
    pub fn atan2(&self, y: &MPSGraphTensor, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, atan2WithPrimaryTensor: y.0,
                                                  secondaryTensor: x.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs bitwise AND of two tensors
    pub fn bitwise_and(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, bitwiseANDWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs bitwise OR of two tensors
    pub fn bitwise_or(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, bitwiseORWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs bitwise XOR of two tensors
    pub fn bitwise_xor(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, bitwiseXORWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs bitwise left shift
    pub fn left_shift(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, leftShiftWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Performs bitwise right shift
    pub fn right_shift(&self, primary: &MPSGraphTensor, secondary: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, rightShiftWithPrimaryTensor: primary.0,
                                                  secondaryTensor: secondary.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    // MARK: - Ternary Operations

    /// Creates a select operation which chooses values from true or false tensor based on predicate
    pub fn select(&self, predicate: &MPSGraphTensor, true_tensor: &MPSGraphTensor, false_tensor: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, selectWithPredicateTensor: predicate.0,
                                                  truePredicateTensor: true_tensor.0,
                                                  falsePredicateTensor: false_tensor.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a clamp operation that clamps values to the given min and max
    pub fn clamp(&self, tensor: &MPSGraphTensor, min_tensor: &MPSGraphTensor, max_tensor: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, clampWithTensor: tensor.0,
                                                  minValueTensor: min_tensor.0,
                                                  maxValueTensor: max_tensor.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    // MARK: - Complex Number Operations

    /// Returns the real part of a complex tensor
    pub fn real_part(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, realPartOfTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the imaginary part of a complex tensor
    pub fn imaginary_part(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, imaginaryPartOfTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a complex tensor from real and imaginary parts
    pub fn complex_with_real_imaginary(&self, real_tensor: &MPSGraphTensor, imaginary_tensor: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, complexTensorWithRealTensor: real_tensor.0,
                                                  imaginaryTensor: imaginary_tensor.0,
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Returns the complex conjugate of a complex tensor
    pub fn conjugate(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![self.0, conjugateWithTensor: x.0, 
                                                  name: name_obj];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}