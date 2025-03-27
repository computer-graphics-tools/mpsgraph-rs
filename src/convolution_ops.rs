use objc2::runtime::AnyObject;
use objc2::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use objc2_foundation::NSString;
use crate::core::{OurNSString, AsRawObject};

/// Convolution operations for MPSGraph
impl MPSGraph {
    /// Creates a 2D convolution operation
    pub fn convolution2d(&self, 
                       input:  &MPSGraphTensor, 
                       weights:  &MPSGraphTensor,
                       strides:  (usize, usize),
                       padding:  (usize, usize, usize, usize),
                       dilations:  (usize, usize),
                       name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => OurNSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Create arrays for strides, paddings, and dilations
            let strides_array = create_dimensions(&[strides.0, strides.1]);
            let padding_array = create_dimensions(&[padding.0, padding.1, padding.2, padding.3]);
            let dilations_array = create_dimensions(&[dilations.0, dilations.1]);
            
            let tensor: *mut AnyObject = msg_send![self.0, convolution2DWithSourceTensor: input.0,
                weightsTensor: weights.0,
                descriptor: std::ptr::null_mut::<AnyObject>(),
                strides: strides_array,
                padding: padding_array,
                dilationRates: dilations_array,
                name: name_obj,
            ];
            
            // Release temporary objects
            objc2::ffi::objc_release(strides_array as *mut _);
            objc2::ffi::objc_release(padding_array as *mut _);
            objc2::ffi::objc_release(dilations_array as *mut _);
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a 2D convolution with bias operation
    pub fn convolution2d_with_bias(&self, 
                                input:  &MPSGraphTensor, 
                                weights:  &MPSGraphTensor,
                                bias:  &MPSGraphTensor,
                                strides:  (usize, usize),
                                padding:  (usize, usize, usize, usize),
                                dilations:  (usize, usize),
                                name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => OurNSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Create arrays for strides, paddings, and dilations
            let strides_array = create_dimensions(&[strides.0, strides.1]);
            let padding_array = create_dimensions(&[padding.0, padding.1, padding.2, padding.3]);
            let dilations_array = create_dimensions(&[dilations.0, dilations.1]);
            
            let tensor: *mut AnyObject = msg_send![self.0, convolution2DWithSourceTensor: input.0,
                weightsTensor: weights.0,
                biasTensor: bias.0,
                descriptor: std::ptr::null_mut::<AnyObject>(),
                strides: strides_array,
                padding: padding_array,
                dilationRates: dilations_array,
                name: name_obj,
            ];
            
            // Release temporary objects
            objc2::ffi::objc_release(strides_array as *mut _);
            objc2::ffi::objc_release(padding_array as *mut _);
            objc2::ffi::objc_release(dilations_array as *mut _);
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a 2D depthwise convolution operation
    pub fn depthwise_convolution2d(&self, 
                                input:  &MPSGraphTensor, 
                                weights:  &MPSGraphTensor,
                                strides:  (usize, usize),
                                padding:  (usize, usize, usize, usize),
                                dilations:  (usize, usize),
                                name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Create arrays for strides, paddings, and dilations
            let strides_array = create_dimensions(&[strides.0, strides.1]);
            let padding_array = create_dimensions(&[padding.0, padding.1, padding.2, padding.3]);
            let dilations_array = create_dimensions(&[dilations.0, dilations.1]);
            
            let tensor: *mut AnyObject = msg_send![self.0, depthwiseConvolution2DWithSourceTensor: input.0,
                weightsTensor: weights.0,
                descriptor: std::ptr::null_mut::<AnyObject>(),
                strides: strides_array,
                padding: padding_array,
                dilationRates: dilations_array,
                name: name_obj,
            ];
            
            // Release temporary objects
            objc2::ffi::objc_release(strides_array as *mut _);
            objc2::ffi::objc_release(padding_array as *mut _);
            objc2::ffi::objc_release(dilations_array as *mut _);
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a 2D transpose convolution operation
    pub fn convolution_transpose2d(&self, 
                                input:  &MPSGraphTensor, 
                                weights:  &MPSGraphTensor,
                                output_shape:  &MPSGraphTensor,
                                strides:  (usize, usize),
                                padding:  (usize, usize, usize, usize),
                                dilations:  (usize, usize),
                                name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            // Create arrays for strides, paddings, and dilations
            let strides_array = create_dimensions(&[strides.0, strides.1]);
            let padding_array = create_dimensions(&[padding.0, padding.1, padding.2, padding.3]);
            let dilations_array = create_dimensions(&[dilations.0, dilations.1]);
            
            let tensor: *mut AnyObject = msg_send![self.0, transposedConvolution2DWithSourceTensor: input.0,
                weightsTensor: weights.0,
                outputShapeTensor: output_shape.0,
                descriptor: std::ptr::null_mut::<AnyObject>(),
                strides: strides_array,
                padding: padding_array,
                dilationRates: dilations_array,
                name: name_obj,
            ];
            
            // Release temporary objects
            objc2::ffi::objc_release(strides_array as *mut _);
            objc2::ffi::objc_release(padding_array as *mut _);
            objc2::ffi::objc_release(dilations_array as *mut _);
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}

/// Helper function to create NSArray of dimensions
fn create_dimensions(dimensions: &[usize]) -> *mut AnyObject {
    unsafe {
        let nsnumber_name = c"NSNumber";
        let cls = objc2::runtime::AnyClass::get(nsnumber_name).unwrap();
        let dimensions: Vec<*mut AnyObject> = dimensions.iter()
            .map(|&d| {
                let obj: *mut AnyObject = msg_send![cls, alloc];
                let obj: *mut AnyObject = msg_send![obj, initWithUnsignedLongLong: d,];
                obj
            })
            .collect();
        
        let nsarray_name = c"NSArray";
        let array_cls = objc2::runtime::AnyClass::get(nsarray_name).unwrap();
        let dims_array: *mut AnyObject = msg_send![array_cls, alloc];
        let dims_array: *mut AnyObject = msg_send![dims_array, initWithObjects: dimensions.as_ptr(),
            count: dimensions.len(),
        ];
        
        // Release NSNumber objects
        for d in dimensions {
            objc2::ffi::objc_release(d as *mut _);
        }
        
        dims_array
    }
}