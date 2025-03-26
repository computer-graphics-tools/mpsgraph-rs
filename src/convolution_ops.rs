use objc::runtime::Object;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::NSString;

/// Convolution operations for MPSGraph
impl MPSGraph {
    /// Creates a 2D convolution operation
    pub fn convolution2d(&self, 
                       input: &MPSGraphTensor, 
                       weights: &MPSGraphTensor,
                       strides: (usize, usize),
                       padding: (usize, usize, usize, usize),
                       dilations: (usize, usize),
                       name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            // Create arrays for strides, paddings, and dilations
            let strides_array = create_dimensions(&[strides.0, strides.1]);
            let padding_array = create_dimensions(&[padding.0, padding.1, padding.2, padding.3]);
            let dilations_array = create_dimensions(&[dilations.0, dilations.1]);
            
            let tensor: *mut Object = msg_send![self.0, 
                convolution2DWithSourceTensor:input.0
                weightsTensor:weights.0
                descriptor:std::ptr::null_mut::<Object>()
                strides:strides_array
                padding:padding_array
                dilationRates:dilations_array
                name:name_obj
            ];
            
            // Release temporary objects
            let _: () = msg_send![strides_array, release];
            let _: () = msg_send![padding_array, release];
            let _: () = msg_send![dilations_array, release];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a 2D convolution with bias operation
    pub fn convolution2d_with_bias(&self, 
                                input: &MPSGraphTensor, 
                                weights: &MPSGraphTensor,
                                bias: &MPSGraphTensor,
                                strides: (usize, usize),
                                padding: (usize, usize, usize, usize),
                                dilations: (usize, usize),
                                name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            // Create arrays for strides, paddings, and dilations
            let strides_array = create_dimensions(&[strides.0, strides.1]);
            let padding_array = create_dimensions(&[padding.0, padding.1, padding.2, padding.3]);
            let dilations_array = create_dimensions(&[dilations.0, dilations.1]);
            
            let tensor: *mut Object = msg_send![self.0, 
                convolution2DWithSourceTensor:input.0
                weightsTensor:weights.0
                biasTensor:bias.0
                descriptor:std::ptr::null_mut::<Object>()
                strides:strides_array
                padding:padding_array
                dilationRates:dilations_array
                name:name_obj
            ];
            
            // Release temporary objects
            let _: () = msg_send![strides_array, release];
            let _: () = msg_send![padding_array, release];
            let _: () = msg_send![dilations_array, release];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a 2D depthwise convolution operation
    pub fn depthwise_convolution2d(&self, 
                                input: &MPSGraphTensor, 
                                weights: &MPSGraphTensor,
                                strides: (usize, usize),
                                padding: (usize, usize, usize, usize),
                                dilations: (usize, usize),
                                name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            // Create arrays for strides, paddings, and dilations
            let strides_array = create_dimensions(&[strides.0, strides.1]);
            let padding_array = create_dimensions(&[padding.0, padding.1, padding.2, padding.3]);
            let dilations_array = create_dimensions(&[dilations.0, dilations.1]);
            
            let tensor: *mut Object = msg_send![self.0, 
                depthwiseConvolution2DWithSourceTensor:input.0
                weightsTensor:weights.0
                descriptor:std::ptr::null_mut::<Object>()
                strides:strides_array
                padding:padding_array
                dilationRates:dilations_array
                name:name_obj
            ];
            
            // Release temporary objects
            let _: () = msg_send![strides_array, release];
            let _: () = msg_send![padding_array, release];
            let _: () = msg_send![dilations_array, release];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a 2D transpose convolution operation
    pub fn convolution_transpose2d(&self, 
                                input: &MPSGraphTensor, 
                                weights: &MPSGraphTensor,
                                output_shape: &MPSGraphTensor,
                                strides: (usize, usize),
                                padding: (usize, usize, usize, usize),
                                dilations: (usize, usize),
                                name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            // Create arrays for strides, paddings, and dilations
            let strides_array = create_dimensions(&[strides.0, strides.1]);
            let padding_array = create_dimensions(&[padding.0, padding.1, padding.2, padding.3]);
            let dilations_array = create_dimensions(&[dilations.0, dilations.1]);
            
            let tensor: *mut Object = msg_send![self.0, 
                transposedConvolution2DWithSourceTensor:input.0
                weightsTensor:weights.0
                outputShapeTensor:output_shape.0
                descriptor:std::ptr::null_mut::<Object>()
                strides:strides_array
                padding:padding_array
                dilationRates:dilations_array
                name:name_obj
            ];
            
            // Release temporary objects
            let _: () = msg_send![strides_array, release];
            let _: () = msg_send![padding_array, release];
            let _: () = msg_send![dilations_array, release];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
}

/// Helper function to create NSArray of dimensions
fn create_dimensions(dimensions: &[usize]) -> *mut Object {
    unsafe {
        let cls = objc::runtime::Class::get("NSNumber").unwrap();
        let dimensions: Vec<*mut Object> = dimensions.iter()
            .map(|&d| {
                let obj: *mut Object = msg_send![cls, alloc];
                let obj: *mut Object = msg_send![obj, initWithUnsignedLongLong:d];
                obj
            })
            .collect();
        
        let array_cls = objc::runtime::Class::get("NSArray").unwrap();
        let dims_array: *mut Object = msg_send![array_cls, alloc];
        let dims_array: *mut Object = msg_send![dims_array,
            initWithObjects:dimensions.as_ptr()
            count:dimensions.len()
        ];
        
        // Release NSNumber objects
        for d in dimensions {
            let _: () = msg_send![d, release];
        }
        
        dims_array
    }
}