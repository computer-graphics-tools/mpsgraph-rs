use objc2::runtime::AnyObject;
use objc2::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use objc2_foundation::{NSString, NSArray, NSNumber};
use crate::core::AsRawObject;
use std::ptr;

/// Convolution padding mode
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MPSGraphPaddingMode {
    /// Valid padding - no padding
    Valid = 0,
    /// Same padding - pad to maintain same size
    Same = 1,
    /// Explicit padding - user-specified padding values
    Explicit = 2,
}

/// Dataflow direction for convolution
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MPSGraphConvolutionDataLayout {
    /// Data is arranged as NHWC (batch, height, width, channels)
    NHWC = 0,
    /// Data is arranged as NCHW (batch, channels, height, width)
    NCHW = 1,
}

/// Weight layout for convolution
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MPSGraphWeightsLayout {
    /// Weights arranged as HWIO (height, width, input channels, output channels)
    HWIO = 0,
    /// Weights arranged as OHWI (output channels, height, width, input channels)
    OHWI = 1,
    /// Weights arranged as IHWO (input channels, height, width, output channels)
    IHWO = 2,
}

/// Descriptor for 2D convolution operations
pub struct MPSGraphConvolution2DOpDescriptor(pub(crate) *mut AnyObject);

impl MPSGraphConvolution2DOpDescriptor {
    /// Creates a new descriptor with default parameters
    pub fn new() -> Self {
        unsafe {
            let cls = objc2::runtime::AnyClass::get(c"MPSGraphConvolution2DOpDescriptor").unwrap();
            let descriptor: *mut AnyObject = msg_send![cls, alloc];
            let descriptor: *mut AnyObject = msg_send![descriptor, init];
            
            Self(descriptor)
        }
    }
    
    /// Set the padding mode
    pub fn set_padding_mode(&mut self, mode: MPSGraphPaddingMode) -> &mut Self {
        unsafe {
            let _: () = msg_send![self.0, setPaddingMode: mode as u64];
            self
        }
    }
    
    /// Set the data layout
    pub fn set_data_layout(&mut self, layout: MPSGraphConvolutionDataLayout) -> &mut Self {
        unsafe {
            let _: () = msg_send![self.0, setDataLayout: layout as u64];
            self
        }
    }
    
    /// Set the weights layout
    pub fn set_weights_layout(&mut self, layout: MPSGraphWeightsLayout) -> &mut Self {
        unsafe {
            let _: () = msg_send![self.0, setWeightsLayout: layout as u64];
            self
        }
    }
    
    /// Set the strides (height, width)
    pub fn set_strides(&mut self, strides: (usize, usize)) -> &mut Self {
        unsafe {
            // Convert to NSArray
            let numbers = [
                NSNumber::new_usize(strides.0),
                NSNumber::new_usize(strides.1),
            ];
            let refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
            let array = NSArray::from_slice(&refs);
            
            // Get pointer and set property - using unsafe for transmute
            let array_ptr = unsafe {
                std::mem::transmute::<&NSArray<NSNumber>, *mut AnyObject>(array.as_ref())
            };
            objc2::ffi::objc_retain(array_ptr as *mut _);
            let _: () = msg_send![self.0, setStrides: array_ptr];
            objc2::ffi::objc_release(array_ptr as *mut _);
            
            self
        }
    }
    
    /// Set the dilations (height, width)
    pub fn set_dilations(&mut self, dilations: (usize, usize)) -> &mut Self {
        unsafe {
            // Convert to NSArray
            let numbers = [
                NSNumber::new_usize(dilations.0),
                NSNumber::new_usize(dilations.1),
            ];
            let refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
            let array = NSArray::from_slice(&refs);
            
            // Get pointer and set property - using unsafe for transmute
            let array_ptr = unsafe {
                std::mem::transmute::<&NSArray<NSNumber>, *mut AnyObject>(array.as_ref())
            };
            objc2::ffi::objc_retain(array_ptr as *mut _);
            let _: () = msg_send![self.0, setDilationRates: array_ptr];
            objc2::ffi::objc_release(array_ptr as *mut _);
            
            self
        }
    }
    
    /// Set the explicit padding (top, left, bottom, right)
    pub fn set_explicit_padding(&mut self, padding: (usize, usize, usize, usize)) -> &mut Self {
        unsafe {
            // Convert to NSArray
            let numbers = [
                NSNumber::new_usize(padding.0), // top
                NSNumber::new_usize(padding.1), // left
                NSNumber::new_usize(padding.2), // bottom
                NSNumber::new_usize(padding.3), // right
            ];
            let refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
            let array = NSArray::from_slice(&refs);
            
            // Get pointer and set property - using unsafe for transmute
            let array_ptr = unsafe {
                std::mem::transmute::<&NSArray<NSNumber>, *mut AnyObject>(array.as_ref())
            };
            objc2::ffi::objc_retain(array_ptr as *mut _);
            let _: () = msg_send![self.0, setExplicitPadding: array_ptr];
            objc2::ffi::objc_release(array_ptr as *mut _);
            
            self
        }
    }
    
    /// Set the groups count for grouped convolution
    pub fn set_groups(&mut self, groups: usize) -> &mut Self {
        unsafe {
            let _: () = msg_send![self.0, setGroups: groups];
            self
        }
    }
}

impl Clone for MPSGraphConvolution2DOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            MPSGraphConvolution2DOpDescriptor(objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject)
        }
    }
}

impl Drop for MPSGraphConvolution2DOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            objc2::ffi::objc_release(self.0 as *mut _);
        }
    }
}

/// Implementation for 3D convolution descriptor
pub struct MPSGraphConvolution3DOpDescriptor(pub(crate) *mut AnyObject);

impl MPSGraphConvolution3DOpDescriptor {
    /// Creates a new descriptor with default parameters
    pub fn new() -> Self {
        unsafe {
            let cls = objc2::runtime::AnyClass::get(c"MPSGraphConvolution3DOpDescriptor").unwrap();
            let descriptor: *mut AnyObject = msg_send![cls, alloc];
            let descriptor: *mut AnyObject = msg_send![descriptor, init];
            
            Self(descriptor)
        }
    }
    
    // Additional methods for 3D convolution descriptor would go here
    // Similar to the 2D version but with depth dimension added
}

impl Clone for MPSGraphConvolution3DOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            MPSGraphConvolution3DOpDescriptor(objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject)
        }
    }
}

impl Drop for MPSGraphConvolution3DOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            objc2::ffi::objc_release(self.0 as *mut _);
        }
    }
}

/// Convolution operations for MPSGraph
impl MPSGraph {
    /// Creates a 2D convolution operation using a descriptor
    pub fn convolution2d_with_descriptor(&self, 
                       input: &MPSGraphTensor, 
                       weights: &MPSGraphTensor,
                       descriptor: &MPSGraphConvolution2DOpDescriptor,
                       name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, convolution2DWithSourceTensor: input.0,
                weightsTensor: weights.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a 2D convolution operation with parameters
    pub fn convolution2d(&self, 
                       input: &MPSGraphTensor, 
                       weights: &MPSGraphTensor,
                       strides: (usize, usize),
                       padding: (usize, usize, usize, usize),
                       dilations: (usize, usize),
                       name: Option<&str>) -> MPSGraphTensor {
        let mut descriptor = MPSGraphConvolution2DOpDescriptor::new();
        descriptor.set_padding_mode(MPSGraphPaddingMode::Explicit)
                  .set_strides(strides)
                  .set_explicit_padding(padding)
                  .set_dilations(dilations);
                  
        self.convolution2d_with_descriptor(input, weights, &descriptor, name)
    }
    
    /// Creates a 2D convolution with bias operation using a descriptor
    pub fn convolution2d_with_bias_and_descriptor(&self, 
                                input: &MPSGraphTensor, 
                                weights: &MPSGraphTensor,
                                bias: &MPSGraphTensor,
                                descriptor: &MPSGraphConvolution2DOpDescriptor,
                                name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, convolution2DWithSourceTensor: input.0,
                weightsTensor: weights.0,
                biasTensor: bias.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a 2D convolution with bias operation with parameters
    pub fn convolution2d_with_bias(&self, 
                                input: &MPSGraphTensor, 
                                weights: &MPSGraphTensor,
                                bias: &MPSGraphTensor,
                                strides: (usize, usize),
                                padding: (usize, usize, usize, usize),
                                dilations: (usize, usize),
                                name: Option<&str>) -> MPSGraphTensor {
        let mut descriptor = MPSGraphConvolution2DOpDescriptor::new();
        descriptor.set_padding_mode(MPSGraphPaddingMode::Explicit)
                  .set_strides(strides)
                  .set_explicit_padding(padding)
                  .set_dilations(dilations);
                  
        self.convolution2d_with_bias_and_descriptor(input, weights, bias, &descriptor, name)
    }
    
    /// Creates a 3D convolution operation using a descriptor
    pub fn convolution3d_with_descriptor(&self, 
                       input: &MPSGraphTensor, 
                       weights: &MPSGraphTensor,
                       descriptor: &MPSGraphConvolution3DOpDescriptor,
                       name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, convolution3DWithSourceTensor: input.0,
                weightsTensor: weights.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a gradient operation for 2D convolution
    pub fn convolution2d_gradient_with_incoming_gradient_source_weights(&self,
                       incoming_gradient: &MPSGraphTensor,
                       source: &MPSGraphTensor,
                       weights: &MPSGraphTensor,
                       descriptor: &MPSGraphConvolution2DOpDescriptor,
                       name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, convolution2DDataGradientWithIncomingGradientTensor: incoming_gradient.0,
                sourceTensor: source.0,
                weightsTensor: weights.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a gradient operation for 2D convolution weights
    pub fn convolution2d_weights_gradient_with_incoming_gradient_source(&self,
                       incoming_gradient: &MPSGraphTensor,
                       source: &MPSGraphTensor,
                       descriptor: &MPSGraphConvolution2DOpDescriptor,
                       name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, convolution2DWeightsGradientWithIncomingGradientTensor: incoming_gradient.0,
                sourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
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
        // Create a descriptor and set it up for depthwise convolution
        let mut descriptor = MPSGraphConvolution2DOpDescriptor::new();
        descriptor.set_padding_mode(MPSGraphPaddingMode::Explicit)
                  .set_strides(strides)
                  .set_explicit_padding(padding)
                  .set_dilations(dilations)
                  .set_groups(0); // Special value for depthwise, actual value determined from input shape
                  
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, depthwiseConvolution2DWithSourceTensor: input.0,
                weightsTensor: weights.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
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
        // Create a descriptor for transpose convolution
        let mut descriptor = MPSGraphConvolution2DOpDescriptor::new();
        descriptor.set_padding_mode(MPSGraphPaddingMode::Explicit)
                  .set_strides(strides)
                  .set_explicit_padding(padding)
                  .set_dilations(dilations);
                  
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, transposedConvolution2DWithSourceTensor: input.0,
                weightsTensor: weights.0,
                outputShapeTensor: output_shape.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}

/// Helper function to create NSArray of dimensions using objc2-foundation
pub(crate) fn create_dimensions(dimensions: &[usize]) -> *mut AnyObject {
    // Create NSNumber objects for each dimension
    let numbers: Vec<objc2::rc::Retained<NSNumber>> = dimensions.iter()
        .map(|&d| NSNumber::new_usize(d))
        .collect();
    
    // Create array from NSNumber objects
    let refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    let array = NSArray::from_slice(&refs);
    
    // Get pointer to NSArray and retain it
    unsafe {
        let array_ptr: *mut AnyObject = std::mem::transmute::<&NSArray<NSNumber>, *mut AnyObject>(array.as_ref());
        objc2::ffi::objc_retain(array_ptr as *mut _)
    }
}