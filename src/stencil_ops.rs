use objc2::runtime::AnyObject;
use objc2::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, AsRawObject};
use crate::shape::MPSShape;

/// The reduction mode for stencil operations.
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum MPSGraphReductionMode {
    /// Min reduction
    Min = 0,
    /// Max reduction
    Max = 1,
    /// Sum reduction
    Sum = 2,
    /// Product reduction
    Product = 3,
    /// Argument Min reduction
    ArgumentMin = 4,
    /// Argument Max reduction
    ArgumentMax = 5,
}

/// Descriptor for stencil operations
pub struct MPSGraphStencilOpDescriptor(pub(crate) *mut AnyObject);

impl Default for MPSGraphStencilOpDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl MPSGraphStencilOpDescriptor {
    /// Creates a new stencil operation descriptor with default values
    pub fn new() -> Self {
        unsafe {
            let class_name = c"MPSGraphStencilOpDescriptor";
            if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                let descriptor: *mut AnyObject = msg_send![cls, descriptor];
                let descriptor = objc2::ffi::objc_retain(descriptor as *mut _) as *mut AnyObject;
                MPSGraphStencilOpDescriptor(descriptor)
            } else {
                // Fall back to a null descriptor if class not found
                MPSGraphStencilOpDescriptor(std::ptr::null_mut())
            }
        }
    }
    
    /// Creates a new stencil operation descriptor with the specified padding style
    pub fn with_padding_style(padding_style: crate::convolution_transpose_ops::PaddingStyle) -> Self {
        unsafe {
            let class_name = c"MPSGraphStencilOpDescriptor";
            if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                let descriptor: *mut AnyObject = msg_send![cls, descriptorWithPaddingStyle: padding_style as i64];
                let descriptor = objc2::ffi::objc_retain(descriptor as *mut _) as *mut AnyObject;
                MPSGraphStencilOpDescriptor(descriptor)
            } else {
                // Fall back to a null descriptor if class not found
                MPSGraphStencilOpDescriptor(std::ptr::null_mut())
            }
        }
    }
    
    /// Creates a new stencil operation descriptor with the specified explicit padding
    pub fn with_explicit_padding(explicit_padding: &MPSShape) -> Self {
        unsafe {
            let class_name = c"MPSGraphStencilOpDescriptor";
            if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                let descriptor: *mut AnyObject = msg_send![cls, descriptorWithExplicitPadding: explicit_padding.0,];
                let descriptor = objc2::ffi::objc_retain(descriptor as *mut _) as *mut AnyObject;
                MPSGraphStencilOpDescriptor(descriptor)
            } else {
                // Fall back to a null descriptor if class not found
                MPSGraphStencilOpDescriptor(std::ptr::null_mut())
            }
        }
    }
    
    /// Creates a new stencil operation descriptor with the specified offsets and explicit padding
    pub fn with_offsets_and_explicit_padding(offsets: &MPSShape, explicit_padding: &MPSShape) -> Self {
        unsafe {
            let class_name = c"MPSGraphStencilOpDescriptor";
            if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                let descriptor: *mut AnyObject = msg_send![cls, descriptorWithOffsets: offsets.0, explicitPadding: explicit_padding.0,];
                let descriptor = objc2::ffi::objc_retain(descriptor as *mut _) as *mut AnyObject;
                MPSGraphStencilOpDescriptor(descriptor)
            } else {
                // Fall back to a null descriptor if class not found
                MPSGraphStencilOpDescriptor(std::ptr::null_mut())
            }
        }
    }
    
    /// Creates a new stencil operation descriptor with all parameters specified
    pub fn with_all_params(
        reduction_mode:  MPSGraphReductionMode,
        offsets:  &MPSShape,
        strides:  &MPSShape,
        dilation_rates:  &MPSShape,
        explicit_padding:  &MPSShape,
        boundary_mode:  crate::sample_grid_ops::MPSGraphPaddingMode,
        padding_style:  crate::convolution_transpose_ops::PaddingStyle,
        padding_constant:  f32
    ) -> Self {
        unsafe {
            let class_name = c"MPSGraphStencilOpDescriptor";
            if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                let descriptor: *mut AnyObject = msg_send![cls, descriptorWithReductionMode: reduction_mode as u64,
                    offsets: offsets.0,
                    strides: strides.0,
                    dilationRates: dilation_rates.0,
                    explicitPadding: explicit_padding.0,
                    boundaryMode: boundary_mode as i64,
                    paddingStyle: padding_style as u64,
                    paddingConstant: padding_constant,
                ];
                let descriptor = objc2::ffi::objc_retain(descriptor as *mut _) as *mut AnyObject;
                MPSGraphStencilOpDescriptor(descriptor)
            } else {
                // Fall back to a null descriptor if class not found
                MPSGraphStencilOpDescriptor(std::ptr::null_mut())
            }
        }
    }
    
    /// Sets the reduction mode
    pub fn set_reduction_mode(&self, mode: MPSGraphReductionMode) {
        unsafe {
            let _: () = msg_send![self.0, setReductionMode: mode as u64];
        }
    }
    
    /// Sets the offsets
    pub fn set_offsets(&self, offsets: &MPSShape) {
        unsafe {
            let _: () = msg_send![self.0, setOffsets: offsets.0,];
        }
    }
    
    /// Sets the strides
    pub fn set_strides(&self, strides: &MPSShape) {
        unsafe {
            let _: () = msg_send![self.0, setStrides: strides.0,];
        }
    }
    
    /// Sets the dilation rates
    pub fn set_dilation_rates(&self, dilation_rates: &MPSShape) {
        unsafe {
            let _: () = msg_send![self.0, setDilationRates: dilation_rates.0,];
        }
    }
    
    /// Sets the explicit padding
    pub fn set_explicit_padding(&self, explicit_padding: &MPSShape) {
        unsafe {
            let _: () = msg_send![self.0, setExplicitPadding: explicit_padding.0,];
        }
    }
    
    /// Sets the boundary mode
    pub fn set_boundary_mode(&self, mode: crate::sample_grid_ops::MPSGraphPaddingMode) {
        unsafe {
            let _: () = msg_send![self.0, setBoundaryMode: mode as i64];
        }
    }
    
    /// Sets the padding style
    pub fn set_padding_style(&self, style: crate::convolution_transpose_ops::PaddingStyle) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingStyle: style as u64];
        }
    }
    
    /// Sets the padding constant
    pub fn set_padding_constant(&self, value: f32) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingConstant: value,];
        }
    }
}

impl Drop for MPSGraphStencilOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            objc2::ffi::objc_release(self.0 as *mut _);
        }
    }
}

impl Clone for MPSGraphStencilOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let desc: *mut AnyObject = msg_send![self.0, copy];
            MPSGraphStencilOpDescriptor(desc)
        }
    }
}

/// Stencil operations for MPSGraph
impl MPSGraph {
    /// Creates a stencil operation and returns the result tensor.
    ///
    /// Performs a weighted reduction operation (See `MPSGraphReductionMode`) on the last 4 dimensions of the `source`
    /// over the window determined by `weights`, according to the value defined in `descriptor`.
    /// The operation can be represented as:
    /// 
    /// `y[i] = reduction{j in w} ( x[i + j] * w[j] )`
    ///
    /// # Arguments
    ///
    /// * `source` - The tensor containing the source data. Must be of rank 4 or greater.
    /// * `weights` - A 4-D tensor containing the weights data.
    /// * `descriptor` - The descriptor object that specifies the parameters for the stencil operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object.
    pub fn stencil(
        &self,
        source:  &MPSGraphTensor,
        weights:  &MPSGraphTensor,
        descriptor:  &MPSGraphStencilOpDescriptor,
        name:  Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, stencilWithSourceTensor: source.0,
                weightsTensor: weights.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}

