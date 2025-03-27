use objc2::runtime::AnyObject;
use objc2::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, AsRawObject};
use crate::shape::MPSShape;
use crate::convolution_transpose_ops::TensorNamedDataLayout;

/// Descriptor for Image to Column operations
pub struct MPSGraphImToColOpDescriptor(pub(crate) *mut AnyObject);

impl MPSGraphImToColOpDescriptor {
    /// Creates a new descriptor with full parameters for im2col operations
    ///
    /// # Arguments
    ///
    /// * `kernel_width` - The kernel size in width dimension
    /// * `kernel_height` - The kernel size in height dimension
    /// * `stride_in_x` - The stride in width dimension
    /// * `stride_in_y` - The stride in height dimension
    /// * `dilation_rate_in_x` - The dilation in width dimension
    /// * `dilation_rate_in_y` - The dilation in height dimension
    /// * `padding_left` - The padding in width dimension on the left side
    /// * `padding_right` - The padding in width dimension on the right side
    /// * `padding_top` - The padding in height dimension at the top
    /// * `padding_bottom` - The padding in height dimension at the bottom
    /// * `data_layout` - The layout of source or output tensor
    pub fn descriptor_with_kernel_dimensions(
        kernel_width:  usize,
        kernel_height:  usize,
        stride_in_x:  usize,
        stride_in_y:  usize,
        dilation_rate_in_x:  usize,
        dilation_rate_in_y:  usize,
        padding_left:  usize,
        padding_right:  usize,
        padding_top:  usize,
        padding_bottom:  usize,
        data_layout:  TensorNamedDataLayout,
    ) -> Self {
        unsafe {
            // Get the class, unwrap it, then use it in msg_send
            let class_name = c"MPSGraphImToColOpDescriptor";
            if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                let descriptor: *mut AnyObject = msg_send![
                    cls, descriptorWithKernelWidth: kernel_width,
                    kernelHeight: kernel_height,
                    strideInX: stride_in_x,
                    strideInY: stride_in_y,
                    dilationRateInX: dilation_rate_in_x,
                    dilationRateInY: dilation_rate_in_y,
                    paddingLeft: padding_left,
                    paddingRight: padding_right,
                    paddingTop: padding_top,
                    paddingBottom: padding_bottom,
                    dataLayout: data_layout as u64
                ];
                let descriptor = objc2::ffi::objc_retain(descriptor as *mut _);
                MPSGraphImToColOpDescriptor(descriptor)
            } else {
                // Fall back to creating an empty object if class not found
                let empty_obj: *mut AnyObject = std::ptr::null_mut();
                MPSGraphImToColOpDescriptor(empty_obj)
            }
        }
    }
    
    /// Creates a new descriptor with a simpler set of parameters for im2col operations
    ///
    /// # Arguments
    ///
    /// * `kernel_width` - The kernel size in width dimension
    /// * `kernel_height` - The kernel size in height dimension
    /// * `stride_in_x` - The stride in width dimension
    /// * `stride_in_y` - The stride in height dimension
    /// * `dilation_rate_in_x` - The dilation in width dimension
    /// * `dilation_rate_in_y` - The dilation in height dimension
    /// * `data_layout` - The layout of source or output tensor
    pub fn descriptor_with_kernel_dimensions_simple(
        kernel_width:  usize,
        kernel_height:  usize,
        stride_in_x:  usize,
        stride_in_y:  usize,
        dilation_rate_in_x:  usize,
        dilation_rate_in_y:  usize,
        data_layout:  TensorNamedDataLayout,
    ) -> Self {
        unsafe {
            // Get the class, unwrap it, then use it in msg_send
            let class_name = c"MPSGraphImToColOpDescriptor";
            if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                let descriptor: *mut AnyObject = msg_send![
                    cls, descriptorWithKernelWidth: kernel_width,
                    kernelHeight: kernel_height,
                    strideInX: stride_in_x,
                    strideInY: stride_in_y,
                    dilationRateInX: dilation_rate_in_x,
                    dilationRateInY: dilation_rate_in_y,
                    dataLayout: data_layout as u64
                ];
                let descriptor = objc2::ffi::objc_retain(descriptor as *mut _);
                MPSGraphImToColOpDescriptor(descriptor)
            } else {
                // Fall back to creating an empty object if class not found
                let empty_obj: *mut AnyObject = std::ptr::null_mut();
                MPSGraphImToColOpDescriptor(empty_obj)
            }
        }
    }
    
    /// For backward compatibility
    pub fn new(
        kernel_width:  usize,
        kernel_height:  usize,
        stride_in_x:  usize,
        stride_in_y:  usize,
        dilation_rate_in_x:  usize,
        dilation_rate_in_y:  usize,
        padding_left:  usize,
        padding_right:  usize,
        padding_top:  usize,
        padding_bottom:  usize,
        data_layout:  TensorNamedDataLayout,
    ) -> Self {
        Self::descriptor_with_kernel_dimensions(
            kernel_width,
            kernel_height,
            stride_in_x,
            stride_in_y,
            dilation_rate_in_x,
            dilation_rate_in_y,
            padding_left,
            padding_right,
            padding_top,
            padding_bottom,
            data_layout,
        )
    }
    
    /// For backward compatibility
    pub fn new_simple(
        kernel_width:  usize,
        kernel_height:  usize,
        stride_in_x:  usize,
        stride_in_y:  usize,
        dilation_rate_in_x:  usize,
        dilation_rate_in_y:  usize,
        data_layout:  TensorNamedDataLayout,
    ) -> Self {
        Self::descriptor_with_kernel_dimensions_simple(
            kernel_width,
            kernel_height,
            stride_in_x,
            stride_in_y,
            dilation_rate_in_x,
            dilation_rate_in_y,
            data_layout,
        )
    }
    
    /// Sets the explicit padding values for the descriptor
    /// 
    /// # Arguments
    ///
    /// * `padding_left` - Padding on the left side
    /// * `padding_right` - Padding on the right side
    /// * `padding_top` - Padding on the top side
    /// * `padding_bottom` - Padding on the bottom side
    ///
    /// # Returns
    ///
    /// Self reference for chaining
    pub fn set_explicit_padding(
        &self, 
        padding_left: usize, 
        padding_right: usize, 
        padding_top: usize, 
        padding_bottom: usize
    ) -> &Self {
        unsafe {
            let _: () = msg_send![
                self.0, 
                setExplicitPaddingWithPaddingLeft: padding_left, 
                paddingRight: padding_right, 
                paddingTop: padding_top, 
                paddingBottom: padding_bottom
            ];
        }
        self
    }
}

impl Drop for MPSGraphImToColOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            objc2::ffi::objc_release(self.0 as *mut _);
        }
    }
}

impl Clone for MPSGraphImToColOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let desc: *mut AnyObject = msg_send![self.0, copy];
            MPSGraphImToColOpDescriptor(desc)
        }
    }
}

/// ImToCol operations for MPSGraph
impl MPSGraph {
    /// Creates an image to column operation that converts a 3D or 4D input tensor to a matrix.
    ///
    /// This operation performs an im2col transformation, which is used in convolution operations.
    /// It extracts patches from the input tensor and arranges them as columns of a matrix.
    ///
    /// # Arguments
    ///
    /// * `source` - The source data tensor. The layout is specified by the descriptor.
    /// * `descriptor` - The descriptor containing parameters for the operation.
    /// * `name` - Optional name for the operation.
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the matrix result.
    pub fn im_to_col(
        &self,
        source:  &MPSGraphTensor,
        descriptor:  &MPSGraphImToColOpDescriptor,
        name:  Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, imToColWithSourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a column to image operation that is the reverse of image to column.
    ///
    /// # Arguments
    ///
    /// * `source` - The source data tensor containing the matrix.
    /// * `output_shape` - The shape of the output image tensor.
    /// * `descriptor` - The descriptor containing parameters for the operation.
    /// * `name` - Optional name for the operation.
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the image result.
    pub fn col_to_im(
        &self,
        source:  &MPSGraphTensor,
        output_shape:  &MPSShape,
        descriptor:  &MPSGraphImToColOpDescriptor,
        name:  Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, colToImWithSourceTensor: source.0,
                outputShape: output_shape.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }
    
    /// Alias for im_to_col for backward compatibility
    pub fn image_to_column(
        &self,
        source:  &MPSGraphTensor,
        descriptor:  &MPSGraphImToColOpDescriptor,
        name:  Option<&str>,
    ) -> MPSGraphTensor {
        self.im_to_col(source, descriptor, name)
    }
    
    /// Alias for col_to_im for backward compatibility
    pub fn column_to_image(
        &self,
        source:  &MPSGraphTensor,
        output_shape:  &MPSShape,
        descriptor:  &MPSGraphImToColOpDescriptor,
        name:  Option<&str>,
    ) -> MPSGraphTensor {
        self.col_to_im(source, output_shape, descriptor, name)
    }
}