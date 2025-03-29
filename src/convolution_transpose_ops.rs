use objc2::msg_send;
use objc2::runtime::AnyObject;
// In objc2, use false as NO and true as YES
const NO: bool = false;
const YES: bool = true;
use crate::core::{AsRawObject, NSString};
use crate::graph::MPSGraph;
use crate::shape::MPSShape;
use crate::tensor::MPSGraphTensor;

/// Defines the data layout for tensors
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum TensorNamedDataLayout {
    NCHW = 0, // Batch, Channels, Height, Width
    NHWC = 1, // Batch, Height, Width, Channels
    CHWN = 2, // Channels, Height, Width, Batch
    HWC = 3,  // Height, Width, Channels
    CHW = 4,  // Channels, Height, Width
}

/// Defines the padding style for convolution operations
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum PaddingStyle {
    Explicit = 0,
    TfSame = 1,
    TfValid = 2,
}

/// Descriptor for 2D convolution operations
pub struct MPSGraphConvolution2DOpDescriptor(pub(crate) *mut AnyObject);

impl Default for MPSGraphConvolution2DOpDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl MPSGraphConvolution2DOpDescriptor {
    /// Creates a new convolution 2D operation descriptor
    pub fn new() -> Self {
        unsafe {
            let class_name = c"MPSGraphConvolution2DOpDescriptor";
            if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                let descriptor: *mut AnyObject = msg_send![cls, descriptor];
                let descriptor = objc2::ffi::objc_retain(descriptor as *mut _) as *mut AnyObject;
                MPSGraphConvolution2DOpDescriptor(descriptor)
            } else {
                panic!("Class MPSGraphConvolution2DOpDescriptor not found")
            }
        }
    }

    /// Sets the stride in X dimension
    pub fn set_stride_in_x(&self, stride: usize) {
        unsafe {
            let _: () = msg_send![self.0, setStrideInX: stride,];
        }
    }

    /// Sets the stride in Y dimension
    pub fn set_stride_in_y(&self, stride: usize) {
        unsafe {
            let _: () = msg_send![self.0, setStrideInY: stride,];
        }
    }

    /// Sets the dilation rate in X dimension
    pub fn set_dilation_rate_in_x(&self, rate: usize) {
        unsafe {
            let _: () = msg_send![self.0, setDilationRateInX: rate,];
        }
    }

    /// Sets the dilation rate in Y dimension
    pub fn set_dilation_rate_in_y(&self, rate: usize) {
        unsafe {
            let _: () = msg_send![self.0, setDilationRateInY: rate,];
        }
    }

    /// Sets the padding on the left
    pub fn set_padding_left(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingLeft: padding,];
        }
    }

    /// Sets the padding on the right
    pub fn set_padding_right(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingRight: padding,];
        }
    }

    /// Sets the padding on the top
    pub fn set_padding_top(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingTop: padding,];
        }
    }

    /// Sets the padding on the bottom
    pub fn set_padding_bottom(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingBottom: padding,];
        }
    }

    /// Sets the padding style
    pub fn set_padding_style(&self, style: PaddingStyle) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingStyle: style as u64];
        }
    }

    /// Sets the data layout
    pub fn set_data_layout(&self, layout: TensorNamedDataLayout) {
        unsafe {
            let _: () = msg_send![self.0, setDataLayout: layout as u64];
        }
    }

    /// Sets the weights layout
    pub fn set_weights_layout(&self, layout: TensorNamedDataLayout) {
        unsafe {
            let _: () = msg_send![self.0, setWeightsLayout: layout as u64];
        }
    }

    /// Sets the explicit padding values
    pub fn set_explicit_padding(&self, left: usize, right: usize, top: usize, bottom: usize) {
        unsafe {
            let _: () = msg_send![
                self.0, setExplicitPaddingWithPaddingLeft: left,
                paddingRight: right,
                paddingTop: top,
                paddingBottom: bottom,
            ];
        }
    }

    /// Sets whether to use same padding in the TensorFlow style
    pub fn set_use_same_padding(&self, use_same_padding: bool) {
        unsafe {
            let _: () =
                msg_send![self.0, setUseSamePadding: if use_same_padding { YES } else { NO }];
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

impl Clone for MPSGraphConvolution2DOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let desc: *mut AnyObject = msg_send![self.0, copy];
            MPSGraphConvolution2DOpDescriptor(desc)
        }
    }
}

/// Transposed convolution operations for MPSGraph
impl MPSGraph {
    /// Creates a 2D convolution transpose operation and returns the result tensor.
    ///
    /// Convolution Tranpose operation is exactly the same as convolution gradient with respect to input image.
    /// Weights tensor and source tensors are interpreted as they are in convolution data gradient.
    /// Convolution with stride `s` downsamples source tensor by factor `s` in spatial dimensions whereas
    /// convolution tranpose with stride `s` upsamples source tensor by factor `s`.
    ///
    /// # Arguments
    ///
    /// * `source` - Source tensor
    /// * `weights` - Weights tensor
    /// * `output_shape` - Shape of the result tensor
    /// * `descriptor` - Descriptor for the corresponding forward 2D-convolution operation
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the result
    pub fn convolution_transpose_2d(
        &self,
        source: &MPSGraphTensor,
        weights: &MPSGraphTensor,
        output_shape: &MPSShape,
        descriptor: &MPSGraphConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, convolutionTranspose2DWithSourceTensor: source.0,
                weightsTensor: weights.0,
                outputShape: output_shape.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a 2D convolution transpose operation with tensor output shape
    ///
    /// # Arguments
    ///
    /// * `source` - Source tensor
    /// * `weights` - Weights tensor
    /// * `output_shape_tensor` - 1D Int32 or Int64 tensor. Shape of the result tensor
    /// * `descriptor` - Descriptor for the corresponding forward 2D-convolution operation
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the result
    pub fn convolution_transpose_2d_with_tensor_shape(
        &self,
        source: &MPSGraphTensor,
        weights: &MPSGraphTensor,
        output_shape_tensor: &MPSGraphTensor,
        descriptor: &MPSGraphConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, convolutionTranspose2DWithSourceTensor: source.0,
                weightsTensor: weights.0,
                outputShapeTensor: output_shape_tensor.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a convolution transpose gradient operation with respect to the source tensor
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - Incoming gradient tensor
    /// * `weights` - Forward pass weights tensor
    /// * `output_shape` - Shape of the forward pass source tensor
    /// * `forward_convolution_descriptor` - Forward pass op descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the gradient with respect to source
    pub fn convolution_transpose_2d_data_gradient(
        &self,
        incoming_gradient: &MPSGraphTensor,
        weights: &MPSGraphTensor,
        output_shape: &MPSShape,
        forward_convolution_descriptor: &MPSGraphConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, convolutionTranspose2DDataGradientWithIncomingGradientTensor: incoming_gradient.0,
                weightsTensor: weights.0,
                outputShape: output_shape.0,
                forwardConvolutionDescriptor: forward_convolution_descriptor.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a convolution transpose gradient operation with respect to the source tensor (with tensor output shape)
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - Incoming gradient tensor
    /// * `weights` - Forward pass weights tensor
    /// * `output_shape_tensor` - 1D Int32 or Int64 tensor. Shape of the forward pass source tensor
    /// * `forward_convolution_descriptor` - Forward pass op descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the gradient with respect to source
    pub fn convolution_transpose_2d_data_gradient_with_tensor_shape(
        &self,
        incoming_gradient: &MPSGraphTensor,
        weights: &MPSGraphTensor,
        output_shape_tensor: &MPSGraphTensor,
        forward_convolution_descriptor: &MPSGraphConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, convolutionTranspose2DDataGradientWithIncomingGradientTensor: incoming_gradient.0,
                weightsTensor: weights.0,
                outputShapeTensor: output_shape_tensor.0,
                forwardConvolutionDescriptor: forward_convolution_descriptor.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a convolution transpose weights gradient operation
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - Incoming gradient tensor
    /// * `source` - Forward pass source tensor
    /// * `output_shape` - Shape of the forward pass weights tensor
    /// * `forward_convolution_descriptor` - Forward pass op descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the gradient with respect to weights
    pub fn convolution_transpose_2d_weights_gradient(
        &self,
        incoming_gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        output_shape: &MPSShape,
        forward_convolution_descriptor: &MPSGraphConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, convolutionTranspose2DWeightsGradientWithIncomingGradientTensor: incoming_gradient.0,
                sourceTensor: source.0,
                outputShape: output_shape.0,
                forwardConvolutionDescriptor: forward_convolution_descriptor.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a convolution transpose weights gradient operation (with tensor output shape)
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - Incoming gradient tensor
    /// * `source` - Forward pass source tensor
    /// * `output_shape_tensor` - 1D Int32 or Int64 tensor. Shape of the forward pass weights tensor
    /// * `forward_convolution_descriptor` - Forward pass op descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the gradient with respect to weights
    pub fn convolution_transpose_2d_weights_gradient_with_tensor_shape(
        &self,
        incoming_gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        output_shape_tensor: &MPSGraphTensor,
        forward_convolution_descriptor: &MPSGraphConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, convolutionTranspose2DWeightsGradientWithIncomingGradientTensor: incoming_gradient.0,
                sourceTensor: source.0,
                outputShapeTensor: output_shape_tensor.0,
                forwardConvolutionDescriptor: forward_convolution_descriptor.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}
