use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

use super::Convolution2DOpDescriptor;
use crate::{Graph, ShapeOrTensor, Tensor, ns_number_array_from_slice};

impl Graph {
    // ----- Forward ----------------------------------------------------------
    /// Forward 2-D convolution-transpose (a.k.a. *de-convolution*).
    ///
    /// This op is equivalent to the data-gradient of a forward convolution.
    /// A stride of `s` upsamples each spatial dimension by `s`.
    ///
    /// The output shape may be ambiguous—`output_shape` resolves the correct
    /// spatial size when multiple are mathematically valid.
    ///
    /// # Arguments
    ///
    /// * `source` – Input tensor.
    /// * `weights` – Weights tensor.
    /// * `output_shape` – Desired result shape (slice or tensor).
    /// * `descriptor` – Forward-pass convolution descriptor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing the convolution-transpose result.
    pub fn convolution_transpose_2d<'a>(
        &self,
        source: &Tensor,
        weights: &Tensor,
        output_shape: ShapeOrTensor<'a>,
        descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match output_shape {
                ShapeOrTensor::Shape(shape) => {
                    msg_send![self,
                        convolutionTranspose2DWithSourceTensor: source,
                        weightsTensor: weights,
                        outputShape: &*ns_number_array_from_slice(shape),
                        descriptor: descriptor,
                        name: name.map(NSString::from_str).as_deref(),
                    ]
                }
                ShapeOrTensor::Tensor(shape_tensor) => {
                    msg_send![self,
                        convolutionTranspose2DWithSourceTensor: source,
                        weightsTensor: weights,
                        outputShapeTensor: shape_tensor,
                        descriptor: descriptor,
                        name: name.map(NSString::from_str).as_deref(),
                    ]
                }
            }
        }
    }

    // The dedicated tensor-output-shape variant is now redundant and has been removed.

    // ----- Gradients --------------------------------------------------------
    /// Gradient w.r.t. the **source** tensor of convolution-transpose.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient_tensor` – Incoming gradient (`dL/dR`).
    /// * `weights_tensor` – Weights tensor from the forward pass.
    /// * `output_shape` – Shape of the original source tensor.
    /// * `forward_convolution_descriptor` – Descriptor used in the forward op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing `dL/dS`.
    pub fn convolution_transpose_2d_data_gradient<'a>(
        &self,
        incoming_gradient_tensor: &Tensor,
        weights_tensor: &Tensor,
        output_shape: ShapeOrTensor<'a>,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match output_shape {
                ShapeOrTensor::Shape(shape) => {
                    msg_send![self,
                        convolutionTranspose2DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        weightsTensor: weights_tensor,
                        outputShape: &*ns_number_array_from_slice(shape),
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
                ShapeOrTensor::Tensor(shape_tensor) => {
                    msg_send![self,
                        convolutionTranspose2DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        weightsTensor: weights_tensor,
                        outputShapeTensor: shape_tensor,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
            }
        }
    }

    // Tensor-only overload removed.

    /// Gradient w.r.t. the **weights** tensor of convolution-transpose.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient_tensor` – Incoming gradient.
    /// * `source_tensor` – Source tensor from the forward pass.
    /// * `output_shape` – Shape of the weights tensor.
    /// * `forward_convolution_descriptor` – Descriptor used in the forward op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing `dL/dW`.
    pub fn convolution_transpose_2d_weights_gradient<'a>(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        output_shape: ShapeOrTensor<'a>,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match output_shape {
                ShapeOrTensor::Shape(shape) => {
                    msg_send![self,
                        convolutionTranspose2DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        sourceTensor: source_tensor,
                        outputShape: &*ns_number_array_from_slice(shape),
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
                ShapeOrTensor::Tensor(shape_tensor) => {
                    msg_send![self,
                        convolutionTranspose2DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        sourceTensor: source_tensor,
                        outputShapeTensor: shape_tensor,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
            }
        }
    }
}
