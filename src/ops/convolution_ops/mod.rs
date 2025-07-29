use crate::{Graph, ShapeOrTensor, Tensor, ns_number_array_from_slice};
use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

mod convolution_2d_op_descriptor;
mod convolution_3d_op_descriptor;

pub use convolution_2d_op_descriptor::Convolution2DOpDescriptor;
pub use convolution_3d_op_descriptor::Convolution3DOpDescriptor;

impl Graph {
    /// Creates a 2-D (forward) convolution operation.
    ///
    /// Corresponds to Objective-C selector
    /// `convolution2DWithSourceTensor:weightsTensor:descriptor:name:`.
    ///
    /// # Arguments
    ///
    /// * `source_tensor` – Rank-4 source tensor (layout defined by
    ///   `descriptor.data_layout`).
    /// * `weights_tensor` – Rank-4 weights tensor (layout defined by
    ///   `descriptor.weights_layout`).
    /// * `descriptor` – Strides, dilation rates, padding, and layout
    ///   information.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A `Retained<Tensor>` containing the convolution result.
    pub fn convolution_2d(
        &self,
        source_tensor: &Tensor,
        weights_tensor: &Tensor,
        descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                convolution2DWithSourceTensor: source_tensor,
                weightsTensor: weights_tensor,
                descriptor: descriptor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Creates a 2-D convolution gradient operation with respect to the
    /// *source* tensor.
    ///
    /// Computes `dL/dS = dL/dR · dR/dS`, where `R` is the forward-pass result
    /// and `incoming_gradient_tensor` supplies `dL/dR`.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient_tensor` – Loss gradient with respect to the forward
    ///   result.
    /// * `weights_tensor` – Weights tensor from the forward pass.
    /// * `output_shape` – Shape of the forward-pass source tensor (static
    ///   slice or dynamic tensor via [`ShapeOrTensor`]).
    /// * `forward_convolution_descriptor` – Descriptor used in the forward op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A `Retained<Tensor>` containing the data-gradient result.
    pub fn convolution_2d_data_gradient<'a>(
        &self,
        incoming_gradient_tensor: &Tensor,
        weights_tensor: &Tensor,
        output_shape: ShapeOrTensor<'a>,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match output_shape {
                ShapeOrTensor::Shape(output_shape) => {
                    msg_send![
                        self,
                        convolution2DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        weightsTensor: weights_tensor,
                        outputShape: &*ns_number_array_from_slice(output_shape),
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
                ShapeOrTensor::Tensor(output_shape_tensor) => {
                    msg_send![
                        self,
                        convolution2DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        weightsTensor: weights_tensor,
                        outputShapeTensor: output_shape_tensor,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
            }
        }
    }

    /// Creates a 2-D convolution gradient operation with respect to the
    /// *weights* tensor.
    ///
    /// Computes `dL/dW = dL/dR · dR/dW`, where `R` is the forward convolution
    /// result.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient_tensor` – Loss gradient with respect to the forward
    ///   result.
    /// * `source_tensor` – Source tensor from the forward pass.
    /// * `output_shape` – Shape of the forward-pass weights tensor (static or
    ///   dynamic via [`ShapeOrTensor`]).
    /// * `forward_convolution_descriptor` – Descriptor used in the forward op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A `Retained<Tensor>` containing the weights-gradient result.
    pub fn convolution_2d_weights_gradient<'a>(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        output_shape: ShapeOrTensor<'a>,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match output_shape {
                ShapeOrTensor::Shape(output_shape) => {
                    msg_send![
                        self,
                        convolution2DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        sourceTensor: source_tensor,
                        outputShape: &*ns_number_array_from_slice(output_shape),
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
                ShapeOrTensor::Tensor(output_shape_tensor) => {
                    msg_send![
                        self,
                        convolution2DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        sourceTensor: source_tensor,
                        outputShapeTensor: output_shape_tensor,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
            }
        }
    }

    /// Creates a 3-D (forward) convolution operation. Parameter semantics
    /// mirror the 2-D variant.
    ///
    /// # Arguments
    ///
    /// * `source_tensor` – Rank-5 source tensor.
    /// * `weights_tensor` – Rank-5 weights tensor.
    /// * `descriptor` – Strides, dilation rates, padding, and layout
    ///   information.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A `Retained<Tensor>` containing the convolution result.
    pub fn convolution_3d<'a>(
        &self,
        source_tensor: &Tensor,
        weights_tensor: &Tensor,
        descriptor: &Convolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                convolution3DWithSourceTensor: source_tensor,
                weightsTensor: weights_tensor,
                descriptor: descriptor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Creates a 3-D convolution gradient operation with respect to the
    /// *source* tensor.
    ///
    /// Computes `dL/dS = dL/dR · dR/dS`, analogous to the 2-D variant.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient_tensor` – Loss gradient with respect to the forward
    ///   result.
    /// * `weights_tensor` – Weights tensor from the forward pass.
    /// * `output_shape` – Forward-pass source shape (static slice or dynamic
    ///   tensor).
    /// * `forward_convolution_descriptor` – Descriptor used in the forward op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A `Retained<Tensor>` containing the data-gradient result.
    pub fn convolution_3d_data_gradient<'a>(
        &self,
        incoming_gradient_tensor: &Tensor,
        weights_tensor: &Tensor,
        output_shape: ShapeOrTensor<'a>,
        forward_convolution_descriptor: &Convolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match output_shape {
                ShapeOrTensor::Shape(output_shape) => {
                    msg_send![
                        self,
                        convolution3DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        weightsTensor: weights_tensor,
                        outputShape: &*ns_number_array_from_slice(output_shape),
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
                ShapeOrTensor::Tensor(output_shape_tensor) => {
                    msg_send![
                        self,
                        convolution3DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        weightsTensor: weights_tensor,
                        outputShapeTensor: output_shape_tensor,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
            }
        }
    }

    /// Creates a 3-D convolution gradient operation with respect to the
    /// *weights* tensor.
    ///
    /// Computes `dL/dW = dL/dR · dR/dW`, analogous to the 2-D variant.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient_tensor` – Loss gradient with respect to the forward
    ///   result.
    /// * `source_tensor` – Source tensor from the forward pass.
    /// * `output_shape` – Forward-pass weights shape (static slice or dynamic
    ///   tensor).
    /// * `forward_convolution_descriptor` – Descriptor used in the forward op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A `Retained<Tensor>` containing the weights-gradient result.
    pub fn convolution_3d_weights_gradient<'a>(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        output_shape: ShapeOrTensor<'a>,
        forward_convolution_descriptor: &Convolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match output_shape {
                ShapeOrTensor::Shape(output_shape) => {
                    msg_send![
                        self,
                        convolution3DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        sourceTensor: source_tensor,
                        outputShape: &*ns_number_array_from_slice(output_shape),
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
                ShapeOrTensor::Tensor(output_shape_tensor) => {
                    msg_send![
                        self,
                        convolution3DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        sourceTensor: source_tensor,
                        outputShapeTensor: output_shape_tensor,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
            }
        }
    }
}
