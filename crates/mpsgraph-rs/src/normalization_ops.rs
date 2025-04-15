use crate::core::AsRawObject;
use crate::graph::Graph;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use objc2_foundation::NSString;

/// Normalization operations for Graph
impl Graph {
    /// Returns the mean of the input tensor along the specified axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axes` - A list of axes over which to perform the reduction
    /// * `name` - An optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    pub fn mean(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        // Convert the axes to NSArray of NSNumbers
        let axes_array = crate::core::create_ns_array_from_i64_slice(axes);

        unsafe {
            let result: *mut AnyObject = msg_send![self.0, meanOfTensor: tensor.0,
                axes: axes_array,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Returns the variance of the input tensor along the specified axes when the mean has been precomputed.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `mean_tensor` - The precomputed mean tensor
    /// * `axes` - A list of axes over which to perform the reduction
    /// * `name` - An optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    pub fn variance_with_mean(
        &self,
        tensor: &Tensor,
        mean_tensor: &Tensor,
        axes: &[i64],
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        // Convert the axes to NSArray of NSNumbers
        let axes_array = crate::core::create_ns_array_from_i64_slice(axes);

        unsafe {
            let result: *mut AnyObject = msg_send![self.0, varianceOfTensor: tensor.0,
                meanTensor: mean_tensor.0,
                axes: axes_array,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Returns the variance of the input tensor along the specified axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axes` - A list of axes over which to perform the reduction
    /// * `name` - An optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    pub fn variance(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        // Convert the axes to NSArray of NSNumbers
        let axes_array = crate::core::create_ns_array_from_i64_slice(axes);

        unsafe {
            let result: *mut AnyObject = msg_send![self.0, varianceOfTensor: tensor.0,
                axes: axes_array,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Creates a batch normalization operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `mean` - The mean tensor
    /// * `variance` - The variance tensor
    /// * `gamma` - The tensor used to scale the normalized result
    /// * `beta` - The tensor used to bias the normalized result
    /// * `epsilon` - A small value to add to the variance when normalizing the inputs
    /// * `name` - An optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    pub fn normalize(
        &self,
        tensor: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        epsilon: f32,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        let gamma_obj = match gamma {
            Some(g) => g.0,
            None => std::ptr::null_mut(),
        };

        let beta_obj = match beta {
            Some(b) => b.0,
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![self.0, normalizationWithTensor: tensor.0,
                meanTensor: mean.0,
                varianceTensor: variance.0,
                gammaTensor: gamma_obj,
                betaTensor: beta_obj,
                epsilon: epsilon,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Creates a normalization gamma-gradient operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - The incoming original result tensor gradient
    /// * `source` - The original input source in forward direction
    /// * `mean` - The mean tensor
    /// * `variance` - The variance tensor
    /// * `axes` - The axes of normalization
    /// * `epsilon` - A small value to add to the variance when normalizing the inputs
    /// * `name` - An optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    pub fn normalization_gamma_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        axes: &[i64],
        epsilon: f32,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        // Convert the axes to NSArray of NSNumbers
        let axes_array = crate::core::create_ns_array_from_i64_slice(axes);

        unsafe {
            let result: *mut AnyObject = msg_send![self.0, normalizationGammaGradientWithIncomingGradientTensor: incoming_gradient.0,
                sourceTensor: source.0,
                meanTensor: mean.0,
                varianceTensor: variance.0,
                reductionAxes: axes_array,
                epsilon: epsilon,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Creates a normalization beta-gradient operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - The incoming original result tensor gradient
    /// * `source` - The original input source in forward direction
    /// * `axes` - The axes of normalization
    /// * `name` - An optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    pub fn normalization_beta_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        axes: &[i64],
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        // Convert the axes to NSArray of NSNumbers
        let axes_array = crate::core::create_ns_array_from_i64_slice(axes);

        unsafe {
            let result: *mut AnyObject = msg_send![self.0, normalizationBetaGradientWithIncomingGradientTensor: incoming_gradient.0,
                sourceTensor: source.0,
                reductionAxes: axes_array,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Creates a normalization input gradient operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - The incoming original result tensor gradient
    /// * `source` - The original input source in forward direction
    /// * `mean` - The mean tensor
    /// * `variance` - The variance tensor
    /// * `gamma` - The gamma tensor
    /// * `gamma_gradient` - The gamma gradient tensor
    /// * `beta_gradient` - The beta gradient tensor
    /// * `axes` - The axes of normalization
    /// * `epsilon` - A small value to add to the variance when normalizing the inputs
    /// * `name` - An optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    pub fn normalization_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        gamma: Option<&Tensor>,
        gamma_gradient: Option<&Tensor>,
        beta_gradient: Option<&Tensor>,
        axes: &[i64],
        epsilon: f32,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        let gamma_obj = match gamma {
            Some(g) => g.0,
            None => std::ptr::null_mut(),
        };

        let gamma_gradient_obj = match gamma_gradient {
            Some(g) => g.0,
            None => std::ptr::null_mut(),
        };

        let beta_gradient_obj = match beta_gradient {
            Some(b) => b.0,
            None => std::ptr::null_mut(),
        };

        // Convert the axes to NSArray of NSNumbers
        let axes_array = crate::core::create_ns_array_from_i64_slice(axes);

        unsafe {
            let result: *mut AnyObject = msg_send![self.0, normalizationGradientWithIncomingGradientTensor: incoming_gradient.0,
                sourceTensor: source.0,
                meanTensor: mean.0,
                varianceTensor: variance.0,
                gammaTensor: gamma_obj,
                gammaGradientTensor: gamma_gradient_obj,
                betaGradientTensor: beta_gradient_obj,
                reductionAxes: axes_array,
                epsilon: epsilon,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }
}
