use crate::core::{AsRawObject, MPSDataType, NSString};
use crate::graph::MPSGraph;
use crate::shape::MPSShape;
use crate::tensor::MPSGraphTensor;
use objc2::runtime::AnyObject;

/// Random distribution types supported by MPSGraph
#[repr(u64)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MPSGraphRandomDistribution {
    /// Uniform distribution, with samples drawn uniformly from [min, max) for float types,
    /// and [min, max] for integer types
    Uniform = 0,
    /// Normal distribution defined by mean and standard deviation
    Normal = 1,
    /// Normal distribution defined by mean and standard deviation, truncated to range [min, max)
    TruncatedNormal = 2,
}

/// Sampling methods for normal distributions
#[repr(u64)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MPSGraphRandomNormalSamplingMethod {
    /// Use inverse erf to convert uniform values to values in the normal distribution
    InverseCDF = 0,
    /// Use Box Muller transform to convert uniform values to values in the normal distribution
    BoxMuller = 1,
}

/// Descriptor for random operations in MPSGraph
pub struct MPSGraphRandomOpDescriptor(pub(crate) *mut AnyObject);

impl MPSGraphRandomOpDescriptor {
    /// Creates a new random operation descriptor with the specified distribution and data type
    pub fn new(distribution: MPSGraphRandomDistribution, data_type: MPSDataType) -> Self {
        unsafe {
            let descriptor: *mut AnyObject = msg_send![
                class!(MPSGraphRandomOpDescriptor), descriptorWithDistribution: distribution as u64,
                dataType: data_type as u64
            ];
            let descriptor = objc2::ffi::objc_retain(descriptor as *mut _) as *mut AnyObject;
            MPSGraphRandomOpDescriptor(descriptor)
        }
    }

    /// Sets the minimum value (for float data types)
    pub fn set_min(&self, min: f32) {
        unsafe {
            let _: () = msg_send![self.0, setMin: min,];
        }
    }

    /// Sets the maximum value (for float data types)
    pub fn set_max(&self, max: f32) {
        unsafe {
            let _: () = msg_send![self.0, setMax: max,];
        }
    }

    /// Sets the minimum integer value (for integer data types)
    pub fn set_min_integer(&self, min: i64) {
        unsafe {
            let _: () = msg_send![self.0, setMinInteger: min,];
        }
    }

    /// Sets the maximum integer value (for integer data types)
    pub fn set_max_integer(&self, max: i64) {
        unsafe {
            let _: () = msg_send![self.0, setMaxInteger: max,];
        }
    }

    /// Sets the mean (for normal distributions)
    pub fn set_mean(&self, mean: f32) {
        unsafe {
            let _: () = msg_send![self.0, setMean: mean,];
        }
    }

    /// Sets the standard deviation (for normal distributions)
    pub fn set_standard_deviation(&self, std_dev: f32) {
        unsafe {
            let _: () = msg_send![self.0, setStandardDeviation: std_dev,];
        }
    }

    /// Sets the sampling method (for normal distributions)
    pub fn set_sampling_method(&self, method: MPSGraphRandomNormalSamplingMethod) {
        unsafe {
            let _: () = msg_send![self.0, setSamplingMethod: method as u64];
        }
    }
}

impl Drop for MPSGraphRandomOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            objc2::ffi::objc_release(self.0 as *mut _);
        }
    }
}

// Enable Send and Sync for MPSGraphRandomOpDescriptor
unsafe impl Send for MPSGraphRandomOpDescriptor {}
unsafe impl Sync for MPSGraphRandomOpDescriptor {}

impl Clone for MPSGraphRandomOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
            MPSGraphRandomOpDescriptor(obj)
        }
    }
}

/// Random operations for MPSGraph
impl MPSGraph {
    /// Creates a tensor representing state using the Philox algorithm with given seed
    pub fn random_philox_state_tensor_with_seed(
        &self,
        seed: usize,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, randomPhiloxStateTensorWithSeed: seed,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a tensor representing state using the Philox algorithm with given counter and key values
    pub fn random_philox_state_tensor_with_counter(
        &self,
        counter_low: usize,
        counter_high: usize,
        key: usize,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, randomPhiloxStateTensorWithCounterLow: counter_low,
                counterHigh: counter_high,
                key: key,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a random tensor with the specified shape and distribution
    pub fn random_tensor(
        &self,
        shape: &[usize],
        descriptor: &MPSGraphRandomOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let shape_obj = MPSShape::from_slice(shape);

            let tensor: *mut AnyObject = msg_send![self.0, randomTensorWithShape: shape_obj.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a random tensor with the specified shape and distribution, using a specific seed
    pub fn random_tensor_with_seed(
        &self,
        shape: &[usize],
        descriptor: &MPSGraphRandomOpDescriptor,
        seed: usize,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let shape_obj = MPSShape::from_slice(shape);

            let tensor: *mut AnyObject = msg_send![self.0, randomTensorWithShape: shape_obj.0,
                descriptor: descriptor.0,
                seed: seed,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a random tensor with specified shape and distribution, using and updating the state tensor
    pub fn random_tensor_with_state(
        &self,
        shape: &[usize],
        descriptor: &MPSGraphRandomOpDescriptor,
        state: &MPSGraphTensor,
        name: Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let shape_obj = MPSShape::from_slice(shape);

            let result: *mut AnyObject = msg_send![self.0, randomTensorWithShape: shape_obj.0,
                descriptor: descriptor.0,
                stateTensor: state.0,
                name: name_obj,
            ];

            // Extract the two tensors from the result array
            let count: usize = msg_send![result, count];
            assert_eq!(
                count, 2,
                "Random tensor with state should return an array of 2 tensors"
            );

            // Get the random tensor and updated state
            let random_tensor: *mut AnyObject = msg_send![result, objectAtIndex: 0];
            let updated_state: *mut AnyObject = msg_send![result, objectAtIndex: 1];

            let random_tensor = objc2::ffi::objc_retain(random_tensor as *mut _) as *mut AnyObject;
            let updated_state = objc2::ffi::objc_retain(updated_state as *mut _) as *mut AnyObject;

            (MPSGraphTensor(random_tensor), MPSGraphTensor(updated_state))
        }
    }

    /// Creates a random uniform tensor with values in range [0.0, 1.0)
    pub fn random_uniform_tensor(&self, shape: &[usize], name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let shape_obj = MPSShape::from_slice(shape);

            let tensor: *mut AnyObject = msg_send![self.0, randomUniformTensorWithShape: shape_obj.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a random uniform tensor with values in range [0.0, 1.0) using a specific seed
    pub fn random_uniform_tensor_with_seed(
        &self,
        shape: &[usize],
        seed: usize,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let shape_obj = MPSShape::from_slice(shape);

            let tensor: *mut AnyObject = msg_send![self.0, randomUniformTensorWithShape: shape_obj.0,
                seed: seed,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a random uniform tensor with values in range [0.0, 1.0), using and updating the state tensor
    pub fn random_uniform_tensor_with_state(
        &self,
        shape: &[usize],
        state: &MPSGraphTensor,
        name: Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let shape_obj = MPSShape::from_slice(shape);

            let result: *mut AnyObject = msg_send![self.0, randomUniformTensorWithShape: shape_obj.0,
                stateTensor: state.0,
                name: name_obj,
            ];

            // Extract the two tensors from the result array
            let count: usize = msg_send![result, count];
            assert_eq!(
                count, 2,
                "Random uniform tensor with state should return an array of 2 tensors"
            );

            // Get the random tensor and updated state
            let random_tensor: *mut AnyObject = msg_send![result, objectAtIndex: 0];
            let updated_state: *mut AnyObject = msg_send![result, objectAtIndex: 1];

            let random_tensor = objc2::ffi::objc_retain(random_tensor as *mut _) as *mut AnyObject;
            let updated_state = objc2::ffi::objc_retain(updated_state as *mut _) as *mut AnyObject;

            (MPSGraphTensor(random_tensor), MPSGraphTensor(updated_state))
        }
    }

    /// Creates a dropout operation which zeros out elements of the input tensor randomly with probability equal to rate
    pub fn dropout(
        &self,
        tensor: &MPSGraphTensor,
        rate: f64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0, dropoutTensor: tensor.0,
                rate: rate,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a dropout operation using a tensor to specify the dropout rate
    pub fn dropout_with_rate_tensor(
        &self,
        tensor: &MPSGraphTensor,
        rate_tensor: &MPSGraphTensor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0, dropoutTensor: tensor.0,
                rateTensor: rate_tensor.0,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
}
