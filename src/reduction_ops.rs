use crate::core::{create_ns_array_from_i64_slice, AsRawObject};
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use objc2_foundation::NSString;
use std::ptr;

/// Reduction operations for MPSGraph
impl MPSGraph {
    /// Creates a reduction sum operation along a single axis
    pub fn reduction_sum_with_tensor_axis(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionSumWithTensor: tensor.0,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction sum operation along multiple axes
    pub fn reduction_sum_with_tensor_axes(
        &self,
        tensor: &MPSGraphTensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let axes_array = match axes {
                Some(a) => create_ns_array_from_i64_slice(a),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionSumWithTensor: tensor.0,
                axes: axes_array,
                name: name_obj
            ];

            if !axes_array.is_null() {
                objc2::ffi::objc_release(axes_array as *mut _);
            }

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction maximum operation along a single axis
    pub fn reduction_maximum_with_tensor_axis(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionMaximumWithTensor: tensor.0,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction maximum operation along multiple axes
    pub fn reduction_maximum_with_tensor_axes(
        &self,
        tensor: &MPSGraphTensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let axes_array = match axes {
                Some(a) => create_ns_array_from_i64_slice(a),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionMaximumWithTensor: tensor.0,
                axes: axes_array,
                name: name_obj
            ];

            if !axes_array.is_null() {
                objc2::ffi::objc_release(axes_array as *mut _);
            }

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction minimum operation along a single axis
    pub fn reduction_minimum_with_tensor_axis(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionMinimumWithTensor: tensor.0,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction minimum operation along multiple axes
    pub fn reduction_minimum_with_tensor_axes(
        &self,
        tensor: &MPSGraphTensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let axes_array = match axes {
                Some(a) => create_ns_array_from_i64_slice(a),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionMinimumWithTensor: tensor.0,
                axes: axes_array,
                name: name_obj
            ];

            if !axes_array.is_null() {
                objc2::ffi::objc_release(axes_array as *mut _);
            }

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction product operation along a single axis
    pub fn reduction_product_with_tensor_axis(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionProductWithTensor: tensor.0,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction product operation along multiple axes
    pub fn reduction_product_with_tensor_axes(
        &self,
        tensor: &MPSGraphTensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let axes_array = match axes {
                Some(a) => create_ns_array_from_i64_slice(a),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionProductWithTensor: tensor.0,
                axes: axes_array,
                name: name_obj
            ];

            if !axes_array.is_null() {
                objc2::ffi::objc_release(axes_array as *mut _);
            }

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction maximum propagate NaN operation along a single axis
    pub fn reduction_maximum_propagate_nan_with_tensor_axis(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionMaximumWithPropagateNaNWithTensor: tensor.0,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction maximum propagate NaN operation along multiple axes
    pub fn reduction_maximum_propagate_nan_with_tensor_axes(
        &self,
        tensor: &MPSGraphTensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let axes_array = match axes {
                Some(a) => create_ns_array_from_i64_slice(a),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionMaximumWithPropagateNaNWithTensor: tensor.0,
                axes: axes_array,
                name: name_obj
            ];

            if !axes_array.is_null() {
                objc2::ffi::objc_release(axes_array as *mut _);
            }

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction minimum propagate NaN operation along a single axis
    pub fn reduction_minimum_propagate_nan_with_tensor_axis(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionMinimumWithPropagateNaNWithTensor: tensor.0,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction minimum propagate NaN operation along multiple axes
    pub fn reduction_minimum_propagate_nan_with_tensor_axes(
        &self,
        tensor: &MPSGraphTensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let axes_array = match axes {
                Some(a) => create_ns_array_from_i64_slice(a),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionMinimumWithPropagateNaNWithTensor: tensor.0,
                axes: axes_array,
                name: name_obj
            ];

            if !axes_array.is_null() {
                objc2::ffi::objc_release(axes_array as *mut _);
            }

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction AND operation along a single axis
    pub fn reduction_and_with_tensor_axis(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionANDWithTensor: tensor.0,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction AND operation along multiple axes
    pub fn reduction_and_with_tensor_axes(
        &self,
        tensor: &MPSGraphTensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let axes_array = match axes {
                Some(a) => create_ns_array_from_i64_slice(a),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionANDWithTensor: tensor.0,
                axes: axes_array,
                name: name_obj
            ];

            if !axes_array.is_null() {
                objc2::ffi::objc_release(axes_array as *mut _);
            }

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction OR operation along a single axis
    pub fn reduction_or_with_tensor_axis(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionORWithTensor: tensor.0,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction OR operation along multiple axes
    pub fn reduction_or_with_tensor_axes(
        &self,
        tensor: &MPSGraphTensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let axes_array = match axes {
                Some(a) => create_ns_array_from_i64_slice(a),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionORWithTensor: tensor.0,
                axes: axes_array,
                name: name_obj
            ];

            if !axes_array.is_null() {
                objc2::ffi::objc_release(axes_array as *mut _);
            }

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction XOR operation along a single axis
    pub fn reduction_xor_with_tensor_axis(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionXORWithTensor: tensor.0,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction XOR operation along multiple axes
    pub fn reduction_xor_with_tensor_axes(
        &self,
        tensor: &MPSGraphTensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let axes_array = match axes {
                Some(a) => create_ns_array_from_i64_slice(a),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionXORWithTensor: tensor.0,
                axes: axes_array,
                name: name_obj
            ];

            if !axes_array.is_null() {
                objc2::ffi::objc_release(axes_array as *mut _);
            }

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction argmax operation along a single axis
    pub fn reduction_arg_maximum_with_tensor_axis(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionArgMaximumWithTensor: tensor.0,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }

    /// Creates a reduction argmin operation along a single axis
    pub fn reduction_arg_minimum_with_tensor_axis(
        &self,
        tensor: &MPSGraphTensor,
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0,
                reductionArgMinimumWithTensor: tensor.0,
                axis: axis,
                name: name_obj
            ];

            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
}
