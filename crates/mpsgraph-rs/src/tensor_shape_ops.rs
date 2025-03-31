use crate::core::create_ns_array_from_i64_slice;
use crate::core::AsRawObject;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use objc2_foundation::NSString;
use std::ptr;

/// Tensor shape operations for MPSGraph
impl MPSGraph {
    /// Creates a reshape operation
    pub fn reshape(&self, x: &MPSGraphTensor, shape: &[i64], name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let shape_array = create_ns_array_from_i64_slice(shape);

            let tensor: *mut AnyObject = msg_send![self.0, reshapeTensor: x.0,
                withShape: shape_array,
                name: name_obj
            ];

            objc2::ffi::objc_release(shape_array as *mut _);

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a flatten2D operation
    pub fn flatten2d(&self, x: &MPSGraphTensor, axis: i64, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, flatten2DTensor: x.0,
                axis: axis,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a broadcast operation
    pub fn broadcast(
        &self,
        x: &MPSGraphTensor,
        shape: &[i64],
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let shape_array = create_ns_array_from_i64_slice(shape);

            let tensor: *mut AnyObject = msg_send![self.0, broadcastTensor: x.0,
                toShape: shape_array,
                name: name_obj
            ];

            objc2::ffi::objc_release(shape_array as *mut _);

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a shape-of operation
    pub fn shape_of(&self, x: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, shapeOfTensor: x.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a cast operation
    pub fn cast(
        &self,
        x: &MPSGraphTensor,
        data_type: crate::core::MPSDataType,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, castTensor: x.0,
                toType: data_type as u32,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a stack operation
    pub fn stack(
        &self,
        tensors: &[MPSGraphTensor],
        axis: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            // Create array of raw tensor pointers
            let raw_tensors: Vec<*mut AnyObject> = tensors.iter().map(|t| t.0).collect();

            // Create NSArray of tensor pointers
            let tensor_array = crate::core::create_ns_array_from_pointers(&raw_tensors);

            let tensor: *mut AnyObject = msg_send![self.0, stackTensors: tensor_array,
                axis: axis,
                name: name_obj
            ];

            objc2::ffi::objc_release(tensor_array as *mut _);

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a split operation
    pub fn split(
        &self,
        x: &MPSGraphTensor,
        num_splits: i64,
        axis: i64,
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0, splitTensor: x.0,
                numSplits: num_splits,
                axis: axis,
                name: name_obj
            ];

            // Convert NSArray to Vec<MPSGraphTensor>
            let count: usize = msg_send![result, count];
            let mut tensors = Vec::with_capacity(count);

            for i in 0..count {
                let tensor_obj: *mut AnyObject = msg_send![result, objectAtIndex: i];
                objc2::ffi::objc_retain(tensor_obj as *mut _);
                tensors.push(MPSGraphTensor(tensor_obj));
            }

            objc2::ffi::objc_release(result as *mut _);

            tensors
        }
    }

    /// Creates a squeeze operation
    pub fn squeeze(&self, x: &MPSGraphTensor, axes: &[i64], name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let axes_array = create_ns_array_from_i64_slice(axes);

            let tensor: *mut AnyObject = msg_send![self.0, squeezeTensor: x.0,
                axes: axes_array,
                name: name_obj
            ];

            objc2::ffi::objc_release(axes_array as *mut _);

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates an expand_dims operation
    pub fn expand_dims(
        &self,
        x: &MPSGraphTensor,
        axes: &[i64],
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let axes_array = create_ns_array_from_i64_slice(axes);

            let tensor: *mut AnyObject = msg_send![self.0, expandDimsTensor: x.0,
                axes: axes_array,
                name: name_obj
            ];

            objc2::ffi::objc_release(axes_array as *mut _);

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a tile operation
    pub fn tile(
        &self,
        x: &MPSGraphTensor,
        multiples: &[i64],
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let multiples_array = create_ns_array_from_i64_slice(multiples);

            let tensor: *mut AnyObject = msg_send![self.0, tileTensor: x.0,
                withMultiples: multiples_array,
                name: name_obj
            ];

            objc2::ffi::objc_release(multiples_array as *mut _);

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a pad operation
    pub fn pad(
        &self,
        x: &MPSGraphTensor,
        padding: &[i64],
        constant: f32,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let padding_array = create_ns_array_from_i64_slice(padding);

            let tensor: *mut AnyObject = msg_send![self.0, padTensor: x.0,
                paddings: padding_array,
                constantValue: constant as f64,
                name: name_obj
            ];

            objc2::ffi::objc_release(padding_array as *mut _);

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a space-to-depth operation
    pub fn space_to_depth(
        &self,
        x: &MPSGraphTensor,
        block_size: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, spaceToDepthWithTensor: x.0,
                blockSize: block_size,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a depth-to-space operation
    pub fn depth_to_space(
        &self,
        x: &MPSGraphTensor,
        block_size: i64,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, depthToSpaceWithTensor: x.0,
                blockSize: block_size,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a reverse operation
    pub fn reverse(&self, x: &MPSGraphTensor, axes: &[i64], name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => ptr::null_mut(),
            };

            let axes_array = create_ns_array_from_i64_slice(axes);

            let tensor: *mut AnyObject = msg_send![self.0, reverseTensor: x.0,
                axes: axes_array,
                name: name_obj
            ];

            objc2::ffi::objc_release(axes_array as *mut _);

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }
}
