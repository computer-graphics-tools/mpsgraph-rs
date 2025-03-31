use crate::core::{create_ns_array_from_i64_slice, AsRawObject, MPSDataType};
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use objc2_foundation::NSString;
use std::ptr;

/// Return indices mode for max pooling operations
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MPSGraphPoolingReturnIndicesMode {
    /// No indices returned
    None = 0,
    /// Returns indices flattened in inner most (last) dimension
    GlobalFlatten1D = 1,
    /// Returns indices flattened in 2 innermost dimensions. eg: HW in NCHW
    GlobalFlatten2D = 2,
    /// Returns indices flattened in 3 innermost dimensions. eg: HWC in NHWC
    GlobalFlatten3D = 3,
    /// Returns indices flattened in 4 innermost dimensions
    GlobalFlatten4D = 4,
    /// Returns indices within pooling window, flattened in inner most dimension
    LocalFlatten1D = 5,
    /// Returns indices within pooling window, flattened in 2 innermost dimensions. eg: HW in NCHW
    LocalFlatten2D = 6,
    /// Returns indices within pooling window, flattened in 3 innermost dimensions. eg: HWC in NHWC
    LocalFlatten3D = 7,
    /// Returns indices within pooling window, flattened in 4 innermost dimensions
    LocalFlatten4D = 8,
}

/// Data layout for 2D tensor operations
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MPSGraphTensorNamedDataLayout {
    /// NCHW layout (batch, channels, height, width)
    NCHW = 0,
    /// NHWC layout (batch, height, width, channels)
    NHWC = 1,
}

/// Padding style for tensor operations
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MPSGraphPaddingStyle {
    /// Explicit padding with specified values
    Explicit = 0,
    /// Same padding (input and output have same dimensions)
    TfSame = 1,
    /// Valid padding (no padding)
    TfValid = 2,
}

/// The descriptor for 2D pooling operations
pub struct MPSGraphPooling2DOpDescriptor(pub(crate) *mut AnyObject);

/// The descriptor for 4D pooling operations
pub struct MPSGraphPooling4DOpDescriptor(pub(crate) *mut AnyObject);

// Implement Send + Sync for thread safety
unsafe impl Send for MPSGraphPooling2DOpDescriptor {}
unsafe impl Sync for MPSGraphPooling2DOpDescriptor {}

unsafe impl Send for MPSGraphPooling4DOpDescriptor {}
unsafe impl Sync for MPSGraphPooling4DOpDescriptor {}

impl Drop for MPSGraphPooling2DOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraphPooling2DOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _);
                MPSGraphPooling2DOpDescriptor(obj)
            } else {
                MPSGraphPooling2DOpDescriptor(ptr::null_mut())
            }
        }
    }
}

impl Drop for MPSGraphPooling4DOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraphPooling4DOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _);
                MPSGraphPooling4DOpDescriptor(obj)
            } else {
                MPSGraphPooling4DOpDescriptor(ptr::null_mut())
            }
        }
    }
}

impl MPSGraphPooling2DOpDescriptor {
    /// Creates a new 2D pooling descriptor with the given parameters
    pub fn new(
        kernel_width: usize,
        kernel_height: usize,
        stride_in_x: usize,
        stride_in_y: usize,
        dilation_rate_in_x: usize,
        dilation_rate_in_y: usize,
        padding_left: usize,
        padding_right: usize,
        padding_top: usize,
        padding_bottom: usize,
        padding_style: MPSGraphPaddingStyle,
        data_layout: MPSGraphTensorNamedDataLayout,
    ) -> Self {
        unsafe {
            let cls = objc2::runtime::AnyClass::get(c"MPSGraphPooling2DOpDescriptor").unwrap();
            let descriptor: *mut AnyObject = msg_send![cls,
                descriptorWithKernelWidth: kernel_width as u64,
                kernelHeight: kernel_height as u64,
                strideInX: stride_in_x as u64,
                strideInY: stride_in_y as u64,
                dilationRateInX: dilation_rate_in_x as u64,
                dilationRateInY: dilation_rate_in_y as u64,
                paddingLeft: padding_left as u64,
                paddingRight: padding_right as u64,
                paddingTop: padding_top as u64,
                paddingBottom: padding_bottom as u64,
                paddingStyle: padding_style as u64,
                dataLayout: data_layout as u64
            ];

            let descriptor = objc2::ffi::objc_retain(descriptor as *mut _);
            MPSGraphPooling2DOpDescriptor(descriptor)
        }
    }

    /// Creates a simplified 2D pooling descriptor
    pub fn new_simple(
        kernel_width: usize,
        kernel_height: usize,
        stride_in_x: usize,
        stride_in_y: usize,
        padding_style: MPSGraphPaddingStyle,
        data_layout: MPSGraphTensorNamedDataLayout,
    ) -> Self {
        unsafe {
            let cls = objc2::runtime::AnyClass::get(c"MPSGraphPooling2DOpDescriptor").unwrap();
            let descriptor: *mut AnyObject = msg_send![cls,
                descriptorWithKernelWidth: kernel_width as u64,
                kernelHeight: kernel_height as u64,
                strideInX: stride_in_x as u64,
                strideInY: stride_in_y as u64,
                paddingStyle: padding_style as u64,
                dataLayout: data_layout as u64
            ];

            let descriptor = objc2::ffi::objc_retain(descriptor as *mut _);
            MPSGraphPooling2DOpDescriptor(descriptor)
        }
    }

    /// Sets explicit padding values and changes padding style to explicit
    pub fn set_explicit_padding(
        &self,
        padding_left: usize,
        padding_right: usize,
        padding_top: usize,
        padding_bottom: usize,
    ) {
        unsafe {
            let _: () = msg_send![self.0,
                setExplicitPaddingWithPaddingLeft: padding_left as u64,
                paddingRight: padding_right as u64,
                paddingTop: padding_top as u64,
                paddingBottom: padding_bottom as u64
            ];
        }
    }

    /// Sets the return indices mode for max pooling operations
    pub fn set_return_indices_mode(&self, mode: MPSGraphPoolingReturnIndicesMode) {
        unsafe {
            let _: () = msg_send![self.0, setReturnIndicesMode: mode as u64];
        }
    }

    /// Sets the data type for returned indices
    pub fn set_return_indices_data_type(&self, data_type: MPSDataType) {
        unsafe {
            let _: () = msg_send![self.0, setReturnIndicesDataType: data_type as u32];
        }
    }

    /// Sets ceil mode for computing output size
    pub fn set_ceil_mode(&self, ceil_mode: bool) {
        unsafe {
            let _: () = msg_send![self.0, setCeilMode: ceil_mode];
        }
    }

    /// Sets whether to include zero padding in average computation
    pub fn set_include_zero_pad_to_average(&self, include: bool) {
        unsafe {
            let _: () = msg_send![self.0, setIncludeZeroPadToAverage: include];
        }
    }
}

impl MPSGraphPooling4DOpDescriptor {
    /// Creates a new 4D pooling descriptor with the given parameters
    pub fn new(
        kernel_sizes: &[usize],
        strides: &[usize],
        dilation_rates: &[usize],
        padding_values: &[usize],
        padding_style: MPSGraphPaddingStyle,
    ) -> Self {
        unsafe {
            // Create NSArrays for parameters
            let kernel_sizes_array = Self::create_number_array(kernel_sizes);
            let strides_array = Self::create_number_array(strides);
            let dilation_rates_array = Self::create_number_array(dilation_rates);
            let padding_values_array = Self::create_number_array(padding_values);

            let cls = objc2::runtime::AnyClass::get(c"MPSGraphPooling4DOpDescriptor").unwrap();
            let descriptor: *mut AnyObject = msg_send![cls,
                descriptorWithKernelSizes: kernel_sizes_array,
                strides: strides_array,
                dilationRates: dilation_rates_array,
                paddingValues: padding_values_array,
                paddingStyle: padding_style as u64
            ];

            // Release NSArrays
            objc2::ffi::objc_release(kernel_sizes_array as *mut _);
            objc2::ffi::objc_release(strides_array as *mut _);
            objc2::ffi::objc_release(dilation_rates_array as *mut _);
            objc2::ffi::objc_release(padding_values_array as *mut _);

            let descriptor = objc2::ffi::objc_retain(descriptor as *mut _);
            MPSGraphPooling4DOpDescriptor(descriptor)
        }
    }

    /// Creates a simplified 4D pooling descriptor
    pub fn new_simple(kernel_sizes: &[usize], padding_style: MPSGraphPaddingStyle) -> Self {
        unsafe {
            // Create NSArray for kernel sizes
            let kernel_sizes_array = Self::create_number_array(kernel_sizes);

            let cls = objc2::runtime::AnyClass::get(c"MPSGraphPooling4DOpDescriptor").unwrap();
            let descriptor: *mut AnyObject = msg_send![cls,
                descriptorWithKernelSizes: kernel_sizes_array,
                paddingStyle: padding_style as u64
            ];

            // Release NSArray
            objc2::ffi::objc_release(kernel_sizes_array as *mut _);

            let descriptor = objc2::ffi::objc_retain(descriptor as *mut _);
            MPSGraphPooling4DOpDescriptor(descriptor)
        }
    }

    /// Sets the return indices mode for max pooling operations
    pub fn set_return_indices_mode(&self, mode: MPSGraphPoolingReturnIndicesMode) {
        unsafe {
            let _: () = msg_send![self.0, setReturnIndicesMode: mode as u64];
        }
    }

    /// Sets the data type for returned indices
    pub fn set_return_indices_data_type(&self, data_type: MPSDataType) {
        unsafe {
            let _: () = msg_send![self.0, setReturnIndicesDataType: data_type as u32];
        }
    }

    /// Sets ceil mode for computing output size
    pub fn set_ceil_mode(&self, ceil_mode: bool) {
        unsafe {
            let _: () = msg_send![self.0, setCeilMode: ceil_mode];
        }
    }

    /// Sets whether to include zero padding in average computation
    pub fn set_include_zero_pad_to_average(&self, include: bool) {
        unsafe {
            let _: () = msg_send![self.0, setIncludeZeroPadToAverage: include];
        }
    }

    /// Helper function to create NSArray of NSNumbers from a slice of usize values
    fn create_number_array(values: &[usize]) -> *mut AnyObject {
        unsafe {
            // Create NSNumber objects for each value using objc2-foundation's NSNumber
            let numbers: Vec<objc2::rc::Retained<objc2_foundation::NSNumber>> = values
                .iter()
                .map(|&value| objc2_foundation::NSNumber::new_u64(value as u64))
                .collect();

            // Convert to slice of references
            let number_refs: Vec<&objc2_foundation::NSNumber> =
                numbers.iter().map(|n| n.as_ref()).collect();

            // Create NSArray from the NSNumber objects
            let array = objc2_foundation::NSArray::from_slice(&number_refs);

            // Get pointer to the array and retain it manually
            let ptr: *mut AnyObject = array.as_ref()
                as *const objc2_foundation::NSArray<objc2_foundation::NSNumber>
                as *mut AnyObject;
            objc2::ffi::objc_retain(ptr as *mut _);

            ptr
        }
    }
}

/// 2D and 4D pooling operations for MPSGraph
impl MPSGraph {
    /// Creates a 2D max pooling operation
    pub fn max_pooling_2d(
        &self,
        source: &MPSGraphTensor,
        descriptor: &MPSGraphPooling2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                maxPooling2DWithSourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a 2D max pooling operation that returns indices
    pub fn max_pooling_2d_return_indices(
        &self,
        source: &MPSGraphTensor,
        descriptor: &MPSGraphPooling2DOpDescriptor,
        name: Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0,
                maxPooling2DReturnIndicesWithSourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            // Get the two result tensors
            let result_array = result as *mut objc2_foundation::NSArray<AnyObject>;
            let result_count: usize = msg_send![result_array, count];

            if result_count != 2 {
                panic!("maxPooling2DReturnIndices should return exactly 2 tensors");
            }

            let pooling_tensor: *mut AnyObject = msg_send![result_array, objectAtIndex: 0u64];
            let indices_tensor: *mut AnyObject = msg_send![result_array, objectAtIndex: 1u64];

            let pooling_tensor = objc2::ffi::objc_retain(pooling_tensor as *mut _);
            let indices_tensor = objc2::ffi::objc_retain(indices_tensor as *mut _);

            // Release the array
            objc2::ffi::objc_release(result as *mut _);

            (
                MPSGraphTensor(pooling_tensor),
                MPSGraphTensor(indices_tensor),
            )
        }
    }

    /// Creates a max pooling gradient operation
    pub fn max_pooling_2d_gradient(
        &self,
        gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        descriptor: &MPSGraphPooling2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                maxPooling2DGradientWithGradientTensor: gradient.0,
                sourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a max pooling gradient operation using indices
    pub fn max_pooling_2d_gradient_with_indices(
        &self,
        gradient: &MPSGraphTensor,
        indices: &MPSGraphTensor,
        output_shape: &[i64],
        descriptor: &MPSGraphPooling2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            // Create MPSShape from the output_shape array
            let shape_array = create_ns_array_from_i64_slice(output_shape);

            let tensor: *mut AnyObject = msg_send![
                self.0,
                maxPooling2DGradientWithGradientTensor: gradient.0,
                indicesTensor: indices.0,
                outputShape: shape_array,
                descriptor: descriptor.0,
                name: name_obj
            ];

            objc2::ffi::objc_release(shape_array as *mut _);

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a max pooling gradient operation using indices tensor
    pub fn max_pooling_2d_gradient_with_indices_tensor(
        &self,
        gradient: &MPSGraphTensor,
        indices: &MPSGraphTensor,
        output_shape_tensor: &MPSGraphTensor,
        descriptor: &MPSGraphPooling2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                maxPooling2DGradientWithGradientTensor: gradient.0,
                indicesTensor: indices.0,
                outputShapeTensor: output_shape_tensor.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a 2D average pooling operation
    pub fn avg_pooling_2d(
        &self,
        source: &MPSGraphTensor,
        descriptor: &MPSGraphPooling2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                avgPooling2DWithSourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates an average pooling gradient operation
    pub fn avg_pooling_2d_gradient(
        &self,
        gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        descriptor: &MPSGraphPooling2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                avgPooling2DGradientWithGradientTensor: gradient.0,
                sourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a 2D L2 norm pooling operation
    pub fn l2_norm_pooling_2d(
        &self,
        source: &MPSGraphTensor,
        descriptor: &MPSGraphPooling2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                avgPooling2DWithSourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates an L2 norm pooling gradient operation
    pub fn l2_norm_pooling_2d_gradient(
        &self,
        gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        descriptor: &MPSGraphPooling2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                avgPooling2DGradientWithGradientTensor: gradient.0,
                sourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a 4D max pooling operation
    pub fn max_pooling_4d(
        &self,
        source: &MPSGraphTensor,
        descriptor: &MPSGraphPooling4DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                maxPooling4DWithSourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a 4D max pooling operation that returns indices
    pub fn max_pooling_4d_return_indices(
        &self,
        source: &MPSGraphTensor,
        descriptor: &MPSGraphPooling4DOpDescriptor,
        name: Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0,
                maxPooling4DReturnIndicesWithSourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            // Get the two result tensors
            let result_array = result as *mut objc2_foundation::NSArray<AnyObject>;
            let result_count: usize = msg_send![result_array, count];

            if result_count != 2 {
                panic!("maxPooling4DReturnIndices should return exactly 2 tensors");
            }

            let pooling_tensor: *mut AnyObject = msg_send![result_array, objectAtIndex: 0];
            let indices_tensor: *mut AnyObject = msg_send![result_array, objectAtIndex: 1];

            let pooling_tensor = objc2::ffi::objc_retain(pooling_tensor as *mut _);
            let indices_tensor = objc2::ffi::objc_retain(indices_tensor as *mut _);

            // Release the array
            objc2::ffi::objc_release(result as *mut _);

            (
                MPSGraphTensor(pooling_tensor),
                MPSGraphTensor(indices_tensor),
            )
        }
    }

    /// Creates a max pooling gradient operation
    pub fn max_pooling_4d_gradient(
        &self,
        gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        descriptor: &MPSGraphPooling4DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                maxPooling4DGradientWithGradientTensor: gradient.0,
                sourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a max pooling gradient operation using indices
    pub fn max_pooling_4d_gradient_with_indices(
        &self,
        gradient: &MPSGraphTensor,
        indices: &MPSGraphTensor,
        output_shape: &[i64],
        descriptor: &MPSGraphPooling4DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            // Create MPSShape from the output_shape array
            let shape_array = create_ns_array_from_i64_slice(output_shape);

            let tensor: *mut AnyObject = msg_send![
                self.0,
                maxPooling4DGradientWithGradientTensor: gradient.0,
                indicesTensor: indices.0,
                outputShape: shape_array,
                descriptor: descriptor.0,
                name: name_obj
            ];

            objc2::ffi::objc_release(shape_array as *mut _);

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a max pooling gradient operation using indices tensor
    pub fn max_pooling_4d_gradient_with_indices_tensor(
        &self,
        gradient: &MPSGraphTensor,
        indices: &MPSGraphTensor,
        output_shape_tensor: &MPSGraphTensor,
        descriptor: &MPSGraphPooling4DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                maxPooling4DGradientWithGradientTensor: gradient.0,
                indicesTensor: indices.0,
                outputShapeTensor: output_shape_tensor.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a 4D average pooling operation
    pub fn avg_pooling_4d(
        &self,
        source: &MPSGraphTensor,
        descriptor: &MPSGraphPooling4DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                avgPooling4DWithSourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates an average pooling gradient operation
    pub fn avg_pooling_4d_gradient(
        &self,
        gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        descriptor: &MPSGraphPooling4DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                avgPooling4DGradientWithGradientTensor: gradient.0,
                sourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates a 4D L2 norm pooling operation
    pub fn l2_norm_pooling_4d(
        &self,
        source: &MPSGraphTensor,
        descriptor: &MPSGraphPooling4DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                L2NormPooling4DWithSourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }

    /// Creates an L2 norm pooling gradient operation
    pub fn l2_norm_pooling_4d_gradient(
        &self,
        gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        descriptor: &MPSGraphPooling4DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0,
                L2NormPooling4DGradientWithGradientTensor: gradient.0,
                sourceTensor: source.0,
                descriptor: descriptor.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            MPSGraphTensor(tensor)
        }
    }
}
