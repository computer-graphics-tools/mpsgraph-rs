use objc2::runtime::AnyObject;
// In objc2, use false as NO and true as YES
const NO: bool = false;
const YES: bool = true;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, MPSShape, AsRawObject};
use crate::convolution_transpose_ops::TensorNamedDataLayout;

/// The resize mode to use for resizing.
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum MPSGraphResizeMode {
    /// Samples the nearest neighbor to the pixel coordinate.
    Nearest = 0,
    /// Samples the 4 neighbors to the pixel coordinate and uses bilinear interpolation.
    Bilinear = 1,
}

/// The rounding mode to use when using nearest resize mode.
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum MPSGraphResizeNearestRoundingMode {
    /// Rounds values to the nearest integer value, with 0.5f offset rounding toward +inf.
    RoundPreferCeil = 0,
    /// Rounds values to the nearest integer value, with 0.5f rounding toward -inf.
    RoundPreferFloor = 1,
    /// Rounds values toward +inf.
    Ceil = 2,
    /// Rounds values toward -inf.
    Floor = 3,
    /// Rounds values to the nearest integer value, with 0.5f rounding toward the closest even value.
    RoundToEven = 4,
    /// Rounds values to the nearest integer value, with 0.5f rounding toward the closest odd value.
    RoundToOdd = 5,
}

/// Resize operations for MPSGraph
impl MPSGraph {
    /// Creates a Resize operation and returns the result tensor.
    /// 
    /// Resamples input images to given size. Result images will be distorted if size is of different aspect ratio.
    /// 
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size: A 2-element shape as [newHeight, newWidth]
    ///   - mode: The resampling mode to use.
    ///   - center_result: Controls if the result image is centered on the input image. When false, the result will have the top left corner aligned.
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners.
    ///   - layout: Specifies what layout the provided tensor is in.
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize(&self,
                  images_tensor:  &MPSGraphTensor,
                  size:  &MPSShape,
                  mode:  MPSGraphResizeMode,
                  center_result:  bool,
                  align_corners:  bool,
                  layout:  TensorNamedDataLayout,
                  name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let center_result_val = if center_result { YES } else { NO };
            let align_corners_val = if align_corners { YES } else { NO };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeTensor: images_tensor.0
                size: size.0
                mode: mode as u64
                centerResult: center_result_val
                alignCorners: align_corners_val
                layout: layout as u64
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize operation using a tensor for size specification and returns the result tensor.
    /// 
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: 1D Int32 or Int64 tensor. A 2-element shape as [newHeight, newWidth]
    ///   - mode: The resampling mode to use.
    ///   - center_result: Controls if the result image is centered on the input image.
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners.
    ///   - layout: Specifies what layout the provided tensor is in.
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_with_size_tensor(&self,
                                  images_tensor:  &MPSGraphTensor,
                                  size_tensor:  &MPSGraphTensor,
                                  mode:  MPSGraphResizeMode,
                                  center_result:  bool,
                                  align_corners:  bool,
                                  layout:  TensorNamedDataLayout,
                                  name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let center_result_val = if center_result { YES } else { NO };
            let align_corners_val = if align_corners { YES } else { NO };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeTensor: images_tensor.0
                sizeTensor: size_tensor.0
                mode: mode as u64
                centerResult: center_result_val
                alignCorners: align_corners_val
                layout: layout as u64
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize operation with nearest neighbor sampling and a specific rounding mode.
    /// 
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: 1D Int32 or Int64 tensor. A 2-element shape as [newHeight, newWidth]
    ///   - nearest_rounding_mode: The rounding mode to use when using nearest resampling.
    ///   - center_result: Controls if the result image is centered on the input image.
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners.
    ///   - layout: Specifies what layout the provided tensor is in.
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_nearest(&self,
                         images_tensor:  &MPSGraphTensor,
                         size_tensor:  &MPSGraphTensor,
                         nearest_rounding_mode:  MPSGraphResizeNearestRoundingMode,
                         center_result:  bool,
                         align_corners:  bool,
                         layout:  TensorNamedDataLayout,
                         name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let center_result_val = if center_result { YES } else { NO };
            let align_corners_val = if align_corners { YES } else { NO };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeNearestWithTensor: images_tensor.0
                sizeTensor: size_tensor.0
                nearestRoundingMode: nearest_rounding_mode as u64
                centerResult: center_result_val
                alignCorners: align_corners_val
                layout: layout as u64
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize operation with bilinear interpolation.
    /// 
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: 1D Int32 or Int64 tensor. A 2-element shape as [newHeight, newWidth]
    ///   - center_result: Controls if the result image is centered on the input image.
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners.
    ///   - layout: Specifies what layout the provided tensor is in.
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_bilinear(&self,
                          images_tensor:  &MPSGraphTensor,
                          size_tensor:  &MPSGraphTensor,
                          center_result:  bool,
                          align_corners:  bool,
                          layout:  TensorNamedDataLayout,
                          name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let center_result_val = if center_result { YES } else { NO };
            let align_corners_val = if align_corners { YES } else { NO };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeBilinearWithTensor: images_tensor.0
                sizeTensor: size_tensor.0
                centerResult: center_result_val
                alignCorners: align_corners_val
                layout: layout as u64
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize operation using explicit scale and offset tensors.
    /// 
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: 1D Int32 or Int64 tensor. A 2-element shape as [newHeight, newWidth]
    ///   - scale_offset_tensor: 1D float tensor. A 4-element shape as [scaleY, scaleX, offsetY, offsetX]
    ///   - mode: The resampling mode to use.
    ///   - layout: Specifies what layout the provided tensor is in.
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_with_scale_offset(&self,
                                   images_tensor:  &MPSGraphTensor,
                                   size_tensor:  &MPSGraphTensor,
                                   scale_offset_tensor:  &MPSGraphTensor,
                                   mode:  MPSGraphResizeMode,
                                   layout:  TensorNamedDataLayout,
                                   name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeTensor: images_tensor.0
                sizeTensor: size_tensor.0
                scaleOffsetTensor: scale_offset_tensor.0
                mode: mode as u64
                layout: layout as u64
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize gradient operation for backpropagation.
    /// 
    /// - Parameters:
    ///   - gradient: Incoming gradient tensor
    ///   - input: Forward pass input tensor
    ///   - mode: The resampling mode to use
    ///   - center_result: Controls if the result image is centered on the input image
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners
    ///   - layout: Specifies what layout the provided tensor is in
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_gradient(&self,
                         gradient:  &MPSGraphTensor,
                         input:  &MPSGraphTensor,
                         mode:  MPSGraphResizeMode,
                         center_result:  bool,
                         align_corners:  bool,
                         layout:  TensorNamedDataLayout,
                         name:  Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let center_result_val = if center_result { YES } else { NO };
            let align_corners_val = if align_corners { YES } else { NO };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeWithGradientTensor: gradient.0
                input: input.0
                mode: mode as u64
                centerResult: center_result_val
                alignCorners: align_corners_val
                layout: layout as u64
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}