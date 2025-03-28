use objc2::runtime::AnyObject;
// In objc2, use false as NO and true as YES
const NO: bool = false;
const YES: bool = true;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, AsRawObject};
use crate::shape::MPSShape;
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
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeTensor: images_tensor.0,
                size: size.0,
                mode: mode as u64,
                centerResult: center_result_val,
                alignCorners: align_corners_val,
                layout: layout as u64,
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
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeTensor: images_tensor.0,
                sizeTensor: size_tensor.0,
                mode: mode as u64,
                centerResult: center_result_val,
                alignCorners: align_corners_val,
                layout: layout as u64,
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize operation using a tensor for size specification and returns the result tensor.
    /// 
    /// This is a rank-agnostic version that works with tensors of any rank (iOS 17+/macOS 14+).
    /// 
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: The target size of the result tensor. 1D Int32 or Int64 tensor of size equal to rank of input.
    ///   - mode: The resampling mode to use.
    ///   - center_result: Controls if the result image is centered on the input image.
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners.
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_rank_agnostic(&self,
                                 images_tensor: &MPSGraphTensor,
                                 size_tensor: &MPSGraphTensor,
                                 mode: MPSGraphResizeMode,
                                 center_result: bool,
                                 align_corners: bool,
                                 name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let center_result_val = if center_result { YES } else { NO };
            let align_corners_val = if align_corners { YES } else { NO };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeTensor: images_tensor.0,
                sizeTensor: size_tensor.0,
                mode: mode as u64,
                centerResult: center_result_val,
                alignCorners: align_corners_val,
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
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeNearestWithTensor: images_tensor.0,
                sizeTensor: size_tensor.0,
                nearestRoundingMode: nearest_rounding_mode as u64,
                centerResult: center_result_val,
                alignCorners: align_corners_val,
                layout: layout as u64,
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize operation with nearest neighbor sampling and a specific rounding mode.
    /// 
    /// This is a rank-agnostic version that works with tensors of any rank (iOS 17+/macOS 14+).
    /// 
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: The target size of the result tensor. 1D Int32 or Int64 tensor of size equal to rank of input.
    ///   - nearest_rounding_mode: The rounding mode to use when using nearest resampling.
    ///   - center_result: Controls if the result image is centered on the input image.
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners.
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_nearest_rank_agnostic(&self,
                                        images_tensor: &MPSGraphTensor,
                                        size_tensor: &MPSGraphTensor,
                                        nearest_rounding_mode: MPSGraphResizeNearestRoundingMode,
                                        center_result: bool,
                                        align_corners: bool,
                                        name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let center_result_val = if center_result { YES } else { NO };
            let align_corners_val = if align_corners { YES } else { NO };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeNearestWithTensor: images_tensor.0,
                sizeTensor: size_tensor.0,
                nearestRoundingMode: nearest_rounding_mode as u64,
                centerResult: center_result_val,
                alignCorners: align_corners_val,
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
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeBilinearWithTensor: images_tensor.0,
                sizeTensor: size_tensor.0,
                centerResult: center_result_val,
                alignCorners: align_corners_val,
                layout: layout as u64,
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize operation with bilinear interpolation.
    /// 
    /// This is a rank-agnostic version that works with tensors of any rank (iOS 17+/macOS 14+).
    /// 
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: The target size of the result tensor. 1D Int32 or Int64 tensor of size equal to rank of input.
    ///   - center_result: Controls if the result image is centered on the input image.
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners.
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_bilinear_rank_agnostic(&self,
                                         images_tensor: &MPSGraphTensor,
                                         size_tensor: &MPSGraphTensor,
                                         center_result: bool,
                                         align_corners: bool,
                                         name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let center_result_val = if center_result { YES } else { NO };
            let align_corners_val = if align_corners { YES } else { NO };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeBilinearWithTensor: images_tensor.0,
                sizeTensor: size_tensor.0,
                centerResult: center_result_val,
                alignCorners: align_corners_val,
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
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeTensor: images_tensor.0,
                sizeTensor: size_tensor.0,
                scaleOffsetTensor: scale_offset_tensor.0,
                mode: mode as u64,
                layout: layout as u64,
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize operation using separate scale and offset tensors.
    ///
    /// Available on iOS 17+/macOS 14+.
    /// 
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: The target size of the result tensor. 1D Int32 or Int64 tensor of size equal to rank of input.
    ///   - scale_tensor: 1D float tensor of size equal to rank of input.
    ///   - offset_tensor: 1D float tensor of size equal to rank of input.
    ///   - mode: The resampling mode to use.
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_with_separate_scale_offset(&self,
                                            images_tensor: &MPSGraphTensor,
                                            size_tensor: &MPSGraphTensor,
                                            scale_tensor: &MPSGraphTensor,
                                            offset_tensor: &MPSGraphTensor,
                                            mode: MPSGraphResizeMode,
                                            name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeTensor: images_tensor.0,
                sizeTensor: size_tensor.0,
                scaleTensor: scale_tensor.0,
                offsetTensor: offset_tensor.0,
                mode: mode as u64,
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize operation with nearest neighbor sampling using separate scale and offset tensors.
    ///
    /// Available on iOS 17+/macOS 14+.
    /// 
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: The target size of the result tensor. 1D Int32 or Int64 tensor of size equal to rank of input.
    ///   - scale_tensor: 1D float tensor of size equal to rank of input.
    ///   - offset_tensor: 1D float tensor of size equal to rank of input.
    ///   - nearest_rounding_mode: The rounding mode to use when using nearest resampling.
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_nearest_with_separate_scale_offset(&self,
                                                   images_tensor: &MPSGraphTensor,
                                                   size_tensor: &MPSGraphTensor,
                                                   scale_tensor: &MPSGraphTensor,
                                                   offset_tensor: &MPSGraphTensor,
                                                   nearest_rounding_mode: MPSGraphResizeNearestRoundingMode,
                                                   name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeNearestWithTensor: images_tensor.0,
                sizeTensor: size_tensor.0,
                scaleTensor: scale_tensor.0,
                offsetTensor: offset_tensor.0,
                nearestRoundingMode: nearest_rounding_mode as u64,
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize operation with bilinear interpolation using separate scale and offset tensors.
    ///
    /// Available on iOS 17+/macOS 14+.
    /// 
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: The target size of the result tensor. 1D Int32 or Int64 tensor of size equal to rank of input.
    ///   - scale_tensor: 1D float tensor of size equal to rank of input.
    ///   - offset_tensor: 1D float tensor of size equal to rank of input.
    ///   - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_bilinear_with_separate_scale_offset(&self,
                                                    images_tensor: &MPSGraphTensor,
                                                    size_tensor: &MPSGraphTensor,
                                                    scale_tensor: &MPSGraphTensor,
                                                    offset_tensor: &MPSGraphTensor,
                                                    name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeBilinearWithTensor: images_tensor.0,
                sizeTensor: size_tensor.0,
                scaleTensor: scale_tensor.0,
                offsetTensor: offset_tensor.0,
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
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeWithGradientTensor: gradient.0,
                input: input.0,
                mode: mode as u64,
                centerResult: center_result_val,
                alignCorners: align_corners_val,
                layout: layout as u64,
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize gradient operation for nearest neighbor sampling with a specific rounding mode.
    /// 
    /// - Parameters:
    ///   - gradient: Incoming gradient tensor
    ///   - input: Forward pass input tensor
    ///   - nearest_rounding_mode: The rounding mode to use when using nearest resampling
    ///   - center_result: Controls if the result image is centered on the input image
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners
    ///   - layout: Specifies what layout the provided tensor is in
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_nearest_gradient(&self,
                                  gradient: &MPSGraphTensor,
                                  input: &MPSGraphTensor,
                                  nearest_rounding_mode: MPSGraphResizeNearestRoundingMode,
                                  center_result: bool,
                                  align_corners: bool,
                                  layout: TensorNamedDataLayout,
                                  name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let center_result_val = if center_result { YES } else { NO };
            let align_corners_val = if align_corners { YES } else { NO };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeNearestWithGradientTensor: gradient.0,
                input: input.0,
                nearestRoundingMode: nearest_rounding_mode as u64,
                centerResult: center_result_val,
                alignCorners: align_corners_val,
                layout: layout as u64,
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize gradient operation for bilinear interpolation.
    /// 
    /// - Parameters:
    ///   - gradient: Incoming gradient tensor
    ///   - input: Forward pass input tensor
    ///   - center_result: Controls if the result image is centered on the input image
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners
    ///   - layout: Specifies what layout the provided tensor is in
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_bilinear_gradient(&self,
                                   gradient: &MPSGraphTensor,
                                   input: &MPSGraphTensor,
                                   center_result: bool,
                                   align_corners: bool,
                                   layout: TensorNamedDataLayout,
                                   name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let center_result_val = if center_result { YES } else { NO };
            let align_corners_val = if align_corners { YES } else { NO };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeBilinearWithGradientTensor: gradient.0,
                input: input.0,
                centerResult: center_result_val,
                alignCorners: align_corners_val,
                layout: layout as u64,
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize gradient operation using explicit scale and offset tensors.
    /// 
    /// - Parameters:
    ///   - gradient: Incoming gradient tensor
    ///   - input: Forward pass input tensor
    ///   - scale_offset_tensor: 1D float tensor. A 4-element shape as [scaleY, scaleX, offsetY, offsetX]
    ///   - mode: The resampling mode to use
    ///   - layout: Specifies what layout the provided tensor is in
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_gradient_with_scale_offset(&self,
                                           gradient: &MPSGraphTensor,
                                           input: &MPSGraphTensor,
                                           scale_offset_tensor: &MPSGraphTensor,
                                           mode: MPSGraphResizeMode,
                                           layout: TensorNamedDataLayout,
                                           name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeWithGradientTensor: gradient.0,
                input: input.0,
                scaleOffsetTensor: scale_offset_tensor.0,
                mode: mode as u64,
                layout: layout as u64,
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize gradient operation for nearest sampling using explicit scale and offset tensors.
    /// 
    /// - Parameters:
    ///   - gradient: Incoming gradient tensor
    ///   - input: Forward pass input tensor
    ///   - scale_offset_tensor: 1D float tensor. A 4-element shape as [scaleY, scaleX, offsetY, offsetX]
    ///   - nearest_rounding_mode: The rounding mode to use when using nearest resampling
    ///   - layout: Specifies what layout the provided tensor is in
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_nearest_gradient_with_scale_offset(&self,
                                                   gradient: &MPSGraphTensor,
                                                   input: &MPSGraphTensor,
                                                   scale_offset_tensor: &MPSGraphTensor,
                                                   nearest_rounding_mode: MPSGraphResizeNearestRoundingMode,
                                                   layout: TensorNamedDataLayout,
                                                   name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeNearestWithGradientTensor: gradient.0,
                input: input.0,
                scaleOffsetTensor: scale_offset_tensor.0,
                nearestRoundingMode: nearest_rounding_mode as u64,
                layout: layout as u64,
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize gradient operation for bilinear sampling using explicit scale and offset tensors.
    /// 
    /// - Parameters:
    ///   - gradient: Incoming gradient tensor
    ///   - input: Forward pass input tensor
    ///   - scale_offset_tensor: 1D float tensor. A 4-element shape as [scaleY, scaleX, offsetY, offsetX]
    ///   - layout: Specifies what layout the provided tensor is in
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_bilinear_gradient_with_scale_offset(&self,
                                                    gradient: &MPSGraphTensor,
                                                    input: &MPSGraphTensor,
                                                    scale_offset_tensor: &MPSGraphTensor,
                                                    layout: TensorNamedDataLayout,
                                                    name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeBilinearWithGradientTensor: gradient.0,
                input: input.0,
                scaleOffsetTensor: scale_offset_tensor.0,
                layout: layout as u64,
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize gradient operation using separate scale and offset tensors.
    ///
    /// Available on iOS 17+/macOS 14+.
    /// 
    /// - Parameters:
    ///   - gradient: Incoming gradient tensor
    ///   - input: Forward pass input tensor
    ///   - scale_tensor: 1D float tensor of size equal to rank of input
    ///   - offset_tensor: 1D float tensor of size equal to rank of input
    ///   - mode: The resampling mode to use
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_gradient_with_separate_scale_offset(&self,
                                                    gradient: &MPSGraphTensor,
                                                    input: &MPSGraphTensor,
                                                    scale_tensor: &MPSGraphTensor,
                                                    offset_tensor: &MPSGraphTensor,
                                                    mode: MPSGraphResizeMode,
                                                    name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeWithGradientTensor: gradient.0,
                input: input.0,
                scaleTensor: scale_tensor.0,
                offsetTensor: offset_tensor.0,
                mode: mode as u64,
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize gradient operation for nearest sampling using separate scale and offset tensors.
    ///
    /// Available on iOS 17+/macOS 14+.
    /// 
    /// - Parameters:
    ///   - gradient: Incoming gradient tensor
    ///   - input: Forward pass input tensor
    ///   - scale_tensor: 1D float tensor of size equal to rank of input
    ///   - offset_tensor: 1D float tensor of size equal to rank of input
    ///   - nearest_rounding_mode: The rounding mode to use when using nearest resampling
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_nearest_gradient_with_separate_scale_offset(&self,
                                                           gradient: &MPSGraphTensor,
                                                           input: &MPSGraphTensor,
                                                           scale_tensor: &MPSGraphTensor,
                                                           offset_tensor: &MPSGraphTensor,
                                                           nearest_rounding_mode: MPSGraphResizeNearestRoundingMode,
                                                           name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeNearestWithGradientTensor: gradient.0,
                input: input.0,
                scaleTensor: scale_tensor.0,
                offsetTensor: offset_tensor.0,
                nearestRoundingMode: nearest_rounding_mode as u64,
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Resize gradient operation for bilinear sampling using separate scale and offset tensors.
    ///
    /// Available on iOS 17+/macOS 14+.
    /// 
    /// - Parameters:
    ///   - gradient: Incoming gradient tensor
    ///   - input: Forward pass input tensor
    ///   - scale_tensor: 1D float tensor of size equal to rank of input
    ///   - offset_tensor: 1D float tensor of size equal to rank of input
    ///   - name: The name for the operation
    /// - Returns: A valid MPSGraphTensor object
    pub fn resize_bilinear_gradient_with_separate_scale_offset(&self,
                                                            gradient: &MPSGraphTensor,
                                                            input: &MPSGraphTensor,
                                                            scale_tensor: &MPSGraphTensor,
                                                            offset_tensor: &MPSGraphTensor,
                                                            name: Option<&str>) -> MPSGraphTensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };
            
            let tensor: *mut AnyObject = msg_send![self.0, resizeBilinearWithGradientTensor: gradient.0,
                input: input.0,
                scaleTensor: scale_tensor.0,
                offsetTensor: offset_tensor.0,
                name: name_obj
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}