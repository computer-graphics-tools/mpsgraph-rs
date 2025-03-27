use objc2::runtime::AnyObject;
// In objc2, use false as NO and true as YES
const NO: bool = false;
const YES: bool = true;
use objc2::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, AsRawObject};

/// The non-maximum suppression coordinate mode.
///
/// This mode specifies the representation used for the 4 box coordinate values.
/// Center coordinate modes define a centered box and the box dimensions.
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum MPSGraphNonMaximumSuppressionCoordinateMode {
    /// [h_start, w_start, h_end, w_end]
    CornersHeightFirst = 0,
    /// [w_start, h_start, w_end, h_end]
    CornersWidthFirst = 1,
    /// [h_center, w_center, box_height, box_width]
    CentersHeightFirst = 2,
    /// [w_center, h_center, box_width, box_height]
    CentersWidthFirst = 3,
}

/// Non-maximum suppression operations for MPSGraph
impl MPSGraph {
    /// Creates a nonMaximumumSuppression operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `boxes_tensor` - A tensor containing the coordinates of the input boxes. 
    ///                    Must be a rank 3 tensor of shape [N,B,4] of type `Float32`
    /// * `scores_tensor` - A tensor containing the scores of the input boxes. 
    ///                    Must be a rank 3 tensor of shape [N,B,K] of type `Float32`
    /// * `iou_threshold` - The threshold for when to reject boxes based on their Intersection Over Union. 
    ///                    Valid range is [0,1].
    /// * `score_threshold` - The threshold for when to reject boxes based on their score, before IOU suppression.
    /// * `per_class_suppression` - When this is specified a box will only suppress another box if they have the same class.
    /// * `coordinate_mode` - The coordinate mode the box coordinates are provided in.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object containing the non-maximum suppression results.
    pub fn non_maximum_suppression(
        &self,
        boxes_tensor:  &MPSGraphTensor,
        scores_tensor:  &MPSGraphTensor,
        iou_threshold:  f32,
        score_threshold:  f32,
        per_class_suppression:  bool,
        coordinate_mode:  MPSGraphNonMaximumSuppressionCoordinateMode,
        name:  Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        let per_class_suppression_obj = if per_class_suppression { YES } else { NO };
        
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, nonMaximumSuppressionWithBoxesTensor: boxes_tensor.0,
                scoresTensor: scores_tensor.0,
                IOUThreshold: iou_threshold,
                scoreThreshold: score_threshold,
                perClassSuppression: per_class_suppression_obj,
                coordinateMode: coordinate_mode as u64,
                name: name_obj
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a nonMaximumumSuppression operation with class indices and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `boxes_tensor` - A tensor containing the coordinates of the input boxes. 
    ///                    Must be a rank 3 tensor of shape [N,B,4] of type `Float32`
    /// * `scores_tensor` - A tensor containing the scores of the input boxes. 
    ///                    Must be a rank 3 tensor of shape [N,B,1] of type `Float32`
    /// * `class_indices_tensor` - A tensor containing the class indices of the input boxes.
    ///                    Must be a rank 2 tensor of shape [N,B] of type `Int32`
    /// * `iou_threshold` - The threshold for when to reject boxes based on their Intersection Over Union. 
    ///                    Valid range is [0,1].
    /// * `score_threshold` - The threshold for when to reject boxes based on their score, before IOU suppression.
    /// * `per_class_suppression` - When this is specified a box will only suppress another box if they have the same class.
    /// * `coordinate_mode` - The coordinate mode the box coordinates are provided in.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object containing the non-maximum suppression results.
    pub fn non_maximum_suppression_with_class_indices(
        &self,
        boxes_tensor:  &MPSGraphTensor,
        scores_tensor:  &MPSGraphTensor,
        class_indices_tensor:  &MPSGraphTensor,
        iou_threshold:  f32,
        score_threshold:  f32,
        per_class_suppression:  bool,
        coordinate_mode:  MPSGraphNonMaximumSuppressionCoordinateMode,
        name:  Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        let per_class_suppression_obj = if per_class_suppression { YES } else { NO };
        
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, nonMaximumSuppressionWithBoxesTensor: boxes_tensor.0,
                scoresTensor: scores_tensor.0,
                classIndicesTensor: class_indices_tensor.0,
                IOUThreshold: iou_threshold,
                scoreThreshold: score_threshold,
                perClassSuppression: per_class_suppression_obj,
                coordinateMode: coordinate_mode as u64,
                name: name_obj
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _) as *mut AnyObject;
            MPSGraphTensor(result)
        }
    }
}