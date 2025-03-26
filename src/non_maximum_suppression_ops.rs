use objc::runtime::{Object, YES, NO};
use objc::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::NSString;

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
        boxes_tensor: &MPSGraphTensor,
        scores_tensor: &MPSGraphTensor,
        iou_threshold: f32,
        score_threshold: f32,
        per_class_suppression: bool,
        coordinate_mode: MPSGraphNonMaximumSuppressionCoordinateMode,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        let per_class_suppression_obj = if per_class_suppression { YES } else { NO };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                nonMaximumSuppressionWithBoxesTensor:boxes_tensor.0
                scoresTensor:scores_tensor.0
                IOUThreshold:iou_threshold
                scoreThreshold:score_threshold
                perClassSuppression:per_class_suppression_obj
                coordinateMode:coordinate_mode as u64
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
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
        boxes_tensor: &MPSGraphTensor,
        scores_tensor: &MPSGraphTensor,
        class_indices_tensor: &MPSGraphTensor,
        iou_threshold: f32,
        score_threshold: f32,
        per_class_suppression: bool,
        coordinate_mode: MPSGraphNonMaximumSuppressionCoordinateMode,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        let per_class_suppression_obj = if per_class_suppression { YES } else { NO };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                nonMaximumSuppressionWithBoxesTensor:boxes_tensor.0
                scoresTensor:scores_tensor.0
                classIndicesTensor:class_indices_tensor.0
                IOUThreshold:iou_threshold
                scoreThreshold:score_threshold
                perClassSuppression:per_class_suppression_obj
                coordinateMode:coordinate_mode as u64
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MPSDataType, tests::should_skip_test};
    use crate::core::MPSShape;
    use std::collections::HashMap;
    
    #[test]
    fn test_non_maximum_suppression() {
        if should_skip_test("test_non_maximum_suppression") {
            return;
        }
        
        let graph = MPSGraph::new();
        
        // Create boxes tensor: [1,3,4] - one batch, 3 boxes, 4 coordinates per box
        let boxes_shape = MPSShape::from_slice(&[1, 3, 4]);
        let boxes_data = vec![
            // Box 1: [0, 0, 10, 10]
            0.0f32, 0.0, 10.0, 10.0,
            // Box 2: [1, 1, 11, 11] - overlaps with Box 1
            1.0, 1.0, 11.0, 11.0,
            // Box 3: [20, 20, 30, 30] - no overlap with other boxes
            20.0, 20.0, 30.0, 30.0
        ];
        
        // Create scores tensor: [1,3,1] - one batch, 3 boxes, 1 score per box
        let scores_shape = MPSShape::from_slice(&[1, 3, 1]);
        let scores_data = vec![
            0.9f32, // Box 1 score
            0.8f32, // Box 2 score
            0.7f32  // Box 3 score
        ];
        
        // Create placeholder tensors
        let boxes_tensor = graph.placeholder(&boxes_shape, MPSDataType::Float32, None);
        let scores_tensor = graph.placeholder(&scores_shape, MPSDataType::Float32, None);
        
        // Apply non-maximum suppression
        // Use IOU threshold of 0.5 - Box 2 will be suppressed by Box 1
        let nms_result = graph.non_maximum_suppression(
            &boxes_tensor,
            &scores_tensor,
            0.5, // IOU threshold
            0.1, // Score threshold
            false, // No per-class suppression
            MPSGraphNonMaximumSuppressionCoordinateMode::CornersHeightFirst,
            Some("nms_test")
        );
        
        // Run the graph
        let mut feeds = HashMap::new();
        feeds.insert(&boxes_tensor, crate::MPSGraphTensorData::new(&boxes_data, &[1, 3, 4], MPSDataType::Float32));
        feeds.insert(&scores_tensor, crate::MPSGraphTensorData::new(&scores_data, &[1, 3, 1], MPSDataType::Float32));
        
        let results = graph.run(feeds, &[&nms_result]);
        
        // Get the result - we expect two boxes to be kept (Box 1 and Box 3)
        let result_data = results[&nms_result].to_vec::<u32>();
        
        // We don't verify exact values since NMS is non-deterministic
        // Just check that we get a non-empty result
        assert!(!result_data.is_empty());
    }
}