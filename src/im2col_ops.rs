use objc::runtime::Object;
use objc::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, MPSShape};
use crate::convolution_transpose_ops::TensorNamedDataLayout;

/// Descriptor for Image to Column operations
pub struct MPSGraphImToColOpDescriptor(pub(crate) *mut Object);

impl MPSGraphImToColOpDescriptor {
    /// Creates a new descriptor with full parameters for im2col operations
    ///
    /// # Arguments
    ///
    /// * `kernel_width` - The kernel size in width dimension
    /// * `kernel_height` - The kernel size in height dimension
    /// * `stride_in_x` - The stride in width dimension
    /// * `stride_in_y` - The stride in height dimension
    /// * `dilation_rate_in_x` - The dilation in width dimension
    /// * `dilation_rate_in_y` - The dilation in height dimension
    /// * `padding_left` - The padding in width dimension on the left side
    /// * `padding_right` - The padding in width dimension on the right side
    /// * `padding_top` - The padding in height dimension at the top
    /// * `padding_bottom` - The padding in height dimension at the bottom
    /// * `data_layout` - The layout of source or output tensor
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
        data_layout: TensorNamedDataLayout,
    ) -> Self {
        unsafe {
            let cls = objc::runtime::Class::get("MPSGraphImToColOpDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![
                cls,
                descriptorWithKernelWidth:kernel_width
                kernelHeight:kernel_height
                strideInX:stride_in_x
                strideInY:stride_in_y
                dilationRateInX:dilation_rate_in_x
                dilationRateInY:dilation_rate_in_y
                paddingLeft:padding_left
                paddingRight:padding_right
                paddingTop:padding_top
                paddingBottom:padding_bottom
                dataLayout:data_layout as u64
            ];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphImToColOpDescriptor(descriptor)
        }
    }
    
    /// Creates a new descriptor for col2im operations (without explicit padding)
    ///
    /// # Arguments
    ///
    /// * `kernel_width` - The kernel size in width dimension
    /// * `kernel_height` - The kernel size in height dimension
    /// * `stride_in_x` - The stride in width dimension
    /// * `stride_in_y` - The stride in height dimension
    /// * `dilation_rate_in_x` - The dilation in width dimension
    /// * `dilation_rate_in_y` - The dilation in height dimension
    /// * `data_layout` - The layout of source or output tensor
    pub fn for_col2im(
        kernel_width: usize,
        kernel_height: usize,
        stride_in_x: usize,
        stride_in_y: usize,
        dilation_rate_in_x: usize,
        dilation_rate_in_y: usize,
        data_layout: TensorNamedDataLayout,
    ) -> Self {
        unsafe {
            let cls = objc::runtime::Class::get("MPSGraphImToColOpDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![
                cls,
                descriptorWithKernelWidth:kernel_width
                kernelHeight:kernel_height
                strideInX:stride_in_x
                strideInY:stride_in_y
                dilationRateInX:dilation_rate_in_x
                dilationRateInY:dilation_rate_in_y
                dataLayout:data_layout as u64
            ];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphImToColOpDescriptor(descriptor)
        }
    }
    
    /// Sets the explicit padding values for the descriptor
    ///
    /// # Arguments
    ///
    /// * `padding_left` - The padding in width dimension on the left side
    /// * `padding_right` - The padding in width dimension on the right side
    /// * `padding_top` - The padding in height dimension at the top
    /// * `padding_bottom` - The padding in height dimension at the bottom
    pub fn set_explicit_padding(
        &self,
        padding_left: usize,
        padding_right: usize,
        padding_top: usize,
        padding_bottom: usize,
    ) {
        unsafe {
            let _: () = msg_send![
                self.0,
                setExplicitPaddingWithPaddingLeft:padding_left
                paddingRight:padding_right
                paddingTop:padding_top
                paddingBottom:padding_bottom
            ];
        }
    }
    
    /// Sets the kernel width
    pub fn set_kernel_width(&self, width: usize) {
        unsafe {
            let _: () = msg_send![self.0, setKernelWidth:width];
        }
    }
    
    /// Sets the kernel height
    pub fn set_kernel_height(&self, height: usize) {
        unsafe {
            let _: () = msg_send![self.0, setKernelHeight:height];
        }
    }
    
    /// Sets the stride in X dimension
    pub fn set_stride_in_x(&self, stride: usize) {
        unsafe {
            let _: () = msg_send![self.0, setStrideInX:stride];
        }
    }
    
    /// Sets the stride in Y dimension
    pub fn set_stride_in_y(&self, stride: usize) {
        unsafe {
            let _: () = msg_send![self.0, setStrideInY:stride];
        }
    }
    
    /// Sets the dilation rate in X dimension
    pub fn set_dilation_rate_in_x(&self, rate: usize) {
        unsafe {
            let _: () = msg_send![self.0, setDilationRateInX:rate];
        }
    }
    
    /// Sets the dilation rate in Y dimension
    pub fn set_dilation_rate_in_y(&self, rate: usize) {
        unsafe {
            let _: () = msg_send![self.0, setDilationRateInY:rate];
        }
    }
    
    /// Sets the padding on the left
    pub fn set_padding_left(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingLeft:padding];
        }
    }
    
    /// Sets the padding on the right
    pub fn set_padding_right(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingRight:padding];
        }
    }
    
    /// Sets the padding on the top
    pub fn set_padding_top(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingTop:padding];
        }
    }
    
    /// Sets the padding on the bottom
    pub fn set_padding_bottom(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingBottom:padding];
        }
    }
    
    /// Sets the data layout
    pub fn set_data_layout(&self, layout: TensorNamedDataLayout) {
        unsafe {
            let _: () = msg_send![self.0, setDataLayout:layout as u64];
        }
    }
}

impl Drop for MPSGraphImToColOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

impl Clone for MPSGraphImToColOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let desc: *mut Object = msg_send![self.0, copy];
            MPSGraphImToColOpDescriptor(desc)
        }
    }
}

/// Image to Column operations for MPSGraph
impl MPSGraph {
    /// Creates an image to column operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `source` - The tensor containing the source data. Must be of rank 4. The layout is defined by descriptor.dataLayout.
    /// * `descriptor` - The descriptor object that specifies the parameters of the operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object
    pub fn im2col(
        &self,
        source: &MPSGraphTensor,
        descriptor: &MPSGraphImToColOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                imToColWithSourceTensor:source.0
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a column to image operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `source` - The tensor containing the source data. Must be of rank 4. The layout is defined by descriptor.dataLayout.
    /// * `output_shape` - The result tensor shape.
    /// * `descriptor` - The descriptor object that specifies the parameters of the operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object
    pub fn col2im(
        &self,
        source: &MPSGraphTensor,
        output_shape: &MPSShape,
        descriptor: &MPSGraphImToColOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                colToImWithSourceTensor:source.0
                outputShape:output_shape.0
                descriptor:descriptor.0
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
    use std::collections::HashMap;
    
    #[test]
    fn test_im2col() {
        if should_skip_test("test_im2col") {
            return;
        }
        
        let graph = MPSGraph::new();
        
        // Create a 1x1x4x4 input tensor (NCHW)
        let input_shape = MPSShape::from_slice(&[1, 1, 4, 4]);
        let input_data = vec![
            1.0f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0
        ];
        
        let tensor = graph.placeholder(&input_shape, MPSDataType::Float32, None);
        
        // Create descriptor with 2x2 kernel, stride 1, no dilation, no padding
        let descriptor = MPSGraphImToColOpDescriptor::new(
            2, 2, // kernel size
            1, 1, // stride
            1, 1, // dilation
            0, 0, 0, 0, // padding
            TensorNamedDataLayout::NCHW // data layout
        );
        
        // Apply im2col operation
        let im2col_result = graph.im2col(&tensor, &descriptor, Some("im2col"));
        
        // Run the graph
        let mut feeds = HashMap::new();
        feeds.insert(&tensor, crate::MPSGraphTensorData::new(&input_data, &[1, 1, 4, 4], MPSDataType::Float32));
        
        let results = graph.run(feeds, &[&im2col_result]);
        
        // Get the result data
        let result_data = results[&im2col_result].to_vec::<f32>();
        
        // For a 4x4 input with 2x2 kernel and stride 1, we expect 3x3 output locations 
        // with 4 values (2x2) at each location, so 3x3x4 = 36 values
        assert_eq!(result_data.len(), 36);
    }
}