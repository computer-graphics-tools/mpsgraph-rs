use metal::{Device, DeviceRef, Buffer, MTLResourceOptions};

/// Helper functions for working with Metal buffers and data transfer
pub mod buffer {
    use super::*;

    /// Create a Metal buffer with data
    pub fn create_buffer_with_data<T: Copy>(device: &DeviceRef, data: &[T]) -> Buffer {
        let buffer_size = (data.len() * std::mem::size_of::<T>()) as u64;
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            buffer_size,
            MTLResourceOptions::StorageModeShared
        );
        buffer
    }

    /// Read data from a Metal buffer into a Vec
    pub fn read_buffer_data<T: Copy>(buffer: &Buffer, len: usize) -> Vec<T> {
        let mut result = Vec::with_capacity(len);
        
        unsafe {
            let ptr = buffer.contents() as *const T;
            for i in 0..len {
                result.push(*ptr.add(i));
            }
        }
        
        result
    }

    /// Copy data from one buffer to another
    pub fn copy_buffer_data(src: &Buffer, dst: &Buffer, size: u64) {
        unsafe {
            let src_ptr = src.contents() as *const u8;
            let dst_ptr = dst.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size as usize);
        }
    }
}

/// Helper functions for tensor operations
pub mod tensor {
    use crate::data_types::{MPSDataType, MPSShapeDescriptor};
    
    /// Create a shape descriptor for a scalar value
    pub fn scalar_shape(data_type: MPSDataType) -> MPSShapeDescriptor {
        MPSShapeDescriptor::new(vec![1], data_type)
    }
    
    /// Create a shape descriptor for a vector
    pub fn vector_shape(length: u64, data_type: MPSDataType) -> MPSShapeDescriptor {
        MPSShapeDescriptor::new(vec![length], data_type)
    }
    
    /// Create a shape descriptor for a matrix
    pub fn matrix_shape(rows: u64, cols: u64, data_type: MPSDataType) -> MPSShapeDescriptor {
        MPSShapeDescriptor::new(vec![rows, cols], data_type)
    }
    
    /// Create a shape descriptor for a 3D tensor
    pub fn tensor3d_shape(dim1: u64, dim2: u64, dim3: u64, data_type: MPSDataType) -> MPSShapeDescriptor {
        MPSShapeDescriptor::new(vec![dim1, dim2, dim3], data_type)
    }
    
    /// Create a shape descriptor for a 4D tensor (typically used for images: [batch, height, width, channels])
    pub fn tensor4d_shape(batch: u64, height: u64, width: u64, channels: u64, data_type: MPSDataType) -> MPSShapeDescriptor {
        MPSShapeDescriptor::new(vec![batch, height, width, channels], data_type)
    }
}