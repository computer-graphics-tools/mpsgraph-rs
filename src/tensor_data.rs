use objc::runtime::Object;
use std::ffi::c_void;
use std::fmt;
use metal::{Buffer, Device};
use metal::foreign_types::ForeignType;
use crate::core::MPSDataType;
use crate::core::MPSShape;

/// A wrapper for MPSGraphTensorData objects
pub struct MPSGraphTensorData(pub(crate) *mut Object);

impl MPSGraphTensorData {
    /// Creates a new MPSGraphTensorData from a slice of data and a shape dimensions
    /// This is a convenience method that converts shape dimensions to an MPSShape
    pub fn new<T: Copy>(data: &[T], shape_dims: &[usize], data_type: MPSDataType) -> Self {
        let shape = MPSShape::from_slice(shape_dims);
        Self::from_bytes(data, &shape, data_type)
    }
    
    /// Creates a new MPSGraphTensorData from a slice of data and a shape
    pub fn from_bytes<T: Copy>(data: &[T], shape: &MPSShape, data_type: MPSDataType) -> Self {
        unsafe {
            let cls = objc::runtime::Class::get("MPSGraphTensorData").unwrap();
            
            let data_ptr = data.as_ptr() as *const c_void;
            let _data_len = (data.len() * std::mem::size_of::<T>()) as usize; // Used for debugging
            
            let obj: *mut Object = msg_send![cls, alloc];
            let obj: *mut Object = msg_send![obj, 
                initWithData:data_ptr
                shape:shape.0
                dataType:data_type as u64
            ];
            
            MPSGraphTensorData(obj)
        }
    }
    
    /// Creates a new MPSGraphTensorData from a Metal buffer
    pub fn from_buffer(buffer: &Buffer, shape: &MPSShape, data_type: MPSDataType) -> Self {
        unsafe {
            let cls = objc::runtime::Class::get("MPSGraphTensorData").unwrap();
            
            let obj: *mut Object = msg_send![cls, alloc];
            let obj: *mut Object = msg_send![obj, 
                initWithBuffer:buffer.as_ptr()
                shape:shape.0
                dataType:data_type as u64
            ];
            
            MPSGraphTensorData(obj)
        }
    }
    
    /// Returns the shape of this tensor data
    pub fn shape(&self) -> MPSShape {
        unsafe {
            let shape: *mut Object = msg_send![self.0, shape];
            let shape: *mut Object = msg_send![shape, retain];
            MPSShape(shape)
        }
    }
    
    /// Returns the data type of this tensor data
    pub fn data_type(&self) -> MPSDataType {
        unsafe {
            let data_type_val: u64 = msg_send![self.0, dataType];
            std::mem::transmute(data_type_val)
        }
    }
    
    /// Returns a pointer to the raw data
    pub fn data(&self) -> *const c_void {
        unsafe {
            msg_send![self.0, data]
        }
    }
    
    /// Copies the data to a new Buffer
    pub fn to_buffer(&self, device: &Device) -> Buffer {
        unsafe {
            let data: *const c_void = msg_send![self.0, data];
            let size: usize = self.shape().element_count() * self.data_type().size_in_bytes();
            
            let buffer = device.new_buffer_with_data(
                data,
                size as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            
            buffer
        }
    }
    
    /// Copies the data to a vector of the specified type
    pub fn to_vec<T: Copy>(&self) -> Vec<T> {
        unsafe {
            let data_ptr: *const c_void = msg_send![self.0, data];
            let data = data_ptr as *const T;
            let count = self.shape().element_count();
            
            let slice = std::slice::from_raw_parts(data, count);
            slice.to_vec()
        }
    }
}

impl Drop for MPSGraphTensorData {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

impl Clone for MPSGraphTensorData {
    fn clone(&self) -> Self {
        unsafe {
            let obj: *mut Object = msg_send![self.0, retain];
            MPSGraphTensorData(obj)
        }
    }
}

impl fmt::Debug for MPSGraphTensorData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MPSGraphTensorData")
            .field("shape", &self.shape())
            .field("data_type", &self.data_type())
            .finish()
    }
}