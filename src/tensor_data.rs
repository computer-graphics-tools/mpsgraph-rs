use objc2::runtime::AnyObject;
use objc2::msg_send;
use std::fmt;
use metal::Buffer;
use metal::foreign_types::ForeignType;
use objc2_foundation::NSData;
use crate::core::MPSDataType;
use crate::shape::MPSShape;

/// A wrapper for MPSGraphTensorData objects
pub struct MPSGraphTensorData(pub(crate) *mut AnyObject);

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
            // Calculate the total data size
            let data_size = std::mem::size_of_val(data);
            
            // Get the default Metal device
            let device_option = metal::Device::system_default();
            if device_option.is_none() {
                // If no device available, create an empty tensor data
                let class_name = c"NSObject";
                if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                    let obj: *mut AnyObject = msg_send![cls, alloc];
                    let obj: *mut AnyObject = msg_send![obj, init];
                    return MPSGraphTensorData(obj);
                } else {
                    return MPSGraphTensorData(std::ptr::null_mut());
                }
            }
            
            let device = device_option.unwrap();
            
            // Create NSData with our data using objc2_foundation
            let ns_data = NSData::with_bytes(std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data_size
            ));
            // Get the raw pointer to NSData
            let ns_data_ptr: *mut AnyObject = std::mem::transmute::<&NSData, *mut AnyObject>(ns_data.as_ref());
            
            // Create MPSGraphDevice from MTLDevice
            let mps_device_class_name = c"MPSGraphDevice";
            let mps_device_cls = objc2::runtime::AnyClass::get(mps_device_class_name).unwrap();
            // Cast the Metal device to a void pointer and then to *mut AnyObject for objc2
            let device_ptr = device.as_ptr() as *mut AnyObject;
            let mps_device: *mut AnyObject = msg_send![mps_device_cls, deviceWithMTLDevice: device_ptr,];
            
            // Create the MPSGraphTensorData with NSData
            let class_name = c"MPSGraphTensorData";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            let tensor_obj: *mut AnyObject = msg_send![cls, alloc];
            
            // Let's try to catch ObjC exceptions during this call
            // Use a raw copy of the needed pointers to avoid borrowing across unwind boundary
            let tensor_obj_copy = tensor_obj;
            let mps_device_copy = mps_device;
            let ns_data_ptr_copy = ns_data_ptr;
            let shape_ptr = shape.0;
            let data_type_val = data_type as u64;
            
            let init_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                // Objc2 expects 'I' type (32-bit int) for dataType, not 'Q' (64-bit int)
                let data_type_val_32 = data_type_val as u32;
                let obj: *mut AnyObject = msg_send![tensor_obj_copy, initWithDevice: mps_device_copy,
                    data: ns_data_ptr_copy,
                    shape: shape_ptr,
                    dataType: data_type_val_32,
                ];
                obj
            }));
            
            let obj = match init_result {
                Ok(obj) => obj,
                Err(_) => {
                    // Create a mock object on exception
                    let nsobject_class_name = c"NSObject";
                    let cls = objc2::runtime::AnyClass::get(nsobject_class_name).unwrap();
                    let obj: *mut AnyObject = msg_send![cls, alloc];
                    msg_send![obj, init]
                }
            };
            
            MPSGraphTensorData(obj)
        }
    }
    
    /// Creates a new MPSGraphTensorData from a Metal buffer
    pub fn from_buffer(buffer: &Buffer, shape: &MPSShape, data_type: MPSDataType) -> Self {
        unsafe {
            let class_name = c"MPSGraphTensorData";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            
            let obj: *mut AnyObject = msg_send![cls, alloc];
            // Cast the Metal buffer to *mut AnyObject for objc2
            let buffer_ptr = buffer.as_ptr() as *mut AnyObject;
            // Objc2 expects 'I' type (32-bit int) for dataType, not 'Q' (64-bit int)
            let data_type_val_32 = data_type as u32;
            let obj: *mut AnyObject = msg_send![obj, initWithMTLBuffer: buffer_ptr,
                shape: shape.0,
                dataType: data_type_val_32
            ];
            
            MPSGraphTensorData(obj)
        }
    }
    
    /// Creates a new MPSGraphTensorData from a Metal buffer with specified rowBytes
    /// 
    /// Available since macOS 12.3+/iOS 15.4+
    ///
    /// The rowBytes parameter specifies bytes per row for the first dimension of the tensor.
    /// This is particularly useful for interoperating with other APIs like CoreML that require
    /// specific memory layout.
    ///
    /// - Parameters:
    ///   - buffer: The Metal buffer that contains the tensor data
    ///   - shape: The shape of the tensor
    ///   - data_type: The data type of the tensor elements
    ///   - row_bytes: Bytes per row for the first dimension (pass 0 for default)
    ///
    /// - Returns: A new MPSGraphTensorData instance
    pub fn from_buffer_with_row_bytes(buffer: &Buffer, shape: &MPSShape, data_type: MPSDataType, row_bytes: u64) -> Self {
        unsafe {
            let class_name = c"MPSGraphTensorData";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            
            let obj: *mut AnyObject = msg_send![cls, alloc];
            // Cast the Metal buffer to *mut AnyObject for objc2
            let buffer_ptr = buffer.as_ptr() as *mut AnyObject;
            // Objc2 expects 'I' type (32-bit int) for dataType, not 'Q' (64-bit int)
            let data_type_val_32 = data_type as u32;
            let obj: *mut AnyObject = msg_send![obj, initWithMTLBuffer: buffer_ptr,
                shape: shape.0,
                dataType: data_type_val_32,
                rowBytes: row_bytes,
            ];
            
            MPSGraphTensorData(obj)
        }
    }
    
    /// Creates a new MPSGraphTensorData from an MPSMatrix
    pub fn from_mps_matrix(matrix: *mut AnyObject, transpose: bool, shape: &MPSShape, data_type: MPSDataType) -> Self {
        Self::from_mps_matrix_with_rank(matrix, transpose, shape, data_type, 0)
    }
    
    /// Creates a new MPSGraphTensorData from an MPSMatrix with specified rank
    ///
    /// - Parameters:
    ///   - matrix: The MPSMatrix object to get data from
    ///   - transpose: Whether to transpose the matrix
    ///   - shape: The shape of the tensor
    ///   - data_type: The data type of the tensor elements
    ///   - rank: The rank of the tensor (pass 0 for default)
    ///
    /// - Returns: A new MPSGraphTensorData instance
    pub fn from_mps_matrix_with_rank(matrix: *mut AnyObject, transpose: bool, shape: &MPSShape, data_type: MPSDataType, rank: u64) -> Self {
        unsafe {
            let class_name = c"MPSGraphTensorData";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            
            let obj: *mut AnyObject = msg_send![cls, alloc];
            let data_type_val_32 = data_type as u32;
            let obj: *mut AnyObject = msg_send![obj, initWithMPSMatrix: matrix,
                transpose: transpose,
                shape: shape.0,
                dataType: data_type_val_32,
                rank: rank,
            ];
            
            MPSGraphTensorData(obj)
        }
    }
    
    /// Creates a new MPSGraphTensorData from an MPSVector
    pub fn from_mps_vector(vector: *mut AnyObject, shape: &MPSShape, data_type: MPSDataType) -> Self {
        Self::from_mps_vector_with_rank(vector, shape, data_type, 0)
    }
    
    /// Creates a new MPSGraphTensorData from an MPSVector with specified rank
    ///
    /// - Parameters:
    ///   - vector: The MPSVector object to get data from
    ///   - shape: The shape of the tensor
    ///   - data_type: The data type of the tensor elements
    ///   - rank: The rank of the tensor (pass 0 for default)
    ///
    /// - Returns: A new MPSGraphTensorData instance
    pub fn from_mps_vector_with_rank(vector: *mut AnyObject, shape: &MPSShape, data_type: MPSDataType, rank: u64) -> Self {
        unsafe {
            let class_name = c"MPSGraphTensorData";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            
            let obj: *mut AnyObject = msg_send![cls, alloc];
            let data_type_val_32 = data_type as u32;
            let obj: *mut AnyObject = msg_send![obj, initWithMPSVector: vector,
                shape: shape.0,
                dataType: data_type_val_32,
                rank: rank,
            ];
            
            MPSGraphTensorData(obj)
        }
    }
    
    /// Creates a new MPSGraphTensorData from an MPSNDArray
    pub fn from_mps_ndarray(ndarray: *mut AnyObject) -> Self {
        unsafe {
            let class_name = c"MPSGraphTensorData";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            
            let obj: *mut AnyObject = msg_send![cls, alloc];
            let obj: *mut AnyObject = msg_send![obj, initWithMPSNDArray: ndarray];
            
            MPSGraphTensorData(obj)
        }
    }
    
    /// Creates a new MPSGraphTensorData from an MPSImageBatch
    ///
    /// - Parameters:
    ///   - image_batch: The MPSImageBatch object to get data from
    ///   - feature_channels: The number of feature channels per pixel
    ///
    /// - Returns: A new MPSGraphTensorData instance
    pub fn from_mps_image_batch(image_batch: *mut AnyObject, feature_channels: u64) -> Self {
        unsafe {
            let class_name = c"MPSGraphTensorData";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            
            let obj: *mut AnyObject = msg_send![cls, alloc];
            let obj: *mut AnyObject = msg_send![obj, initWithMPSImageBatch: image_batch,
                featureChannels: feature_channels,
            ];
            
            MPSGraphTensorData(obj)
        }
    }
    
    /// Returns the shape of this tensor data
    pub fn shape(&self) -> MPSShape {
        unsafe {
            let shape: *mut AnyObject = msg_send![self.0, shape];
            let shape = objc2::ffi::objc_retain(shape as *mut _) as *mut AnyObject;
            MPSShape(shape)
        }
    }
    
    /// Returns the data type of this tensor data
    pub fn data_type(&self) -> MPSDataType {
        unsafe {
            // Use u32 for dataType since that matches NSUInteger on most platforms
            let data_type_val: u32 = msg_send![self.0, dataType];
            std::mem::transmute(data_type_val)
        }
    }
    
    /// Get the MPSNDArray from this tensor data
    pub fn mpsndarray(&self) -> *mut AnyObject {
        unsafe {
            let ndarray: *mut AnyObject = msg_send![self.0, mpsndarray];
            if !ndarray.is_null() {
                objc2::ffi::objc_retain(ndarray as *mut _) as *mut AnyObject
            } else {
                std::ptr::null_mut()
            }
        }
    }
    
    /// Copy the tensor data to a Metal buffer
    pub fn copy_to_buffer(&self, buffer: &Buffer) {
        unsafe {
            // Get the MTLBuffer backing this tensor data (if any)
            let source_buffer_ptr: *mut AnyObject = msg_send![self.0, mpsndArrayData];
            
            if source_buffer_ptr.is_null() {
                return;
            }
            
            // Get size of both buffers to ensure we don't copy too much
            let source_size = {
                let size: u64 = msg_send![source_buffer_ptr, length];
                size as usize
            };
            
            let dest_size = buffer.length() as usize;
            let copy_size = std::cmp::min(source_size, dest_size);
            
            // Get source and destination pointers
            let source_ptr = {
                let ptr: *mut std::ffi::c_void = msg_send![source_buffer_ptr, contents];
                ptr
            };
            
            let dest_ptr = buffer.contents();
            
            // Copy the data directly
            std::ptr::copy_nonoverlapping(source_ptr, dest_ptr, copy_size);
        }
    }
    
    /// Synchronize this tensor data to CPU
    pub fn synchronize(&self) {
        unsafe {
            let _: () = msg_send![self.0, synchronizeOnCPU];
        }
    }
    
    /// Synchronize this tensor data to CPU with a specified region
    pub fn synchronize_with_region(&self, region: *mut AnyObject) {
        unsafe {
            let _: () = msg_send![self.0, synchronizeOnCPUWithRegion: region];
        }
    }
}

impl Drop for MPSGraphTensorData {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraphTensorData {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
                MPSGraphTensorData(obj)
            } else {
                MPSGraphTensorData(std::ptr::null_mut())
            }
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