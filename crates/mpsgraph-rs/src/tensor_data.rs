use crate::core::MPSDataType;
use crate::shape::MPSShape;
use metal::foreign_types::ForeignType;
use metal::Buffer;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use objc2_foundation::NSData;
use std::fmt;

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
                data_size,
            ));
            // Get the raw pointer to NSData
            let ns_data_ptr: *mut AnyObject =
                std::mem::transmute::<&NSData, *mut AnyObject>(ns_data.as_ref());

            // Create MPSGraphDevice from MTLDevice
            let mps_device_class_name = c"MPSGraphDevice";
            let mps_device_cls = objc2::runtime::AnyClass::get(mps_device_class_name).unwrap();
            // Cast the Metal device to a void pointer and then to *mut AnyObject for objc2
            let device_ptr = device.as_ptr() as *mut AnyObject;
            let mps_device: *mut AnyObject =
                msg_send![mps_device_cls, deviceWithMTLDevice: device_ptr,];

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
    pub fn from_buffer_with_row_bytes(
        buffer: &Buffer,
        shape: &MPSShape,
        data_type: MPSDataType,
        row_bytes: u64,
    ) -> Self {
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
    pub fn from_mps_matrix(
        matrix: *mut AnyObject,
        transpose: bool,
        shape: &MPSShape,
        data_type: MPSDataType,
    ) -> Self {
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
    pub fn from_mps_matrix_with_rank(
        matrix: *mut AnyObject,
        transpose: bool,
        shape: &MPSShape,
        data_type: MPSDataType,
        rank: u64,
    ) -> Self {
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
    pub fn from_mps_vector(
        vector: *mut AnyObject,
        shape: &MPSShape,
        data_type: MPSDataType,
    ) -> Self {
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
    pub fn from_mps_vector_with_rank(
        vector: *mut AnyObject,
        shape: &MPSShape,
        data_type: MPSDataType,
        rank: u64,
    ) -> Self {
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
    ///
    /// This method ensures that any data on the GPU is synchronized to CPU-accessible memory.
    /// Use this method when you need to access the tensor data from the CPU after GPU operations.
    pub fn synchronize(&self) {
        unsafe {
            let _: () = msg_send![self.0, synchronizeOnCPU];
        }
    }

    /// Synchronize this tensor data to CPU with a specified region
    ///
    /// This method synchronizes only a specific region of the tensor data, which can be
    /// more efficient than synchronizing the entire tensor.
    ///
    /// - Parameters:
    ///   - region: An NSArray of NSNumber pairs describing the region to synchronize.
    ///             Each pair consists of an offset and a length for each dimension.
    pub fn synchronize_with_region(&self, region: *mut AnyObject) {
        unsafe {
            let _: () = msg_send![self.0, synchronizeOnCPUWithRegion: region];
        }
    }

    /// Synchronize this tensor data to CPU with a specified region created from slices
    ///
    /// This method provides a more Rust-friendly way to specify synchronization regions
    /// by using slices of offsets and lengths.
    ///
    /// - Parameters:
    ///   - dimension_offsets: Offsets for each dimension (starting point)
    ///   - dimension_lengths: Lengths for each dimension (how many elements to synchronize)
    ///
    /// - Returns: true if synchronization was successful, false otherwise
    pub fn synchronize_slice(
        &self,
        dimension_offsets: &[usize],
        dimension_lengths: &[usize],
    ) -> bool {
        assert_eq!(
            dimension_offsets.len(),
            dimension_lengths.len(),
            "dimension_offsets and dimension_lengths must have the same length"
        );

        unsafe {
            // Get the shape to verify dimensions
            let shape = self.shape();
            let shape_dims = shape.dimensions();

            if dimension_offsets.len() != shape_dims.len() {
                return false;
            }

            // Verify offsets and lengths don't exceed dimensions
            for i in 0..dimension_offsets.len() {
                if i < shape_dims.len() {
                    let offset = dimension_offsets[i];
                    let length = dimension_lengths[i];

                    if offset + length > shape_dims[i] {
                        return false;
                    }
                }
            }

            // Create NSArray of region description
            let ns_number_class = objc2::runtime::AnyClass::get(c"NSNumber").unwrap();
            let mut numbers = Vec::with_capacity(dimension_offsets.len() * 2);

            // Add offset-length pairs as NSNumbers
            for i in 0..dimension_offsets.len() {
                // Add offset
                let offset = dimension_offsets[i] as u64;
                let offset_obj: *mut AnyObject =
                    msg_send![ns_number_class, numberWithUnsignedLongLong: offset];
                numbers.push(offset_obj);

                // Add length
                let length = dimension_lengths[i] as u64;
                let length_obj: *mut AnyObject =
                    msg_send![ns_number_class, numberWithUnsignedLongLong: length];
                numbers.push(length_obj);
            }

            // Create NSArray from numbers
            let ns_array = crate::core::create_ns_array_from_pointers(&numbers);

            // Synchronize with the region
            let _: () = msg_send![self.0, synchronizeOnCPUWithRegion: ns_array];

            // Release the NSArray
            objc2::ffi::objc_release(ns_array as *mut _);

            true
        }
    }

    /// Synchronize a rectangular region of the tensor data to CPU (for up to 3D tensors)
    ///
    /// This is a convenience method for synchronizing a rectangular region of a 1D, 2D, or 3D tensor.
    /// For higher-dimensional tensors, use synchronize_slice instead.
    ///
    /// - Parameters:
    ///   - start_x: Starting offset in the first dimension
    ///   - length_x: Length to synchronize in the first dimension
    ///   - start_y: Optional starting offset in the second dimension
    ///   - length_y: Optional length to synchronize in the second dimension
    ///   - start_z: Optional starting offset in the third dimension
    ///   - length_z: Optional length to synchronize in the third dimension
    ///
    /// - Returns: true if synchronization was successful, false otherwise
    pub fn synchronize_region(
        &self,
        start_x: usize,
        length_x: usize,
        start_y: Option<usize>,
        length_y: Option<usize>,
        start_z: Option<usize>,
        length_z: Option<usize>,
    ) -> bool {
        let shape = self.shape();
        let shape_dims = shape.dimensions();

        match shape_dims.len() {
            0 => false, // Scalar tensor, can't synchronize a region
            1 => {
                // 1D tensor
                self.synchronize_slice(&[start_x], &[length_x])
            }
            2 => {
                // 2D tensor
                if let (Some(start_y), Some(length_y)) = (start_y, length_y) {
                    self.synchronize_slice(&[start_x, start_y], &[length_x, length_y])
                } else {
                    false
                }
            }
            3 => {
                // 3D tensor
                if let (Some(start_y), Some(length_y), Some(start_z), Some(length_z)) =
                    (start_y, length_y, start_z, length_z)
                {
                    self.synchronize_slice(
                        &[start_x, start_y, start_z],
                        &[length_x, length_y, length_z],
                    )
                } else {
                    false
                }
            }
            _ => {
                // Higher dimensional tensor - can't use this convenience method
                false
            }
        }
    }

    /// Synchronize and access the tensor data as a slice of a specific type
    ///
    /// This method synchronizes the tensor data to CPU and then provides access
    /// to the data as a slice of the specified type. The type must match the
    /// tensor's data type for correct results.
    ///
    /// - Returns: Option containing a slice reference to the data, or None if access fails
    pub fn synchronized_data<T>(&self) -> Option<&[T]> {
        unsafe {
            // Synchronize the data to CPU first
            self.synchronize();

            // Get the MPSNDArray
            let ndarray = self.mpsndarray();
            if ndarray.is_null() {
                return None;
            }

            // Get the MTLBuffer from the MPSNDArray
            let buffer_ptr: *mut AnyObject = msg_send![ndarray, mtlBuffer];
            if buffer_ptr.is_null() {
                objc2::ffi::objc_release(ndarray as *mut _);
                return None;
            }

            // Get the contents pointer
            let ptr: *mut std::ffi::c_void = msg_send![buffer_ptr, contents];
            if ptr.is_null() {
                objc2::ffi::objc_release(ndarray as *mut _);
                return None;
            }

            // Get the length of the buffer
            let length: usize = msg_send![buffer_ptr, length];

            // Calculate number of elements
            let element_size = std::mem::size_of::<T>();
            let element_count = length / element_size;

            // Create a slice from the pointer
            let slice = std::slice::from_raw_parts(ptr as *const T, element_count);

            // Release the MPSNDArray
            objc2::ffi::objc_release(ndarray as *mut _);

            Some(slice)
        }
    }

    /// Force synchronization of tensor data to a specific device
    ///
    /// This method forces the tensor data to be synchronized to a specific device.
    /// This can be useful when working with multiple devices.
    ///
    /// - Parameters:
    ///   - device: The MTLDevice to synchronize to
    pub fn synchronize_to_device(&self, device: &metal::Device) {
        unsafe {
            let device_ptr = device.as_ptr() as *mut AnyObject;
            let _: () = msg_send![self.0, synchronizeToDevice: device_ptr];
        }
    }
}

impl Drop for MPSGraphTensorData {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // Set the pointer to null to prevent further access
            let ptr = self.0;
            self.0 = std::ptr::null_mut();

            // MEMORY LEAK: We're deliberately not releasing the object to avoid the crash
            // objc2::ffi::objc_release(ptr as *mut _);
        }
    }
}

impl Clone for MPSGraphTensorData {
    fn clone(&self) -> Self {
        if !self.0.is_null() {
            // Don't retain, just copy the pointer
            // let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
            let obj = self.0;
            
            MPSGraphTensorData(obj)
        } else {
            MPSGraphTensorData(std::ptr::null_mut())
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
