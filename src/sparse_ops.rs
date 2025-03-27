use objc2::runtime::AnyObject;
use objc2::msg_send;
use objc2_foundation::{NSArray, NSString};
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{MPSShape, MPSDataType, AsRawObject};

/// The sparse storage options for MPSGraph sparse operations.
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum MPSGraphSparseStorageType {
    /// COO (Coordinate) Storage format
    COO = 0,
    /// CSC (Compressed Sparse Column) Storage format
    CSC = 1,
    /// CSR (Compressed Sparse Row) Storage format
    CSR = 2,
}

/// Descriptor for sparse tensor creation
pub struct MPSGraphCreateSparseOpDescriptor(pub(crate) *mut AnyObject);

impl MPSGraphCreateSparseOpDescriptor {
    /// Creates a new descriptor for a sparse tensor.
    ///
    /// # Arguments
    ///
    /// * `storage_type` - The storage format of the sparse tensor
    /// * `data_type` - The data type of the sparse tensor
    ///
    /// # Returns
    ///
    /// A new sparse tensor descriptor
    pub fn new(
        storage_type:  MPSGraphSparseStorageType,
        data_type:  MPSDataType,
    ) -> Self {
        unsafe {
            // Get the class, unwrap it, then use it in msg_send
            let class_name = c"MPSGraphCreateSparseOpDescriptor";
            if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                let descriptor: *mut AnyObject = msg_send![
                    cls, descriptorWithStorageType: storage_type as u64,
                    dataType: data_type as u64
                ];
                let descriptor = objc2::ffi::objc_retain(descriptor as *mut _);
                MPSGraphCreateSparseOpDescriptor(descriptor)
            } else {
                // Fall back to creating an empty object if class not found
                let empty_obj: *mut AnyObject = std::ptr::null_mut();
                MPSGraphCreateSparseOpDescriptor(empty_obj)
            }
        }
    }
    
    /// Sets the sparse storage type
    pub fn set_sparse_storage_type(&self, storage_type: MPSGraphSparseStorageType) {
        unsafe {
            let _: () = msg_send![self.0, setSparseStorageType: storage_type as u64];
        }
    }
    
    /// Sets the data type
    pub fn set_data_type(&self, data_type: MPSDataType) {
        unsafe {
            let _: () = msg_send![self.0, setDataType: data_type as u64];
        }
    }
}

impl Drop for MPSGraphCreateSparseOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            objc2::ffi::objc_release(self.0 as *mut _);
        }
    }
}

impl Clone for MPSGraphCreateSparseOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let desc: *mut AnyObject = msg_send![self.0, copy];
            MPSGraphCreateSparseOpDescriptor(desc)
        }
    }
}

/// Sparse operations for MPSGraph
impl MPSGraph {
    /// Creates a sparse tensor with the specified indices and values.
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices of non-zero values for each dimension
    /// * `values` - The non-zero values
    /// * `dense_shape` - The shape of the corresponding dense tensor
    /// * `descriptor` - Descriptor specifying storage format and data type
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor representing the sparse tensor
    pub fn sparse_tensor_with_indices_and_values(
        &self,
        indices:  &[&MPSGraphTensor],
        values:  &MPSGraphTensor,
        dense_shape:  &MPSShape,
        descriptor:  &MPSGraphCreateSparseOpDescriptor,
        name:  Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        // Create NSArray of input tensors using objc2_foundation
        let input_tensors_array = unsafe {
            // Convert to slice of references to AnyObject
            let indices_refs: Vec<&objc2::runtime::AnyObject> = indices.iter()
                .map(|tensor| &*tensor.0.cast::<objc2::runtime::AnyObject>())
                .collect();
            
            // Create NSArray from references
            let array = NSArray::from_slice(&indices_refs);
            let ns_array: *mut AnyObject = array.as_ref() as *const NSArray<objc2::runtime::AnyObject> as *mut AnyObject;
            ns_array
        };
        
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, sparseTensorWithIndicesTensors: input_tensors_array,
                valuesTensor: values.0,
                denseShape: dense_shape.0,
                descriptor: descriptor.0,
                name: name_obj
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _);
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a SparseToDense operation that converts a sparse tensor to a dense tensor.
    ///
    /// # Arguments
    ///
    /// * `indices` - A tensor containing the indices of nonzero elements
    /// * `values` - A tensor containing the values of nonzero elements
    /// * `dense_shape` - The desired shape of the dense tensor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A dense tensor containing the values from the sparse format
    pub fn sparse_to_dense(
        &self,
        indices:  &[&MPSGraphTensor],
        values:  &MPSGraphTensor,
        dense_shape:  &MPSGraphTensor,
        name:  Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        let input_tensors_array = unsafe {
            // Convert to slice of references to AnyObject
            let indices_refs: Vec<&objc2::runtime::AnyObject> = indices.iter()
                .map(|tensor| &*tensor.0.cast::<objc2::runtime::AnyObject>())
                .collect();
            
            // Create NSArray from references
            let array = NSArray::from_slice(&indices_refs);
            let ns_array: *mut AnyObject = array.as_ref() as *const NSArray<objc2::runtime::AnyObject> as *mut AnyObject;
            ns_array
        };
        
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, sparseToDenseWithIndicesTensors: input_tensors_array,
                valuesTensor: values.0,
                denseShapeTensor: dense_shape.0,
                name: name_obj
            ];
            
            let result = objc2::ffi::objc_retain(result as *mut _);
            MPSGraphTensor(result)
        }
    }
}