use objc::runtime::Object;
use objc::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, MPSShape, MPSDataType};

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
pub struct MPSGraphCreateSparseOpDescriptor(pub(crate) *mut Object);

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
        storage_type: MPSGraphSparseStorageType,
        data_type: MPSDataType,
    ) -> Self {
        unsafe {
            let cls = objc::runtime::Class::get("MPSGraphCreateSparseOpDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![
                cls,
                descriptorWithStorageType:storage_type as u64
                dataType:data_type as u64
            ];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphCreateSparseOpDescriptor(descriptor)
        }
    }
    
    /// Sets the sparse storage type
    pub fn set_sparse_storage_type(&self, storage_type: MPSGraphSparseStorageType) {
        unsafe {
            let _: () = msg_send![self.0, setSparseStorageType:storage_type as u64];
        }
    }
    
    /// Sets the data type
    pub fn set_data_type(&self, data_type: MPSDataType) {
        unsafe {
            let _: () = msg_send![self.0, setDataType:data_type as u64];
        }
    }
}

impl Drop for MPSGraphCreateSparseOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

impl Clone for MPSGraphCreateSparseOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let desc: *mut Object = msg_send![self.0, copy];
            MPSGraphCreateSparseOpDescriptor(desc)
        }
    }
}

/// Sparse operations for MPSGraph
impl MPSGraph {
    /// Creates a sparse tensor representation.
    ///
    /// # Arguments
    ///
    /// * `storage_type` - The storage format of the sparse tensor
    /// * `input_tensors` - Array of input tensors as [sparse_vals, index_tensor0, index_tensor1]
    ///   - sparse_vals: The non-zero values in the matrix
    ///   - index_tensor0: For COO: x index, For CSC: rowIndex, For CSR: colIndex
    ///   - index_tensor1: For COO: y index, For CSC: colStarts, For CSR: rowStarts
    /// * `shape` - The shape of the sparse tensor
    /// * `data_type` - The data type of the sparse tensor
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object
    pub fn sparse_tensor(
        &self,
        storage_type: MPSGraphSparseStorageType,
        input_tensors: &[&MPSGraphTensor],
        shape: &MPSShape,
        data_type: MPSDataType,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        // Create NSArray of input tensors
        let input_tensors_array = unsafe {
            let cls = objc::runtime::Class::get("NSArray").unwrap();
            let mut ns_objects: Vec<*mut Object> = Vec::with_capacity(input_tensors.len());
            
            for tensor in input_tensors {
                ns_objects.push(tensor.0);
            }
            
            let count = ns_objects.len();
            let ns_array: *mut Object = msg_send![
                cls,
                arrayWithObjects:ns_objects.as_ptr()
                count:count
            ];
            ns_array
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                sparseTensorWithType:storage_type as u64
                tensors:input_tensors_array
                shape:shape.0
                dataType:data_type as u64
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a sparse tensor representation using a descriptor.
    ///
    /// # Arguments
    ///
    /// * `descriptor` - The descriptor for sparse tensor creation
    /// * `input_tensors` - Array of input tensors as [sparse_vals, index_tensor0, index_tensor1]
    ///   - sparse_vals: The non-zero values in the matrix
    ///   - index_tensor0: For COO: x index, For CSC: rowIndex, For CSR: colIndex
    ///   - index_tensor1: For COO: y index, For CSC: colStarts, For CSR: rowStarts
    /// * `shape` - The shape of the sparse tensor
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object
    pub fn sparse_tensor_with_descriptor(
        &self,
        descriptor: &MPSGraphCreateSparseOpDescriptor,
        input_tensors: &[&MPSGraphTensor],
        shape: &MPSShape,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        // Create NSArray of input tensors
        let input_tensors_array = unsafe {
            let cls = objc::runtime::Class::get("NSArray").unwrap();
            let mut ns_objects: Vec<*mut Object> = Vec::with_capacity(input_tensors.len());
            
            for tensor in input_tensors {
                ns_objects.push(tensor.0);
            }
            
            let count = ns_objects.len();
            let ns_array: *mut Object = msg_send![
                cls,
                arrayWithObjects:ns_objects.as_ptr()
                count:count
            ];
            ns_array
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                sparseTensorWithDescriptor:descriptor.0
                tensors:input_tensors_array
                shape:shape.0
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
    fn test_sparse_tensor_coo() {
        if should_skip_test("test_sparse_tensor_coo") {
            return;
        }
        
        let graph = MPSGraph::new();
        
        // Create a simple 3x3 sparse matrix in COO format
        // With non-zero values at (0,0)=1.0, (1,1)=2.0, (2,2)=3.0
        
        // Values
        let values_shape = MPSShape::from_slice(&[3]);
        let values_data = vec![1.0f32, 2.0, 3.0];
        let values = graph.placeholder(&values_shape, MPSDataType::Float32, None);
        
        // Row indices
        let row_indices_shape = MPSShape::from_slice(&[3]);
        let row_indices_data = vec![0i32, 1, 2]; // x/row indices
        let row_indices = graph.placeholder(&row_indices_shape, MPSDataType::Int32, None);
        
        // Column indices
        let col_indices_shape = MPSShape::from_slice(&[3]);
        let col_indices_data = vec![0i32, 1, 2]; // y/column indices
        let col_indices = graph.placeholder(&col_indices_shape, MPSDataType::Int32, None);
        
        // Create sparse tensor in COO format
        let sparse_shape = MPSShape::from_slice(&[3, 3]);
        let sparse_tensor = graph.sparse_tensor(
            MPSGraphSparseStorageType::COO,
            &[&values, &row_indices, &col_indices],
            &sparse_shape,
            MPSDataType::Float32,
            Some("sparse_coo")
        );
        
        // Run the graph
        let mut feeds = HashMap::new();
        feeds.insert(&values, crate::MPSGraphTensorData::new(&values_data, &[3], MPSDataType::Float32));
        feeds.insert(&row_indices, crate::MPSGraphTensorData::new(&row_indices_data, &[3], MPSDataType::Int32));
        feeds.insert(&col_indices, crate::MPSGraphTensorData::new(&col_indices_data, &[3], MPSDataType::Int32));
        
        let results = graph.run(feeds, &[&sparse_tensor]);
        
        // The sparse tensor result is in an internal format, so we can't easily check its values.
        // We just assert that we get a valid result back.
        assert!(results.contains_key(&sparse_tensor));
    }
    
    #[test]
    fn test_sparse_tensor_with_descriptor() {
        if should_skip_test("test_sparse_tensor_with_descriptor") {
            return;
        }
        
        let graph = MPSGraph::new();
        
        // Create a descriptor for CSR format
        let descriptor = MPSGraphCreateSparseOpDescriptor::new(
            MPSGraphSparseStorageType::CSR,
            MPSDataType::Float32
        );
        
        // Create a simple 3x3 sparse matrix in CSR format
        // With non-zero values at (0,0)=1.0, (1,1)=2.0, (2,2)=3.0
        
        // Values
        let values_shape = MPSShape::from_slice(&[3]);
        let values_data = vec![1.0f32, 2.0, 3.0];
        let values = graph.placeholder(&values_shape, MPSDataType::Float32, None);
        
        // Column indices (for CSR)
        let col_indices_shape = MPSShape::from_slice(&[3]);
        let col_indices_data = vec![0i32, 1, 2]; // column indices
        let col_indices = graph.placeholder(&col_indices_shape, MPSDataType::Int32, None);
        
        // Row starts (for CSR)
        let row_starts_shape = MPSShape::from_slice(&[4]); // n+1 for n rows
        let row_starts_data = vec![0i32, 1, 2, 3]; // pointers to start of each row
        let row_starts = graph.placeholder(&row_starts_shape, MPSDataType::Int32, None);
        
        // Create sparse tensor with descriptor
        let sparse_shape = MPSShape::from_slice(&[3, 3]);
        let sparse_tensor = graph.sparse_tensor_with_descriptor(
            &descriptor,
            &[&values, &col_indices, &row_starts],
            &sparse_shape,
            Some("sparse_csr")
        );
        
        // Run the graph
        let mut feeds = HashMap::new();
        feeds.insert(&values, crate::MPSGraphTensorData::new(&values_data, &[3], MPSDataType::Float32));
        feeds.insert(&col_indices, crate::MPSGraphTensorData::new(&col_indices_data, &[3], MPSDataType::Int32));
        feeds.insert(&row_starts, crate::MPSGraphTensorData::new(&row_starts_data, &[4], MPSDataType::Int32));
        
        let results = graph.run(feeds, &[&sparse_tensor]);
        
        // Just assert that we get a valid result back
        assert!(results.contains_key(&sparse_tensor));
    }
}