use crate::core::MPSDataType;
use crate::shape::MPSShape;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use std::fmt;
use std::ptr;

/// A wrapper for MPSGraphType objects
///
/// `MPSGraphType` is the base class for types used on tensors in MPSGraph.
pub struct MPSGraphType(pub(crate) *mut AnyObject);

// Implement Send + Sync for wrapper types
unsafe impl Send for MPSGraphType {}
unsafe impl Sync for MPSGraphType {}

impl MPSGraphType {
    /// Create a new MPSGraphType
    pub fn new() -> Self {
        unsafe {
            let class_name = c"MPSGraphType";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            let obj: *mut AnyObject = msg_send![cls, alloc];
            let initialized: *mut AnyObject = msg_send![obj, init];
            MPSGraphType(initialized)
        }
    }

    /// Returns a string describing this type
    pub fn description(&self) -> String {
        unsafe {
            let desc: *mut AnyObject = msg_send![self.0, description];
            if desc.is_null() {
                return String::from("<null>");
            }

            let utf8: *const i8 = msg_send![desc, UTF8String];
            if utf8.is_null() {
                return String::from("<null>");
            }

            std::ffi::CStr::from_ptr(utf8).to_string_lossy().to_string()
        }
    }
}

impl Drop for MPSGraphType {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraphType {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _);
                MPSGraphType(obj)
            } else {
                MPSGraphType(ptr::null_mut())
            }
        }
    }
}

impl fmt::Debug for MPSGraphType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MPSGraphType")
            .field("description", &self.description())
            .finish()
    }
}

/// A wrapper for MPSGraphShapedType objects
///
/// `MPSGraphShapedType` is a subclass of `MPSGraphType` that includes shape and data type information.
pub struct MPSGraphShapedType(pub(crate) *mut AnyObject);

// Implement Send + Sync for wrapper types
unsafe impl Send for MPSGraphShapedType {}
unsafe impl Sync for MPSGraphShapedType {}

impl MPSGraphShapedType {
    /// Create a new shaped type with shape and data type
    pub fn new(shape: &MPSShape, data_type: MPSDataType) -> Self {
        unsafe {
            let class_name = c"MPSGraphShapedType";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            let obj: *mut AnyObject = msg_send![cls, alloc];

            let data_type_val_32 = data_type as u32;
            let initialized: *mut AnyObject = msg_send![
                obj,
                initWithShape: shape.0,
                dataType: data_type_val_32
            ];

            MPSGraphShapedType(initialized)
        }
    }

    /// Returns the shape of this type
    pub fn shape(&self) -> MPSShape {
        unsafe {
            let shape_ptr: *mut AnyObject = msg_send![self.0, shape];
            if shape_ptr.is_null() {
                // Return an empty shape if null
                return MPSShape::from_slice(&[]);
            }

            let shape_ptr = objc2::ffi::objc_retain(shape_ptr as *mut _);
            MPSShape(shape_ptr)
        }
    }

    /// Returns the data type of this type
    pub fn data_type(&self) -> MPSDataType {
        unsafe {
            let data_type_val: u32 = msg_send![self.0, dataType];
            std::mem::transmute(data_type_val)
        }
    }

    /// Returns the rank of this type (calculated from shape)
    pub fn rank(&self) -> u64 {
        let shape = self.shape();
        if shape.dimensions().is_empty() {
            0
        } else {
            shape.dimensions().len() as u64
        }
    }

    /// Returns whether this type is ranked (has a specific rank) or unranked
    ///
    /// Note: This is a best-effort approximation as the isRanked method may not be available
    /// in all versions of MPSGraph
    pub fn is_ranked(&self) -> bool {
        let shape = self.shape();
        !shape.dimensions().is_empty()
    }

    /// Create a tensor type with the specified rank
    ///
    /// This creates a shaped type with dimensions of size 1 for each rank
    pub fn tensor_type_with_rank(rank: u64, data_type: MPSDataType) -> Self {
        // Create a shape with dimensions of size 1 for each rank
        let dimensions = vec![1usize; rank as usize];
        let shape = crate::shape::MPSShape::from_slice(&dimensions);

        // Create a shaped type with the shape and data type
        Self::new(&shape, data_type)
    }

    /// Create an unranked tensor type with the specified data type
    ///
    /// This creates a shaped type with an empty shape to represent an unranked tensor
    pub fn unranked_tensor_type(data_type: MPSDataType) -> Self {
        // Create a shaped type with an empty shape to represent an unranked tensor
        let shape = crate::shape::MPSShape::from_slice(&[]);

        // Create a shaped type with the empty shape and data type
        Self::new(&shape, data_type)
    }
}

impl Drop for MPSGraphShapedType {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraphShapedType {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _);
                MPSGraphShapedType(obj)
            } else {
                MPSGraphShapedType(ptr::null_mut())
            }
        }
    }
}

impl fmt::Debug for MPSGraphShapedType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MPSGraphShapedType")
            .field("shape", &self.shape())
            .field("data_type", &self.data_type())
            .field("rank", &self.rank())
            .field("is_ranked", &self.is_ranked())
            .finish()
    }
}

// MPS Graph execution options (keeping from original file)
#[derive(Debug, Clone, Copy)]
pub enum ExecutionMode {
    Synchronous,
    Asynchronous,
}

// Shape descriptor for tensors (keeping from original file)
#[derive(Debug, Clone)]
pub struct MPSShapeDescriptor {
    pub dimensions: Vec<u64>,
    pub data_type: MPSDataType,
}

impl MPSShapeDescriptor {
    pub fn new(dimensions: Vec<u64>, data_type: MPSDataType) -> Self {
        Self {
            dimensions,
            data_type,
        }
    }

    /// Get the total number of elements in this shape
    pub fn element_count(&self) -> u64 {
        self.dimensions.iter().fold(1, |acc, &dim| acc * dim)
    }

    /// Get the total size in bytes for this shape
    pub fn size_in_bytes(&self) -> u64 {
        self.element_count() * self.data_type.size_in_bytes() as u64
    }

    /// Create a new shape with different dimensions but same data type
    pub fn with_dimensions(&self, dimensions: Vec<u64>) -> Self {
        Self {
            dimensions,
            data_type: self.data_type,
        }
    }

    /// Create a new shape with different data type but same dimensions
    pub fn with_data_type(&self, data_type: MPSDataType) -> Self {
        Self {
            dimensions: self.dimensions.clone(),
            data_type,
        }
    }

    /// Create a scalar shape with the given data type
    pub fn scalar(data_type: MPSDataType) -> Self {
        Self {
            dimensions: vec![1],
            data_type,
        }
    }

    /// Create a vector shape with the given length and data type
    pub fn vector(length: u64, data_type: MPSDataType) -> Self {
        Self {
            dimensions: vec![length],
            data_type,
        }
    }

    /// Create a matrix shape with the given rows, columns and data type
    pub fn matrix(rows: u64, columns: u64, data_type: MPSDataType) -> Self {
        Self {
            dimensions: vec![rows, columns],
            data_type,
        }
    }
}
