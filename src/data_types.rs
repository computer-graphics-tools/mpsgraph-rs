use objc2::runtime::AnyObject;

// MPS Graph execution options
#[derive(Debug, Clone, Copy)]
pub enum ExecutionMode {
    Synchronous,
    Asynchronous,
}

// Define the base types for MPS Graph objects
pub struct MPSGraphType(pub(crate) *mut AnyObject);
pub struct MPSGraphTensorType(pub(crate) *mut AnyObject);
pub struct MPSGraphOperationType(pub(crate) *mut AnyObject);
pub struct MPSGraphExecutableType(pub(crate) *mut AnyObject);

// Common data types used in MPS Graph
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MPSDataType {
    Invalid = 0,
    
    // Floating point types
    Float32 = 0x10000000 | 32,
    Float16 = 0x10000000 | 16,
    Float64 = 0x10000000 | 64,
    
    // Signed integer types
    Int8 = 0x20000000 | 8,
    Int16 = 0x20000000 | 16,
    Int32 = 0x20000000 | 32,
    Int64 = 0x20000000 | 64,
    
    // Unsigned integer types
    UInt8 = 8,
    UInt16 = 16,
    UInt32 = 32,
    UInt64 = 64,
    
    // Boolean type
    Bool = 0x40000000 | 8,
    
    // Complex types
    Complex32 = 0x10000000 | 0x80000000 | 32,
    Complex64 = 0x10000000 | 0x80000000 | 64,
}

impl MPSDataType {
    pub fn size_in_bytes(&self) -> usize {
        match *self {
            MPSDataType::Float16 => 2,
            MPSDataType::Float32 => 4,
            MPSDataType::Float64 => 8,
            MPSDataType::Int8 => 1,
            MPSDataType::Int16 => 2,
            MPSDataType::Int32 => 4,
            MPSDataType::Int64 => 8,
            MPSDataType::UInt8 => 1,
            MPSDataType::UInt16 => 2,
            MPSDataType::UInt32 => 4,
            MPSDataType::UInt64 => 8,
            MPSDataType::Bool => 1,
            MPSDataType::Complex32 => 8,  // 2 * Float32
            MPSDataType::Complex64 => 16, // 2 * Float64
            MPSDataType::Invalid => 0,
        }
    }
}

// Shape descriptor for tensors
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
            data_type:  self.data_type,
        }
    }
    
    /// Create a new shape with different data type but same dimensions
    pub fn with_data_type(&self, data_type: MPSDataType) -> Self {
        Self {
            dimensions:  self.dimensions.clone(),
            data_type,
        }
    }
    
    /// Create a scalar shape with the given data type
    pub fn scalar(data_type: MPSDataType) -> Self {
        Self {
            dimensions:  vec![1],
            data_type,
        }
    }
    
    /// Create a vector shape with the given length and data type
    pub fn vector(length: u64, data_type: MPSDataType) -> Self {
        Self {
            dimensions:  vec![length],
            data_type,
        }
    }
    
    /// Create a matrix shape with the given rows, columns and data type
    pub fn matrix(rows: u64, columns: u64, data_type: MPSDataType) -> Self {
        Self {
            dimensions:  vec![rows, columns],
            data_type,
        }
    }
}