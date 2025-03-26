use objc::runtime::Object;
use objc::runtime::Class;
use std::ffi::c_void;
use std::fmt;
// We might need these later
// use std::ptr;
// use std::ops::Deref;

/// Type for NSArray objects that represent shape vectors
pub struct MPSShape(pub(crate) *mut Object);

impl MPSShape {
    /// Create an MPSShape from a slice of dimensions
    pub fn from_slice(dimensions: &[usize]) -> Self {
        unsafe {
            let cls = Class::get("NSNumber").unwrap();
            let dimensions: Vec<*mut Object> = dimensions.iter()
                .map(|&d| {
                    let obj: *mut Object = msg_send![cls, alloc];
                    let obj: *mut Object = msg_send![obj, initWithUnsignedLongLong:d];
                    obj
                })
                .collect();
            
            let array_cls = Class::get("NSArray").unwrap();
            let shape_array: *mut Object = msg_send![array_cls, alloc];
            let shape_array: *mut Object = msg_send![shape_array,
                initWithObjects:dimensions.as_ptr()
                count:dimensions.len()
            ];
            
            // Release NSNumber objects
            for d in dimensions {
                let _: () = msg_send![d, release];
            }
            
            MPSShape(shape_array)
        }
    }
    
    /// Create an MPSShape representing a scalar
    pub fn scalar() -> Self {
        Self::from_slice(&[1])
    }
    
    /// Create an MPSShape representing a vector
    pub fn vector(length: usize) -> Self {
        Self::from_slice(&[length])
    }
    
    /// Create an MPSShape representing a matrix
    pub fn matrix(rows: usize, columns: usize) -> Self {
        Self::from_slice(&[rows, columns])
    }
    
    /// Create an MPSShape representing a 3D tensor
    pub fn tensor3d(dim1: usize, dim2: usize, dim3: usize) -> Self {
        Self::from_slice(&[dim1, dim2, dim3])
    }
    
    /// Create an MPSShape representing a 4D tensor
    pub fn tensor4d(dim1: usize, dim2: usize, dim3: usize, dim4: usize) -> Self {
        Self::from_slice(&[dim1, dim2, dim3, dim4])
    }
    
    /// Get the number of dimensions (rank) of the shape
    pub fn rank(&self) -> usize {
        unsafe {
            let count: usize = msg_send![self.0, count];
            count
        }
    }
    
    /// Get the dimensions as a vector
    pub fn dimensions(&self) -> Vec<usize> {
        unsafe {
            let count: usize = msg_send![self.0, count];
            let mut result = Vec::with_capacity(count);
            
            for i in 0..count {
                let num: *mut Object = msg_send![self.0, objectAtIndex:i];
                let value: usize = msg_send![num, unsignedLongLongValue];
                result.push(value);
            }
            
            result
        }
    }
    
    /// Get the total number of elements in this shape
    pub fn element_count(&self) -> usize {
        self.dimensions().iter().product()
    }
}

impl Drop for MPSShape {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

impl Clone for MPSShape {
    fn clone(&self) -> Self {
        unsafe {
            let obj: *mut Object = msg_send![self.0, retain];
            MPSShape(obj)
        }
    }
}

impl fmt::Debug for MPSShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MPSShape")
            .field(&self.dimensions())
            .finish()
    }
}

/// MPS Graph data types
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
    /// Returns the size in bytes for this data type
    pub fn size_in_bytes(&self) -> usize {
        match self {
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

/// Options for MPSGraph execution
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MPSGraphOptions {
    /// No Options
    None = 0,
    /// The graph synchronizes results to the CPU using a blit encoder if on a discrete GPU at the end of execution
    SynchronizeResults = 1,
    /// The framework prints more logging info
    Verbose = 2,
    /// Default options (same as SynchronizeResults)
    Default = 3,
}

/// Optimization levels for MPSGraph compilation
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MPSGraphOptimization {
    /// Graph performs core optimizations only
    Level0 = 0,
    /// Graph performs additional optimizations
    Level1 = 1,
}

/// Optimization profile for MPSGraph
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MPSGraphOptimizationProfile {
    /// Default, graph optimized for performance
    Performance = 0,
    /// Graph optimized for power efficiency
    PowerEfficiency = 1,
}

/// Execution events that can be used with shared events
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MPSGraphExecutionStage {
    /// Stage when execution of the graph completes
    Completed = 0,
}

/// Internal helper to create NSString objects with proper memory management
pub(crate) struct NSString(pub(crate) *mut Object);

impl NSString {
    pub fn from_str(s: &str) -> Self {
        unsafe {
            let cls = Class::get("NSString").unwrap();
            let s = std::ffi::CString::new(s).unwrap();
            let obj: *mut Object = msg_send![cls, alloc];
            let obj: *mut Object = msg_send![obj, initWithUTF8String:s.as_ptr()];
            NSString(obj)
        }
    }
}

impl Drop for NSString {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

/// Internal helper for creating NSDictionary objects
pub(crate) struct NSDictionary(pub(crate) *mut Object);

impl NSDictionary {
    #[allow(dead_code)]
    pub fn new() -> Self {
        unsafe {
            let cls = Class::get("NSDictionary").unwrap();
            let obj: *mut Object = msg_send![cls, new];
            NSDictionary(obj)
        }
    }
    
    pub fn from_keys_and_objects(keys: &[*mut Object], objects: &[*mut Object]) -> Self {
        unsafe {
            assert_eq!(keys.len(), objects.len());
            let cls = Class::get("NSDictionary").unwrap();
            let obj: *mut Object = msg_send![cls, alloc];
            let obj: *mut Object = msg_send![obj, 
                initWithObjects:objects.as_ptr() 
                forKeys:keys.as_ptr() 
                count:keys.len()
            ];
            NSDictionary(obj)
        }
    }
}

impl Drop for NSDictionary {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

/// Internal helper for NSArray objects
pub(crate) struct NSArray(pub(crate) *mut Object);

impl NSArray {
    pub fn from_objects(objects: &[*mut Object]) -> Self {
        unsafe {
            let cls = Class::get("NSArray").unwrap();
            let obj: *mut Object = msg_send![cls, alloc];
            let obj: *mut Object = msg_send![obj, 
                initWithObjects:objects.as_ptr() 
                count:objects.len()
            ];
            NSArray(obj)
        }
    }
    
    pub fn from_slice<T>(objects: &[T]) -> Self
    where 
        T: AsRef<crate::tensor::MPSGraphTensor>
    {
        // This is safe because we're only accessing the 0 field which is a raw pointer
        // and we're not dereferencing it, just passing it to from_objects
        let raw_objects: Vec<*mut Object> = objects
            .iter()
            .map(|obj| {
                let tensor = obj.as_ref();
                let ptr = tensor.0;
                ptr
            })
            .collect();
        
        Self::from_objects(&raw_objects)
    }
    
    #[allow(dead_code)]
    pub fn count(&self) -> usize {
        unsafe {
            msg_send![self.0, count]
        }
    }
    
    #[allow(dead_code)]
    pub fn object_at(&self, index: usize) -> *mut Object {
        unsafe {
            msg_send![self.0, objectAtIndex:index]
        }
    }
}

impl Drop for NSArray {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

/// Type to wrap Objective-C block objects for callbacks
#[repr(C)]
#[allow(dead_code)]
pub(crate) struct Block<F> {
    pub isa: *const c_void,
    pub flags: i32,
    pub reserved: i32,
    pub invoke: F,
}

/// Wrapper type for NSError
pub struct NSError(pub(crate) *mut Object);

impl NSError {
    pub fn localized_description(&self) -> String {
        unsafe {
            let desc: *mut Object = msg_send![self.0, localizedDescription];
            let utf8: *const i8 = msg_send![desc, UTF8String];
            std::ffi::CStr::from_ptr(utf8).to_string_lossy().to_string()
        }
    }
    
    // Keeping for backward compatibility
    #[allow(non_snake_case)]
    pub fn localizedDescription(&self) -> String {
        self.localized_description()
    }
}

impl Drop for NSError {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

impl fmt::Debug for NSError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NSError: {}", self.localizedDescription())
    }
}