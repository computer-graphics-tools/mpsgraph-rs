use objc2::runtime::{AnyObject};
use objc2::msg_send;
use objc2::rc::Retained;

// Import and re-export Foundation types for use in other modules
pub use objc2_foundation::{NSArray, NSNumber, NSString, NSDictionary, NSError, NSData};

// Helper extension trait to get raw AnyObject pointer from various types
pub trait AsRawObject {
    fn as_raw_object(&self) -> *mut AnyObject;
}

// Specialized implementation for NSString
impl AsRawObject for objc2::rc::Retained<NSString> {
    fn as_raw_object(&self) -> *mut AnyObject {
        // Create a new OurNSString from the content of this NSString
        // and return its raw pointer
        let string_content = self.to_string();
        let our_string = OurNSString::from_str(&string_content);
        our_string.0
    }
}

// Implementation for NSNumber
impl AsRawObject for objc2::rc::Retained<NSNumber> {
    fn as_raw_object(&self) -> *mut AnyObject {
        unsafe {
            let ptr: *mut AnyObject = std::mem::transmute::<&NSNumber, *mut AnyObject>(self.as_ref());
            objc2::ffi::objc_retain(ptr as *mut _);
            ptr
        }
    }
}

// Implementation for NSArray - generic version
impl<T: objc2::Message> AsRawObject for objc2::rc::Retained<NSArray<T>> {
    fn as_raw_object(&self) -> *mut AnyObject {
        unsafe {
            let ptr: *mut AnyObject = std::mem::transmute::<&NSArray<T>, *mut AnyObject>(self.as_ref());
            objc2::ffi::objc_retain(ptr as *mut _);
            ptr
        }
    }
}

// Implementation for NSDictionary - generic version
impl<K: objc2::Message, V: objc2::Message> AsRawObject for objc2::rc::Retained<NSDictionary<K, V>> {
    fn as_raw_object(&self) -> *mut AnyObject {
        unsafe {
            let ptr: *mut AnyObject = std::mem::transmute::<&NSDictionary<K, V>, *mut AnyObject>(self.as_ref());
            objc2::ffi::objc_retain(ptr as *mut _);
            ptr
        }
    }
}

// Implementation for NSData
impl AsRawObject for objc2::rc::Retained<NSData> {
    fn as_raw_object(&self) -> *mut AnyObject {
        unsafe {
            let ptr: *mut AnyObject = std::mem::transmute::<&NSData, *mut AnyObject>(self.as_ref());
            objc2::ffi::objc_retain(ptr as *mut _);
            ptr
        }
    }
}

// Implementations for other types will be added as needed
use std::ffi::c_void;
use std::fmt;
use std::ptr;
// We might need these later
// use std::ops::Deref;

/// Type for NSArray objects that represent shape vectors
pub struct MPSShape(pub(crate) *mut AnyObject);

impl MPSShape {
    /// Create an MPSShape from a slice of dimensions
    pub fn from_slice(dimensions: &[usize]) -> Self {
        unsafe {
            // Create NSNumbers for each dimension using numberWithUnsignedLongLong Objective-C method
            // (since new_uint is no longer available in objc2-foundation)
            let class_name = c"NSNumber";
            let numbers: Vec<Retained<NSNumber>> = if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                // Map directly to Retained<NSNumber> objects
                dimensions.iter()
                    .map(|&d| {
                        let number_ptr: *mut NSNumber = msg_send![cls, numberWithUnsignedLongLong:d as u64];
                        Retained::from_raw(number_ptr).unwrap_or_else(|| panic!("Failed to create NSNumber"))
                    })
                    .collect()
            } else {
                panic!("NSNumber class not found");
            };
            
            // Convert to slice of references
            let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
            
            // Create NSArray from the NSNumber objects
            let array = NSArray::from_slice(&number_refs);
            
            // Get pointer to the array and retain it manually
            let ptr: *mut AnyObject = std::mem::transmute::<&NSArray<NSNumber>, *mut AnyObject>(array.as_ref());
            objc2::ffi::objc_retain(ptr as *mut _);
            
            MPSShape(ptr)
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
            let ns_array: &NSArray<NSNumber> = &*(self.0 as *const objc2_foundation::NSArray<objc2_foundation::NSNumber>);
            ns_array.len()
        }
    }
    
    /// Get the dimensions as a vector
    pub fn dimensions(&self) -> Vec<usize> {
        unsafe {
            let ns_array: &NSArray<NSNumber> = &*(self.0 as *const objc2_foundation::NSArray<objc2_foundation::NSNumber>);
            let count = ns_array.len();
            let mut result = Vec::with_capacity(count);
            
            for i in 0..count {
                // Use objectAtIndex: method and convert to NSNumber
                let num_ptr: *mut NSNumber = msg_send![ns_array, objectAtIndex:i];
                let num_obj: &NSNumber = &*num_ptr;
                let value = num_obj.integerValue() as usize;
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
            // Convert to NSObject and release
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSShape {
    fn clone(&self) -> Self {
        unsafe {
            // Retain and return new instance
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
                MPSShape(obj)
            } else {
                MPSShape(ptr::null_mut())
            }
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
#[repr(u32)]  // Changed from u64 to u32 to match Objective-C's NSUInteger on 32-bit platforms
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
pub(crate) struct OurNSString(pub(crate) *mut AnyObject);

impl OurNSString {
    pub fn from_str(s: &str) -> Self {
        // Create a C string from the Rust string
        let c_string = std::ffi::CString::new(s).unwrap();
        unsafe {
            // Create NSString directly using Objective-C
            let class_name = c"NSString";
            let ns_string = if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                let string_ptr: *mut NSString = msg_send![cls, stringWithUTF8String:c_string.as_ptr()];
                Retained::from_raw(string_ptr).unwrap_or_else(|| panic!("Failed to create NSString"))
            } else {
                panic!("NSString class not found");
            };
            let ptr: *mut AnyObject = std::mem::transmute::<&NSString, *mut AnyObject>(ns_string.as_ref());
            
            // Retain manually since we're extracting the pointer
            objc2::ffi::objc_retain(ptr as *mut _);
            
            OurNSString(ptr as *mut _)
        }
    }
    
    #[allow(dead_code)]
    pub fn to_string(&self) -> String {
        unsafe {
            let ns_string_ref: &NSString = &*(self.0 as *const objc2_foundation::NSString);
            ns_string_ref.to_string()
        }
    }
}

// Implement AsRawObject for OurNSString
impl AsRawObject for OurNSString {
    fn as_raw_object(&self) -> *mut AnyObject {
        self.0
    }
}

impl Drop for OurNSString {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

/// Internal helper for creating NSDictionary objects
pub(crate) struct OurNSDictionary(pub(crate) *mut AnyObject);

impl OurNSDictionary {
    #[allow(dead_code)]
    pub fn new() -> Self {
        unsafe {
            // Use the dictionary constructor directly
            let dict = NSDictionary::<objc2::runtime::AnyObject, objc2::runtime::AnyObject>::dictionary();
            let ptr: *mut AnyObject = std::mem::transmute::<&NSDictionary<objc2::runtime::AnyObject, objc2::runtime::AnyObject>, *mut AnyObject>(dict.as_ref());
            
            // Retain it since we're extracting the pointer
            objc2::ffi::objc_retain(ptr as *mut _);
            
            OurNSDictionary(ptr)
        }
    }
    
    pub fn from_keys_and_objects(keys: &[*mut AnyObject], objects: &[*mut AnyObject]) -> Self {
        unsafe {
            assert_eq!(keys.len(), objects.len());
            
            if keys.is_empty() {
                return Self::new();
            }
            
            // First create slices for keys and objects that work with objc2-foundation
            let key_refs: Vec<&objc2::runtime::AnyObject> = keys.iter()
                .map(|&ptr| &*ptr.cast::<objc2::runtime::AnyObject>())
                .collect();
                
            let obj_refs: Vec<&objc2::runtime::AnyObject> = objects.iter()
                .map(|&ptr| &*ptr.cast::<objc2::runtime::AnyObject>())
                .collect();
                
            // In objc2-foundation, NSDictionary::from_slices requires CopyingHelper trait
            // but AnyObject doesn't implement that. Let's create a new dictionary directly with Objective-C
            
            // Create dictionary using the dictionaryWithObjects:forKeys:count: class method
            let cls_name = c"NSDictionary";
            let dict_ptr: *mut AnyObject = if let Some(cls) = objc2::runtime::AnyClass::get(cls_name) {
                if !key_refs.is_empty() {
                    msg_send![cls, dictionaryWithObjects: obj_refs.as_ptr(), 
                              forKeys: key_refs.as_ptr(), 
                              count: key_refs.len()]
                } else {
                    msg_send![cls, dictionary]
                }
            } else {
                panic!("NSDictionary class not found");
            };
            
            // The object is already retained by the Objective-C runtime
            // Just use the pointer directly
            let ptr = dict_ptr;
            
            OurNSDictionary(ptr)
        }
    }
}

impl Drop for OurNSDictionary {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

/// Internal helper for NSArray objects
pub(crate) struct OurNSArray(pub(crate) *mut AnyObject);

impl OurNSArray {
    pub fn from_objects(objects: &[*mut AnyObject]) -> Self {
        unsafe {
            if objects.is_empty() {
                // Use the empty array constructor directly
                let array = NSArray::<objc2::runtime::AnyObject>::array();
                let ptr: *mut AnyObject = std::mem::transmute::<&NSArray<objc2::runtime::AnyObject>, *mut AnyObject>(array.as_ref());
                
                // Retain it since we're extracting the pointer
                objc2::ffi::objc_retain(ptr as *mut _);
                
                return OurNSArray(ptr);
            }
            
            // Convert raw pointers to references to AnyObject
            let refs: Vec<&objc2::runtime::AnyObject> = objects.iter()
                .map(|&p| &*p.cast::<objc2::runtime::AnyObject>())
                .collect();
            
            // Create array from references
            let array = NSArray::from_slice(&refs);
            let ptr: *mut AnyObject = std::mem::transmute::<&NSArray<objc2::runtime::AnyObject>, *mut AnyObject>(array.as_ref());
            
            // Retain it since we're extracting the pointer
            objc2::ffi::objc_retain(ptr as *mut _);
            
            OurNSArray(ptr)
        }
    }
    
    #[allow(dead_code)]
    pub fn from_slice<T>(objects: &[T]) -> Self
    where 
        T: AsRef<crate::tensor::MPSGraphTensor>,
    {
        // This is safe because we're only accessing the 0 field which is a raw pointer
        // and we're not dereferencing it, just passing it to from_objects
        let raw_objects: Vec<*mut AnyObject> = objects
            .iter()
            .map(|obj| {
                let tensor = obj.as_ref();
                
                tensor.0
            })
            .collect();
        
        Self::from_objects(&raw_objects)
    }
    
    /// Creates an NSArray from a slice of i64 values
    pub fn from_i64_slice(integers: &[i64]) -> Self {
        unsafe {
            // Create NSNumbers for each integer
            let class_name = c"NSNumber";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap();
            
            let ns_numbers: Vec<*mut AnyObject> = integers.iter()
                .map(|&i| {
                    let number: *mut AnyObject = msg_send![cls, numberWithLongLong:i];
                    objc2::ffi::objc_retain(number as *mut _);
                    number
                })
                .collect();
            
            Self::from_objects(&ns_numbers)
        }
    }
    
    #[allow(dead_code)]
    pub fn count(&self) -> usize {
        unsafe {
            let array: &NSArray<objc2::runtime::AnyObject> = &*(self.0 as *const objc2_foundation::NSArray);
            array.len()
        }
    }
    
    #[allow(dead_code)]
    pub fn object_at(&self, index: usize) -> *mut AnyObject {
        unsafe {
            let array: &NSArray<objc2::runtime::AnyObject> = &*(self.0 as *const objc2_foundation::NSArray);
            if index >= array.len() {
                return ptr::null_mut();
            }
            
            // Get the object and convert it to a raw pointer using objectAtIndex instead of get
            let obj: *mut AnyObject = msg_send![array, objectAtIndex:index];
            obj
        }
    }
}

impl Drop for OurNSArray {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
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
pub struct OurNSError(pub(crate) *mut AnyObject);

impl OurNSError {
    pub fn new(error_ptr: *mut AnyObject) -> Self {
        unsafe {
            if !error_ptr.is_null() {
                objc2::ffi::objc_retain(error_ptr as *mut _);
            }
            OurNSError(error_ptr)
        }
    }
    
    pub fn localized_description(&self) -> String {
        unsafe {
            if self.0.is_null() {
                return String::from("<null error>");
            }
            
            let ns_error: &NSError = &*(self.0 as *const objc2_foundation::NSError);
            ns_error.localizedDescription().to_string()
        }
    }
    
    // Keeping for backward compatibility
    #[allow(non_snake_case)]
    pub fn localizedDescription(&self) -> String {
        self.localized_description()
    }
}

impl Drop for OurNSError {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for OurNSError {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
                OurNSError(obj)
            } else {
                OurNSError(ptr::null_mut())
            }
        }
    }
}

impl fmt::Debug for OurNSError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NSError: {}", self.localizedDescription())
    }
}