use objc2::runtime::{AnyObject};
use objc2::msg_send;
use objc2::rc::Retained;

// Import and re-export Foundation types for use in other modules
pub use objc2_foundation::{NSArray, NSNumber, NSString, NSDictionary, NSError, NSData};

// Helper extension trait to get raw AnyObject pointer from various types
// Returns a raw pointer with +1 retain count (caller responsible for releasing)
pub trait AsRawObject {
    fn as_raw_object(&self) -> *mut AnyObject;
}

// Generic implementation for all Retained<T> where T: Message
impl<T: objc2::Message> AsRawObject for objc2::rc::Retained<T> {
    fn as_raw_object(&self) -> *mut AnyObject {
        unsafe {
            let ptr: *const T = objc2::rc::Retained::as_ptr(self);
            let ptr = ptr as *mut AnyObject;
            objc2::ffi::objc_retain(ptr as *mut _);
            ptr
        }
    }
}

// Helper function to create NSArray from AnyObject pointers
pub fn create_ns_array_from_pointers(objects: &[*mut AnyObject]) -> *mut AnyObject {
    unsafe {
        // Convert raw pointers to references
        let refs: Vec<&objc2::runtime::AnyObject> = objects.iter()
            .map(|&p| &*p.cast::<objc2::runtime::AnyObject>())
            .collect();
        
        // Create array from references
        let array = NSArray::from_slice(&refs);
        let ptr: *mut AnyObject = std::mem::transmute::<&NSArray<objc2::runtime::AnyObject>, 
                                                     *mut AnyObject>(array.as_ref());
        
        objc2::ffi::objc_retain(ptr as *mut _);
        ptr
    }
}

// Helper function to create NSArray from i64 slice
pub fn create_ns_array_from_i64_slice(values: &[i64]) -> *mut AnyObject {
    unsafe {
        // Create NSNumbers for each value
        let class_name = c"NSNumber";
        if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
            // Create NSNumber objects for each value
            let numbers: Vec<objc2::rc::Retained<NSNumber>> = values.iter()
                .map(|&value| {
                    let number_ptr: *mut NSNumber = msg_send![cls, numberWithLongLong:value];
                    objc2::rc::Retained::from_raw(number_ptr).unwrap_or_else(|| panic!("Failed to create NSNumber"))
                })
                .collect();
            
            // Convert to slice of references
            let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
            
            // Create NSArray from the NSNumber objects
            let array = NSArray::from_slice(&number_refs);
            
            // Get pointer to the array and retain it manually
            let ptr: *mut AnyObject = std::mem::transmute::<&NSArray<NSNumber>, *mut AnyObject>(array.as_ref());
            objc2::ffi::objc_retain(ptr as *mut _);
            
            ptr
        } else {
            panic!("NSNumber class not found");
        }
    }
}

// Helper function to create NSDictionary from keys and objects pointers
pub fn create_ns_dictionary_from_pointers(keys: &[*mut AnyObject], objects: &[*mut AnyObject]) -> *mut AnyObject {
    unsafe {
        if keys.len() != objects.len() {
            panic!("keys and objects must have the same length");
        }
        
        // Create references needed for Objective-C
        let key_refs: Vec<&objc2::runtime::AnyObject> = keys.iter()
            .map(|&ptr| &*ptr.cast::<objc2::runtime::AnyObject>())
            .collect();
                
        let obj_refs: Vec<&objc2::runtime::AnyObject> = objects.iter()
            .map(|&ptr| &*ptr.cast::<objc2::runtime::AnyObject>())
            .collect();
        
        // Create dictionary using Objective-C method
        let cls = objc2::runtime::AnyClass::get(c"NSDictionary").unwrap();
        let dict_ptr: *mut AnyObject = msg_send![cls, 
            dictionaryWithObjects: obj_refs.as_ptr(), 
            forKeys: key_refs.as_ptr(), 
            count: key_refs.len()
        ];
        
        objc2::ffi::objc_retain(dict_ptr as *mut _);
        dict_ptr
    }
}

// Implementations for other types will be added as needed
use std::ffi::c_void;
use std::fmt;
use std::ptr;
// We might need these later
// use std::ops::Deref;

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