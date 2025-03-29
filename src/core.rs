use objc2::msg_send;
use objc2::runtime::AnyObject;

// Import and re-export Foundation types for use in other modules
pub use objc2_foundation::{NSArray, NSData, NSDictionary, NSError, NSNumber, NSString};

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
        let refs: Vec<&objc2::runtime::AnyObject> = objects
            .iter()
            .map(|&p| &*p.cast::<objc2::runtime::AnyObject>())
            .collect();

        // Create array from references
        let array = NSArray::from_slice(&refs);
        let ptr: *mut AnyObject = std::mem::transmute::<
            &NSArray<objc2::runtime::AnyObject>,
            *mut AnyObject,
        >(array.as_ref());

        objc2::ffi::objc_retain(ptr as *mut _);
        ptr
    }
}

// Helper function to create NSArray from i64 slice
pub fn create_ns_array_from_i64_slice(values: &[i64]) -> *mut AnyObject {
    unsafe {
        // Create NSNumber objects for each value using objc2-foundation's NSNumber
        let numbers: Vec<objc2::rc::Retained<NSNumber>> = values
            .iter()
            .map(|&value| NSNumber::new_i64(value))
            .collect();

        // Convert to slice of references
        let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();

        // Create NSArray from the NSNumber objects
        let array = NSArray::from_slice(&number_refs);

        // Get pointer to the array and retain it manually
        let ptr: *mut AnyObject =
            std::mem::transmute::<&NSArray<NSNumber>, *mut AnyObject>(array.as_ref());
        objc2::ffi::objc_retain(ptr as *mut _);

        ptr
    }
}

// Helper function to create NSDictionary from keys and objects pointers
pub fn create_ns_dictionary_from_pointers(
    keys: &[*mut AnyObject],
    objects: &[*mut AnyObject],
) -> *mut AnyObject {
    unsafe {
        if keys.len() != objects.len() {
            panic!("keys and objects must have the same length");
        }

        // Create references needed for Objective-C
        let key_refs: Vec<&objc2::runtime::AnyObject> = keys
            .iter()
            .map(|&ptr| &*ptr.cast::<objc2::runtime::AnyObject>())
            .collect();

        let obj_refs: Vec<&objc2::runtime::AnyObject> = objects
            .iter()
            .map(|&ptr| &*ptr.cast::<objc2::runtime::AnyObject>())
            .collect();

        // Get the NSDictionary class
        let cls = objc2::runtime::AnyClass::get(c"NSDictionary").unwrap();

        // Create dictionary using Objective-C method
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
// We might need these later
// use std::ops::Deref;

/// MPS Graph data types
#[repr(u32)] // Changed from u64 to u32 to match Objective-C's NSUInteger on 32-bit platforms
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
    /// Converts a u32 value to an MPSDataType
    pub fn from_u32(value: u32) -> Self {
        match value {
            // Since we're given the exact value from ObjC, check for exact enum values
            val if val == MPSDataType::Float32 as u32 => MPSDataType::Float32,
            val if val == MPSDataType::Float16 as u32 => MPSDataType::Float16,
            val if val == MPSDataType::Float64 as u32 => MPSDataType::Float64,
            val if val == MPSDataType::Int8 as u32 => MPSDataType::Int8,
            val if val == MPSDataType::Int16 as u32 => MPSDataType::Int16,
            val if val == MPSDataType::Int32 as u32 => MPSDataType::Int32,
            val if val == MPSDataType::Int64 as u32 => MPSDataType::Int64,
            val if val == MPSDataType::UInt8 as u32 => MPSDataType::UInt8,
            val if val == MPSDataType::UInt16 as u32 => MPSDataType::UInt16,
            val if val == MPSDataType::UInt32 as u32 => MPSDataType::UInt32,
            val if val == MPSDataType::UInt64 as u32 => MPSDataType::UInt64,
            val if val == MPSDataType::Bool as u32 => MPSDataType::Bool,
            val if val == MPSDataType::Complex32 as u32 => MPSDataType::Complex32,
            val if val == MPSDataType::Complex64 as u32 => MPSDataType::Complex64,
            _ => MPSDataType::Invalid,
        }
    }

    /// Returns the u32 representation of this data type
    pub fn as_u32(&self) -> u32 {
        *self as u32
    }
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
