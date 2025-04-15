use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;

// Import Foundation types for NSArray and NSNumber
pub use objc2_foundation::{NSArray, NSNumber};

use std::fmt;
use std::ptr;

use crate::core::AsRawObject;

/// Type for Metal Performance Shaders Graph shape objects (NSArray) that represent tensor dimensions
pub struct Shape(pub(crate) *mut AnyObject);

// Implement AsRawObject for Shape
impl AsRawObject for Shape {
    fn as_raw_object(&self) -> *mut AnyObject {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_retain(self.0 as *mut _);
            }
            self.0
        }
    }
}

impl Shape {
    /// Create a Shape from a slice of dimensions
    pub fn from_slice(dimensions: &[usize]) -> Self {
        unsafe {
            // Create NSNumbers for each dimension using numberWithUnsignedLongLong Objective-C method
            // (since new_uint is no longer available in objc2-foundation)
            let class_name = c"NSNumber";
            let numbers: Vec<Retained<NSNumber>> =
                if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                    // Map directly to Retained<NSNumber> objects
                    dimensions
                        .iter()
                        .map(|&d| {
                            let number_ptr: *mut NSNumber =
                                msg_send![cls, numberWithUnsignedLongLong:d as u64];
                            Retained::from_raw(number_ptr)
                                .unwrap_or_else(|| panic!("Failed to create NSNumber"))
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
            let ptr: *mut AnyObject = array.as_ref()
                as *const objc2_foundation::NSArray<objc2_foundation::NSNumber>
                as *mut AnyObject;
            objc2::ffi::objc_retain(ptr as *mut _);

            Shape(ptr)
        }
    }

    /// Create a Shape representing a scalar
    pub fn scalar() -> Self {
        Self::from_slice(&[1])
    }

    /// Create a Shape representing a vector
    pub fn vector(length: usize) -> Self {
        Self::from_slice(&[length])
    }

    /// Create a Shape representing a matrix
    pub fn matrix(rows: usize, columns: usize) -> Self {
        Self::from_slice(&[rows, columns])
    }

    /// Create a Shape representing a 3D tensor
    pub fn tensor3d(dim1: usize, dim2: usize, dim3: usize) -> Self {
        Self::from_slice(&[dim1, dim2, dim3])
    }

    /// Create a Shape representing a 4D tensor
    pub fn tensor4d(dim1: usize, dim2: usize, dim3: usize, dim4: usize) -> Self {
        Self::from_slice(&[dim1, dim2, dim3, dim4])
    }

    /// Get the number of dimensions (rank) of the shape
    pub fn rank(&self) -> usize {
        unsafe {
            let ns_array: &NSArray<NSNumber> =
                &*(self.0 as *const objc2_foundation::NSArray<objc2_foundation::NSNumber>);
            ns_array.len()
        }
    }

    /// Get the dimensions as a vector
    pub fn dimensions(&self) -> Vec<usize> {
        unsafe {
            let ns_array: &NSArray<NSNumber> =
                &*(self.0 as *const objc2_foundation::NSArray<objc2_foundation::NSNumber>);
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

impl Drop for Shape {
    fn drop(&mut self) {
        unsafe {
            // Convert to NSObject and release
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for Shape {
    fn clone(&self) -> Self {
        unsafe {
            // Retain and return new instance
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _);
                Shape(obj)
            } else {
                Shape(ptr::null_mut())
            }
        }
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Shape").field(&self.dimensions()).finish()
    }
}
