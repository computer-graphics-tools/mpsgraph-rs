use crate::core::MPSDataType;
use crate::operation::MPSGraphOperation;
use crate::shape::MPSShape;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use std::convert::AsRef;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ptr;

/// A wrapper for MPSGraphTensor objects
pub struct MPSGraphTensor(pub(crate) *mut AnyObject);

// Implement Send + Sync for the wrapper type
unsafe impl Send for MPSGraphTensor {}
unsafe impl Sync for MPSGraphTensor {}

impl MPSGraphTensor {
    /// Returns the data type of this tensor
    pub fn data_type(&self) -> MPSDataType {
        unsafe {
            let data_type_val: u32 = msg_send![self.0, dataType];
            std::mem::transmute(data_type_val)
        }
    }

    /// Returns the shape of this tensor
    pub fn shape(&self) -> MPSShape {
        unsafe {
            let shape: *mut AnyObject = msg_send![self.0, shape];
            // Check if shape is null (unranked tensor)
            if shape.is_null() {
                return MPSShape::from_slice(&[]);
            }
            let shape = objc2::ffi::objc_retain(shape as *mut _);
            MPSShape(shape)
        }
    }

    /// Returns the operation that produced this tensor
    pub fn operation(&self) -> MPSGraphOperation {
        unsafe {
            let operation: *mut AnyObject = msg_send![self.0, operation];
            let operation = objc2::ffi::objc_retain(operation as *mut _);
            MPSGraphOperation(operation)
        }
    }

    /// Returns the dimensions of the tensor
    pub fn dimensions(&self) -> Vec<usize> {
        self.shape().dimensions().to_vec()
    }

    /// Returns the rank (number of dimensions) of this tensor
    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    /// Returns the total number of elements in this tensor
    pub fn element_count(&self) -> usize {
        self.shape().element_count()
    }

    /// Returns the name of this tensor
    pub fn name(&self) -> String {
        unsafe {
            let name: *mut AnyObject = msg_send![self.0, name];

            // Handle case where name is nil
            if name.is_null() {
                return String::from("<unnamed>");
            }

            let utf8: *const i8 = msg_send![name, UTF8String];
            std::ffi::CStr::from_ptr(utf8).to_string_lossy().to_string()
        }
    }
}

impl Drop for MPSGraphTensor {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // TEMPORARY: For debugging, don't release objects to avoid the crash
            // Save pointer value for debugging
            let ptr = self.0;

            // Print debug info (without objc_release)
            println!(
                "MPSGraphTensor dropping with ptr: {:p} - SKIPPING RELEASE",
                self.0
            );

            // Set the pointer to null to prevent further access
            self.0 = std::ptr::null_mut();

            // MEMORY LEAK: We're deliberately not releasing the object to avoid the crash
            // objc2::ffi::objc_release(ptr as *mut _);
            println!(
                "MPSGraphTensor NOT released: {:p} (skipped to avoid crash)",
                ptr
            );
        }
    }
}

impl Clone for MPSGraphTensor {
    fn clone(&self) -> Self {
        if !self.0.is_null() {
            // TEMPORARY: For debugging, no need to retain either since we're not releasing
            println!("MPSGraphTensor cloning ptr: {:p}", self.0);

            // Just copy the pointer without retaining (since we're not releasing)
            // let obj = objc2::ffi::objc_retain(self.0 as *mut _);
            let obj = self.0; // Just copy the pointer without retaining

            println!(
                "MPSGraphTensor cloned with ptr: {:p} (no retain for debug)",
                obj
            );

            MPSGraphTensor(obj)
        } else {
            println!("MPSGraphTensor cloning NULL pointer");
            MPSGraphTensor(ptr::null_mut())
        }
    }
}

impl PartialEq for MPSGraphTensor {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for MPSGraphTensor {}

impl Hash for MPSGraphTensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.0 as usize).hash(state);
    }
}

impl fmt::Debug for MPSGraphTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MPSGraphTensor")
            .field("name", &self.name())
            .field("data_type", &self.data_type())
            .field("dimensions", &self.dimensions())
            .finish()
    }
}

impl AsRef<MPSGraphTensor> for MPSGraphTensor {
    fn as_ref(&self) -> &MPSGraphTensor {
        self
    }
}
