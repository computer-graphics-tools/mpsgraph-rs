use objc2::runtime::AnyObject;
use objc2::msg_send;
use metal::DeviceRef;
use metal::foreign_types::ForeignTypeRef;
use std::fmt;
use std::ptr;

/// A wrapper for an MPSGraphDevice object
pub struct MPSGraphDevice(pub(crate) *mut AnyObject);

impl Default for MPSGraphDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl MPSGraphDevice {
    /// Creates a new MPSGraphDevice using the system default Metal device
    pub fn new() -> Self {
        let device = metal::Device::system_default().expect("No Metal device found");
        Self::with_device(&device)
    }
    
    /// Creates a new MPSGraphDevice from a Metal device
    pub fn with_device(device: &DeviceRef) -> Self {
        unsafe {
            let class_name = c"MPSGraphDevice";
            let cls = objc2::runtime::AnyClass::get(class_name).unwrap_or_else(|| panic!("MPSGraphDevice class not found"));
            // Convert the Metal device to a void pointer to avoid RefEncode issues
            let metal_device_ptr = device.as_ptr() as *mut std::ffi::c_void;
            let obj: *mut AnyObject = msg_send![cls, deviceWithMTLDevice: metal_device_ptr];
            let obj = objc2::ffi::objc_retain(obj as *mut _) as *mut AnyObject;
            MPSGraphDevice(obj)
        }
    }
}

impl Drop for MPSGraphDevice {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraphDevice {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
                MPSGraphDevice(obj)
            } else {
                MPSGraphDevice(ptr::null_mut())
            }
        }
    }
}

impl fmt::Debug for MPSGraphDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MPSGraphDevice")
    }
}