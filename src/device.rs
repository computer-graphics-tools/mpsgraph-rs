use objc::runtime::Object;
use metal::DeviceRef;
use std::fmt;

/// A wrapper for an MPSGraphDevice object
pub struct MPSGraphDevice(pub(crate) *mut Object);

impl MPSGraphDevice {
    /// Creates a new MPSGraphDevice using the system default Metal device
    pub fn new() -> Self {
        let device = metal::Device::system_default().expect("No Metal device found");
        Self::with_device(&device)
    }
    
    /// Creates a new MPSGraphDevice from a Metal device
    pub fn with_device(device: &DeviceRef) -> Self {
        unsafe {
            let cls = objc::runtime::Class::get("MPSGraphDevice").unwrap();
            let metal_device = device as *const _;
            let obj: *mut Object = msg_send![cls, deviceWithMTLDevice:metal_device];
            let obj: *mut Object = msg_send![obj, retain];
            MPSGraphDevice(obj)
        }
    }
}

impl Drop for MPSGraphDevice {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

impl Clone for MPSGraphDevice {
    fn clone(&self) -> Self {
        unsafe {
            let obj: *mut Object = msg_send![self.0, retain];
            MPSGraphDevice(obj)
        }
    }
}

impl fmt::Debug for MPSGraphDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MPSGraphDevice")
    }
}