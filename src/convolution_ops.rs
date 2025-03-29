use objc2::msg_send;
use objc2::runtime::AnyObject;
use std::ptr;

/// Convolution padding mode
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MPSGraphPaddingMode {
    /// Valid padding - no padding
    Valid = 0,
    /// Same padding - pad to maintain same size
    Same = 1,
    /// Explicit padding - user-specified padding values
    Explicit = 2,
}

/// Dataflow direction for convolution
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MPSGraphConvolutionDataLayout {
    /// Data is arranged as NHWC (batch, height, width, channels)
    NHWC = 0,
    /// Data is arranged as NCHW (batch, channels, height, width)
    NCHW = 1,
}

/// Weight layout for convolution
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MPSGraphWeightsLayout {
    /// Weights arranged as HWIO (height, width, input channels, output channels)
    HWIO = 0,
    /// Weights arranged as OHWI (output channels, height, width, input channels)
    OHWI = 1,
    /// Weights arranged as IHWO (input channels, height, width, output channels)
    IHWO = 2,
}

/// Descriptor for 2D convolution operations
pub struct MPSGraphConvolution2DOpDescriptor(pub(crate) *mut AnyObject);

/// Descriptor for 3D convolution operations
pub struct MPSGraphConvolution3DOpDescriptor(pub(crate) *mut AnyObject);

impl Drop for MPSGraphConvolution2DOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraphConvolution2DOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
                MPSGraphConvolution2DOpDescriptor(obj)
            } else {
                MPSGraphConvolution2DOpDescriptor(ptr::null_mut())
            }
        }
    }
}

impl Drop for MPSGraphConvolution3DOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for MPSGraphConvolution3DOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
                MPSGraphConvolution3DOpDescriptor(obj)
            } else {
                MPSGraphConvolution3DOpDescriptor(ptr::null_mut())
            }
        }
    }
}

impl MPSGraphConvolution2DOpDescriptor {
    /// Creates a new descriptor with default parameters
    pub fn new() -> Self {
        unsafe {
            let cls = objc2::runtime::AnyClass::get(c"MPSGraphConvolution2DOpDescriptor").unwrap();
            let desc: *mut AnyObject = msg_send![cls, descriptor];
            MPSGraphConvolution2DOpDescriptor(
                objc2::ffi::objc_retain(desc as *mut _) as *mut AnyObject
            )
        }
    }
}

impl MPSGraphConvolution3DOpDescriptor {
    /// Creates a new descriptor with default parameters
    pub fn new() -> Self {
        unsafe {
            let cls = objc2::runtime::AnyClass::get(c"MPSGraphConvolution3DOpDescriptor").unwrap();
            let desc: *mut AnyObject = msg_send![cls, descriptor];
            MPSGraphConvolution3DOpDescriptor(
                objc2::ffi::objc_retain(desc as *mut _) as *mut AnyObject
            )
        }
    }
}
