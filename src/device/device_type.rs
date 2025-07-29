use objc2::Encoding;
use objc2::{Encode, RefEncode};

/// Device type (MPSGraphDeviceType)
#[allow(dead_code)]
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum DeviceType {
    /// Metal device
    Metal = 0,
}

impl From<DeviceType> for u32 {
    fn from(dt: DeviceType) -> Self {
        dt as u32
    }
}

unsafe impl Encode for DeviceType {
    const ENCODING: Encoding = u32::ENCODING;
}

unsafe impl RefEncode for DeviceType {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
