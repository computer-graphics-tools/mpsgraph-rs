use objc2::Encoding;
use objc2::{Encode, RefEncode};
use objc2_foundation::NSUInteger;

#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum TensorNamedDataLayout {
    /// LayoutNCHW
    NCHW = 0,
    /// LayoutNHWC
    NHWC = 1,
    /// LayoutOIHW
    OIHW = 2,
    /// LayoutHWIO
    HWIO = 3,
    /// LayoutCHW
    CHW = 4,
    /// LayoutHWC
    HWC = 5,
    /// LayoutHW
    HW = 6,
    /// LayoutNCDHW
    NCDHW = 7,
    /// LayoutNDHWC
    NDHWC = 8,
    /// LayoutOIDHW
    OIDHW = 9,
    /// LayoutDHWIO
    DHWIO = 10,
}

impl From<TensorNamedDataLayout> for NSUInteger {
    fn from(layout: TensorNamedDataLayout) -> Self {
        layout as NSUInteger
    }
}

unsafe impl Encode for TensorNamedDataLayout {
    const ENCODING: Encoding = NSUInteger::ENCODING;
}

unsafe impl RefEncode for TensorNamedDataLayout {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
