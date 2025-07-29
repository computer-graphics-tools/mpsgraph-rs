use objc2::Encoding;
use objc2::{Encode, RefEncode};
use objc2_foundation::NSInteger;

/// A signed coordinate with x, y and z components
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsoffset?language=objc)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MPSOffset {
    /// The horizontal component of the offset. Units: pixels
    pub x: NSInteger,
    /// The vertical component of the offset. Units: pixels
    pub y: NSInteger,
    /// The depth component of the offset. Units: pixels
    pub z: NSInteger,
}

unsafe impl Encode for MPSOffset {
    const ENCODING: Encoding = Encoding::Struct(
        "?",
        &[
            <NSInteger>::ENCODING,
            <NSInteger>::ENCODING,
            <NSInteger>::ENCODING,
        ],
    );
}

unsafe impl RefEncode for MPSOffset {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
