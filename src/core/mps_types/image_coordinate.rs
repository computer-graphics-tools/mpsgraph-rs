use objc2::Encoding;
use objc2::{Encode, RefEncode};
use objc2_foundation::NSUInteger;

/// A unsigned coordinate with x, y and channel components
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsimagecoordinate?language=objc)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageCoordinate {
    /// The horizontal component of the coordinate. Units: pixels
    pub x: NSUInteger,
    /// The vertical component of the coordinate. Units: pixels
    pub y: NSUInteger,
    /// The index of the channel or feature channel within the pixel
    pub channel: NSUInteger,
}

unsafe impl Encode for ImageCoordinate {
    const ENCODING: Encoding = Encoding::Struct(
        "MPSImageCoordinate",
        &[
            <NSUInteger>::ENCODING,
            <NSUInteger>::ENCODING,
            <NSUInteger>::ENCODING,
        ],
    );
}

unsafe impl RefEncode for ImageCoordinate {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
