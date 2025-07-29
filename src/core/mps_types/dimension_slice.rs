use objc2::Encoding;
use objc2::{Encode, RefEncode};
use objc2_foundation::NSUInteger;

/// Describes a sub-region of an array dimension
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsdimensionslice?language=objc)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DimensionSlice {
    /// the position of the first element in the slice
    pub start: NSUInteger,
    /// the number of elements in the slice.
    pub length: NSUInteger,
}

unsafe impl Encode for DimensionSlice {
    const ENCODING: Encoding = Encoding::Struct(
        "MPSDimensionSlice",
        &[<NSUInteger>::ENCODING, <NSUInteger>::ENCODING],
    );
}

unsafe impl RefEncode for DimensionSlice {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
