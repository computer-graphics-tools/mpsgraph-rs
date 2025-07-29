use ::core::ffi::c_double;
use objc2::Encoding;
use objc2::{Encode, RefEncode};

/// A size of a region in an image
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpssize?language=objc)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Size {
    /// The width of the region
    pub width: c_double,
    /// The height of the region
    pub height: c_double,
    /// The depth of the region
    pub depth: c_double,
}

unsafe impl Encode for Size {
    const ENCODING: Encoding = Encoding::Struct(
        "MPSSize",
        &[
            <c_double>::ENCODING,
            <c_double>::ENCODING,
            <c_double>::ENCODING,
        ],
    );
}

unsafe impl RefEncode for Size {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
