use ::core::ffi::c_double;
use objc2::Encoding;
use objc2::{Encode, RefEncode};

/// A position in an image
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsorigin?language=objc)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Origin {
    /// The x coordinate of the position
    pub x: c_double,
    /// The y coordinate of the position
    pub y: c_double,
    /// The z coordinate of the position
    pub z: c_double,
}

unsafe impl Encode for Origin {
    const ENCODING: Encoding = Encoding::Struct(
        "MPSOrigin",
        &[
            <c_double>::ENCODING,
            <c_double>::ENCODING,
            <c_double>::ENCODING,
        ],
    );
}

unsafe impl RefEncode for Origin {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
