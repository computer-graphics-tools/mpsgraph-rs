use ::core::ffi::c_double;
use objc2::Encoding;
use objc2::{Encode, RefEncode};

/// Transform matrix for explict control over resampling in MPSImageScale.
///
/// The MPSScaleTransform is equivalent to:
///
/// ```text
///           (CGAffineTransform) {
///                .a = scaleX,        .b = 0,
///                .c = 0,             .d = scaleY,
///                .tx = translateX,   .ty = translateY
///            }
/// ```
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsscaletransform?language=objc)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleTransform {
    /// horizontal scaling factor
    pub scale_x: c_double,
    /// vertical scaling factor
    pub scale_y: c_double,
    /// horizontal translation
    pub translate_x: c_double,
    /// vertical translation
    pub translate_y: c_double,
}

unsafe impl Encode for ScaleTransform {
    const ENCODING: Encoding = Encoding::Struct(
        "MPSScaleTransform",
        &[
            <c_double>::ENCODING,
            <c_double>::ENCODING,
            <c_double>::ENCODING,
            <c_double>::ENCODING,
        ],
    );
}

unsafe impl RefEncode for ScaleTransform {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
