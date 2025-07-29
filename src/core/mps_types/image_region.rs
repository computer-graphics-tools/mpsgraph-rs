use objc2::Encoding;
use objc2::{Encode, RefEncode};

use super::ImageCoordinate;

/// A rectangular subregion of a MPSImage
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsimageregion?language=objc)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageRegion {
    /// The position of the top left corner of the subregion
    pub offset: ImageCoordinate,
    /// The size {pixels, pixels, channels} of the subregion
    pub size: ImageCoordinate,
}

unsafe impl Encode for ImageRegion {
    const ENCODING: Encoding = Encoding::Struct(
        "MPSImageRegion",
        &[<ImageCoordinate>::ENCODING, <ImageCoordinate>::ENCODING],
    );
}

unsafe impl RefEncode for ImageRegion {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
