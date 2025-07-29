use super::origin::Origin;
use super::size::Size;
use objc2::Encoding;
use objc2::{Encode, RefEncode};

/// A region of an image
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsregion?language=objc)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    /// The top left corner of the region.  Units: pixels
    pub origin: Origin,
    /// The size of the region. Units: pixels
    pub size: Size,
}

unsafe impl Encode for Region {
    const ENCODING: Encoding =
        Encoding::Struct("MPSRegion", &[<Origin>::ENCODING, <Size>::ENCODING]);
}

unsafe impl RefEncode for Region {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
