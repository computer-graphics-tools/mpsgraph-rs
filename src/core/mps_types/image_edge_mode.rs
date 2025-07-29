use objc2::Encoding;
use objc2::{Encode, RefEncode};
use objc2_foundation::NSUInteger;

/// Apple’s `MPSImageEdgeMode`
/// <https://developer.apple.com/documentation/metalperformanceshaders/mpsimageedgemode>
#[allow(dead_code)]
#[repr(usize)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum MPSImageEdgeMode {
    /// Out-of-bound pixels are (0,0,0,1) for images without alpha,
    /// (0,0,0,0) otherwise.
    Zero = 0,
    /// Out-of-bound pixels are clamped to the nearest edge pixel.
    Clamp = 1,
    /// Pixels are mirrored w.r.t. the nearest edge pixel center.
    Mirror = 2,
    /// Pixels are mirrored w.r.t. the nearest edge pixel border (edge repeated).
    MirrorWithEdge = 3,
    /// Pixels are filled with a constant value defined by the filter.
    Constant = 4,
}

impl From<MPSImageEdgeMode> for NSUInteger {
    fn from(mode: MPSImageEdgeMode) -> Self {
        mode as NSUInteger
    }
}

unsafe impl Encode for MPSImageEdgeMode {
    const ENCODING: Encoding = NSUInteger::ENCODING;
}

unsafe impl RefEncode for MPSImageEdgeMode {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
