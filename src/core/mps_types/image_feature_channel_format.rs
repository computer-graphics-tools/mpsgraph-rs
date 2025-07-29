use objc2::Encoding;
use objc2::{Encode, RefEncode};
use objc2_foundation::NSUInteger;

/// Apple’s `MPSImageFeatureChannelFormat`
/// <https://developer.apple.com/documentation/metalperformanceshaders/mpsimagefeaturechannelformat>
#[allow(dead_code)]
#[repr(usize)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum ImageFeatureChannelFormat {
    /// Invalid / any format (let MPS choose)
    None = 0,
    /// uint8 	o [0,255] encoding [0,1]
    Unorm8 = 1,
    /// uint16 to [0,65535] encoding [0,1]
    Unorm16 = 2,
    /// 16-bit IEEE-754 float (half precision)
    Float16 = 3,
    /// 32-bit IEEE-754 float (single precision)
    Float32 = 4,
    /// Reserved for future use
    _Reserved0 = 5,
    /// Count of defined formats (reserved too)
    Count = 6,
}

impl From<ImageFeatureChannelFormat> for NSUInteger {
    fn from(format: ImageFeatureChannelFormat) -> Self {
        format as NSUInteger
    }
}

unsafe impl Encode for ImageFeatureChannelFormat {
    const ENCODING: Encoding = NSUInteger::ENCODING;
}

unsafe impl RefEncode for ImageFeatureChannelFormat {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
