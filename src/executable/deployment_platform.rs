use objc2::{Encode, Encoding, RefEncode};

/// Represents the deployment platform for a graph
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphdeploymentplatform?language=objc)
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DeploymentPlatform {
    /// macOS platform
    MacOS = 0,
    /// iOS platform
    IOS = 1,
    /// tvOS platform
    TVOS = 2,
    /// visionOS platform
    VisionOS = 3,
}

unsafe impl Encode for DeploymentPlatform {
    const ENCODING: Encoding = u64::ENCODING;
}

unsafe impl RefEncode for DeploymentPlatform {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
