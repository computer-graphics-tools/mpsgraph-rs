use objc2_foundation::{NSArray, NSNumber};

/// An array of NSNumbers where dimension lengths provided by the user goes from slowest moving to fastest moving dimension.
/// This is same order as MLMultiArray in coreML and most frameworks in Python.
///
/// ```text
///   A shape @[5, 4, 2] would mean fastest moving 0th dimension is one with size 2,
///   1st dimension is size 4 finally slowest moving 2nd dimension is size 5.
/// ```
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsshape?language=objc)
pub type Shape = NSArray<NSNumber>;
