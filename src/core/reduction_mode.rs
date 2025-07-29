use objc2::Encoding;
use objc2::{Encode, RefEncode};

#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum ReductionMode {
    /// Min
    Min = 0,
    /// Max
    Max = 1,
    /// Sum
    Sum = 2,
    /// Product
    Product = 3,
    /// Argument Min
    ArgumentMin = 4,
    /// Argument Max
    ArgumentMax = 5,
}

unsafe impl Encode for ReductionMode {
    const ENCODING: Encoding = u64::ENCODING;
}

unsafe impl RefEncode for ReductionMode {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
