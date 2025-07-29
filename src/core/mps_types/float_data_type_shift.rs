use objc2::Encoding;
use objc2::{Encode, RefEncode};

#[allow(dead_code)]
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum FloatDataTypeShift {
    SignShift = 23,
    ExponentShift = 18,
    MantissaShift = 10,
}

impl From<FloatDataTypeShift> for u32 {
    fn from(shift: FloatDataTypeShift) -> Self {
        shift as u32
    }
}

unsafe impl Encode for FloatDataTypeShift {
    const ENCODING: Encoding = u32::ENCODING;
}

unsafe impl RefEncode for FloatDataTypeShift {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
