use objc2::Encoding;
use objc2::{Encode, RefEncode};

#[allow(dead_code)]
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum FloatDataTypeBit {
    SignBit = 0x0080_0000,
    ExponentBit = 0x007C_0000,
    MantissaBit = 0x0003_FC00,
}

impl From<FloatDataTypeBit> for u32 {
    fn from(bit: FloatDataTypeBit) -> Self {
        bit as u32
    }
}

unsafe impl Encode for FloatDataTypeBit {
    const ENCODING: Encoding = u32::ENCODING;
}

unsafe impl RefEncode for FloatDataTypeBit {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
