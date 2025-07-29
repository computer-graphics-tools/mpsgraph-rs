use objc2::Encoding;
use objc2::{Encode, RefEncode};
use objc2_foundation::NSUInteger;

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AliasingStrategy {
    DontCare = 0,
    ShallAlias = 1 << 0,
    ShallNotAlias = 1 << 1,
    AliasingReserved = (Self::ShallAlias as usize | Self::ShallNotAlias as usize),
    PreferTemporaryMemory = 1 << 2,
    PreferNonTemporaryMemory = 1 << 3,
}

impl Default for AliasingStrategy {
    fn default() -> Self {
        Self::DontCare
    }
}

unsafe impl Encode for AliasingStrategy {
    const ENCODING: Encoding = NSUInteger::ENCODING;
}

unsafe impl RefEncode for AliasingStrategy {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
