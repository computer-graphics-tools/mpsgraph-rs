use objc2::{Encode, Encoding, RefEncode};

#[allow(dead_code)]
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum DataType {
    // Base flags
    Invalid = 0,
    FloatBit = 0x1000_0000,
    ComplexBit = 0x0100_0000,
    SignedBit = 0x2000_0000,
    AlternateEncodingBit = 0x8000_0000,
    NormalizedBit = 0x4000_0000,

    // Floating-point types
    Float32 = DataType::FloatBit as u32 | 32,
    Float16 = DataType::FloatBit as u32 | 16,

    // Complex types
    ComplexFloat32 = DataType::FloatBit as u32 | DataType::ComplexBit as u32 | 64,
    ComplexFloat16 = DataType::FloatBit as u32 | DataType::ComplexBit as u32 | 32,

    // Signed integers
    Int2 = DataType::SignedBit as u32 | 2,
    Int4 = DataType::SignedBit as u32 | 4,
    Int8 = DataType::SignedBit as u32 | 8,
    Int16 = DataType::SignedBit as u32 | 16,
    Int32 = DataType::SignedBit as u32 | 32,
    Int64 = DataType::SignedBit as u32 | 64,

    // Unsigned integers
    UInt2 = 2,
    UInt4 = 4,
    UInt8 = 8,
    UInt16 = 16,
    UInt32 = 32,
    UInt64 = 64,

    // Alternate encodings / normalized types
    Bool = DataType::AlternateEncodingBit as u32 | 8,
    BFloat16 = DataType::AlternateEncodingBit as u32 | DataType::Float16 as u32,
    Unorm1 = DataType::NormalizedBit as u32 | 1,
    Unorm8 = DataType::NormalizedBit as u32 | 8,
}

// From <-> Into conversions with `u32`
impl From<DataType> for u32 {
    fn from(dt: DataType) -> Self {
        dt as u32
    }
}

// Tell objc2 that `DataType` is ABI-compatible with `u32`.
unsafe impl Encode for DataType {
    const ENCODING: Encoding = u32::ENCODING;
}
unsafe impl RefEncode for DataType {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
