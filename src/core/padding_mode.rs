use objc2::{Encode, Encoding, RefEncode};

/// Tensor padding mode
#[allow(dead_code)]
#[repr(i64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum PaddingMode {
    /// Constant
    Constant = 0,
    /// Reflect
    Reflect = 1,
    /// Symmetric
    Symmetric = 2,
    /// ClampToEdge (PyTorch ReplicationPad)
    ClampToEdge = 3,
    /// Zero
    Zero = 4,
    /// Periodic `x[-2] -> x[L-3]`
    Periodic = 5,
    /// Anti Periodic `x[-2] -> -x[L-3]`
    AntiPeriodic = 6,
}

unsafe impl Encode for PaddingMode {
    const ENCODING: Encoding = i64::ENCODING;
}

unsafe impl RefEncode for PaddingMode {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
