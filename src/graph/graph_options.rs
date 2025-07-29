use objc2::{Encode, Encoding, RefEncode};

#[allow(dead_code)]
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum GraphOptions {
    /// No options.
    None = 0,
    /// Synchronize results to CPU on discrete GPUs.
    SynchronizeResults = 1,
    /// Enable verbose logging.
    Verbose = 2,
}

impl Default for GraphOptions {
    fn default() -> Self {
        GraphOptions::SynchronizeResults
    }
}

impl From<GraphOptions> for u64 {
    fn from(opt: GraphOptions) -> Self {
        opt as u64
    }
}

unsafe impl Encode for GraphOptions {
    const ENCODING: Encoding = u64::ENCODING;
}

unsafe impl RefEncode for GraphOptions {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
