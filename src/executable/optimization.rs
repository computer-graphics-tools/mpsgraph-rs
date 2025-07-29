/// Represents the optimization level for graph compilation
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u64)]
pub enum Optimization {
    /// Graph performs core optimizations only
    Level0 = 0,
    /// Graph performs additional optimizations
    Level1 = 1,
}
