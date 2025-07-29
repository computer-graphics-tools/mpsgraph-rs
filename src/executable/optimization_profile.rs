/// Represents the optimization profile for graph compilation
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u64)]
pub enum OptimizationProfile {
    /// Profile optimized for performance
    Performance = 0,
    /// Profile optimized for power efficiency
    PowerEfficiency = 1,
}
