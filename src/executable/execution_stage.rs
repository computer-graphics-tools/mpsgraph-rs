/// Represents the stages of execution for a graph
#[repr(u64)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ExecutionStage {
    /// Execution is completed
    Completed = 0,
}
