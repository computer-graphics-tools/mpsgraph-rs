/// Command buffer status
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CommandBufferStatus {
    /// The command buffer has not been enqueued yet
    NotEnqueued = 0,
    /// The command buffer is enqueued, but not committed
    Enqueued = 1,
    /// Committed to the command queue, but not yet scheduled for execution
    Committed = 2,
    /// All dependencies have been resolved and scheduled for execution
    Scheduled = 3,
    /// The command buffer has finished executing successfully
    Completed = 4,
    /// Execution was aborted due to an error
    Error = 5,
}
