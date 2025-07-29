#[cfg(not(any(target_os = "macos", target_os = "ios")))]
compile_error!("MetalPerformanceShadersGraph is only available on macOS and iOS");

mod command_buffer;
mod conversion;
mod core;
mod device;
mod executable;
mod graph;
mod operation;
mod ops;
mod scalar_or_tensor;
mod shape_or_tensor;
mod tensor;
mod tensor_data;
mod utils;

pub use command_buffer::*;
pub use conversion::*;
pub use core::*;
pub use device::*;
pub use executable::*;
pub use graph::*;
pub use operation::*;
pub use ops::*;
pub use scalar_or_tensor::*;
pub use shape_or_tensor::*;
pub use tensor::*;
pub use tensor_data::*;
pub use utils::*;
