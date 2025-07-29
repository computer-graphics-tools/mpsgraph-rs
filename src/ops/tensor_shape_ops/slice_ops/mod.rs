//! Slice-related tensor shape operations.
//!
//! This module re-exports the helpers for plain slicing, strided slicing,
//! slice updates, and their gradients. Importing `mpsgraph_rs::*` brings these
//! items into scope by default.
//!
//! The individual sub-modules mirror the Objective-C APIs:
//!
//! * `slice_ops` – "Slice" / `sliceTensor...`
//! * `strided_slice_ops` – "StridedSlice" / `stridedSliceTensor...`
//! * `slice_update_ops` – "SliceUpdate" / `sliceTensor:update...`
//! * `slice_gradient_ops` – gradient helpers for the above
//!
//! For convenience all public items are re-exported at this level.
//!
mod scalars_or_tensors;
mod slice_gradient_ops;
mod slice_ops;
mod slice_update_ops;
mod strided_slice_ops;

pub use scalars_or_tensors::*;
pub use slice_gradient_ops::*;
pub use slice_ops::*;
pub use slice_update_ops::*;
pub use strided_slice_ops::*;
