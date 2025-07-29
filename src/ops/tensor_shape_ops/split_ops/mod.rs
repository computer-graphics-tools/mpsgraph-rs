//! Tensor *split* helper operations.
//!
//! Provides 1) size-based split (`split_tensor`) and 2) equal-part split
//! (`split_tensor_num_splits`). Both return boxed slices of result tensors.
//!

mod sizes_or_tensor;

pub use sizes_or_tensor::SizesOrTensor;

use crate::{Graph, Tensor, ns_number_array_from_slice};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSString};

impl Graph {
    /// Splits `tensor` into variable-sized chunks.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor to split.
    /// * `split_sizes` – Slice or tensor specifying the size of each chunk.
    ///   The sum must equal `tensor.shape[axis]`.
    /// * `axis` – Dimension along which to split.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A boxed slice of [`Tensor`] objects, one for each split.
    pub fn split<'a>(
        &self,
        tensor: &Tensor,
        split_sizes: SizesOrTensor<'a>,
        axis: i64,
        name: Option<&str>,
    ) -> Box<[Retained<Tensor>]> {
        match split_sizes {
            SizesOrTensor::Sizes(split_sizes) => {
                let result: Retained<NSArray<Tensor>> = unsafe {
                    msg_send![
                        self,
                        splitTensor: tensor,
                        splitSizes: &*ns_number_array_from_slice(split_sizes),
                        axis: axis,
                        name: name.map(NSString::from_str).as_deref(),
                    ]
                };
                result.to_vec().into_boxed_slice()
            }
            SizesOrTensor::Tensor(split_sizes_tensor) => unsafe {
                let result: Retained<NSArray<Tensor>> = unsafe {
                    msg_send![
                        self,
                        splitTensor: tensor,
                        splitSizesTensor: split_sizes_tensor,
                        axis: axis,
                        name: name.map(NSString::from_str).as_deref(),
                    ]
                };
                result.to_vec().into_boxed_slice()
            },
        }
    }

    /// Splits `tensor` into `num_splits` equal-sized chunks.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor to split.
    /// * `num_splits` – Number of equal parts to create (must divide
    ///   `tensor.shape[axis]`).
    /// * `axis` – Dimension along which to split.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A boxed slice of [`Tensor`] objects, one for each split.
    pub fn split_num_splits(
        &self,
        tensor: &Tensor,
        num_splits: u64,
        axis: i64,
        name: Option<&str>,
    ) -> Box<[Retained<Tensor>]> {
        let result: Retained<NSArray<Tensor>> = unsafe {
            msg_send![
                self,
                splitTensor: tensor,
                numSplits: num_splits,
                axis: axis,
                name: name.map(NSString::from_str).as_deref(),
            ]
        };
        result.to_vec().into_boxed_slice()
    }
}
