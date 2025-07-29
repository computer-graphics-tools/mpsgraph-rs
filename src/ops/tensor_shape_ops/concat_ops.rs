use crate::{Graph, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSString};

impl Graph {
    /// Concatenates two tensors along a given axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` – First tensor.
    /// * `tensor2` – Second tensor (must be broadcast compatible along all
    ///   other axes and share the same datatype).
    /// * `dimension_index` – Axis along which to concatenate (supports negative
    ///   indexing).
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing the concatenation result.
    pub fn concat(
        &self,
        tensor: &Tensor,
        tensor2: &Tensor,
        dimension_index: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                concatTensor: tensor,
                withTensor: tensor2,
                dimension: dimension_index,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Concatenates a slice of tensors along a given axis.
    ///
    /// All input tensors must be broadcast compatible along the non-concat
    /// dimensions and share the same datatype.
    ///
    /// If `interleave` is `true`, the tensors are interleaved rather than
    /// stacked. Example:
    /// ```rust,no_run
    /// use mpsgraph::{Graph, Tensor};
    /// let graph = Graph::new();
    /// let a: &Tensor = unsafe { core::mem::transmute(0usize) };
    /// let b: &Tensor = unsafe { core::mem::transmute(0usize) };
    /// let _ = graph.concat_tensors(&[a, b], 0, true, None);
    /// // axis = 0, interleave = true
    /// ```
    ///
    /// # Arguments
    ///
    /// * `tensors` – Slice of tensors to concatenate.
    /// * `dimension_index` – Axis along which to concatenate (supports negative
    ///   indexing).
    /// * `interleave` – Whether to interleave the tensors along the concat
    ///   axis.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] containing the concatenation result.
    pub fn concat_tensors(
        &self,
        tensors: &[&Tensor],
        dimension_index: i64,
        interleave: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let tensors_array = NSArray::from_slice(tensors);
        if interleave {
            unsafe {
                msg_send![
                    self,
                    concatTensors: &*tensors_array,
                    dimension: dimension_index,
                    interleave: interleave,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            }
        } else {
            unsafe {
                msg_send![
                    self,
                    concatTensors: &*tensors_array,
                    dimension: dimension_index,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            }
        }
    }
}
