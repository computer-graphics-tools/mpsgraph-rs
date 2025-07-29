use crate::{Graph, ScalarOrTensor, Shape, Tensor, ns_number_array_from_slice};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Generates a coordinate tensor along a single axis.
    ///
    /// The resulting tensor has the given `shape` and contains the index value
    /// of `axis` at every position: `result[i₀ … iₙ] = i_axis`.
    ///
    /// Example:
    /// ```rust,no_run
    /// # use mpsgraph::{Graph, ScalarOrTensor};
    /// # let graph = Graph::new();
    /// let _ = graph.coordinate_along_axis(ScalarOrTensor::Scalar(0), &[5], None);  // ➜ [0, 1, 2, 3, 4]
    /// let _ = graph.coordinate_along_axis(ScalarOrTensor::Scalar(0), &[3, 2], None); // ➜ [[0, 0], [1, 1], [2, 2]]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `axis` – Axis whose coordinate values to generate (negative indices
    ///   wrap around).
    /// * `shape` – Desired output shape.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] filled with coordinate values along `axis`.
    pub fn coordinate_along_axis<'a>(
        &self,
        axis: ScalarOrTensor<'a, i64>,
        shape: &[isize],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let shape = ns_number_array_from_slice(shape);
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                    self,
                    coordinateAlongAxis: axis,
                    withShape: &*shape,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    coordinateAlongAxisTensor: axis,
                    withShape: &*shape,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Same as [`coordinate_along_axis`], but accepts the shape as a tensor.
    ///
    /// # Arguments
    ///
    /// * `axis` – Axis whose coordinate values to generate (scalar or tensor).
    /// * `shape_tensor` – Rank-1 tensor (`i32`/`i64`) defining the output shape.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] filled with coordinate values along `axis`.
    pub fn coordinate_along_axis_with_shape_tensor<'a>(
        &self,
        axis: ScalarOrTensor<'a, i64>,
        shape_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                    self,
                    coordinateAlongAxis: axis,
                    withShapeTensor: shape_tensor,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    coordinateAlongAxisTensor: axis,
                    withShapeTensor: shape_tensor,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
