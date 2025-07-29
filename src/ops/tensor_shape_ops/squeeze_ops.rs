use crate::{AxesOrTensor, Graph, Tensor, ns_number_array_from_slice};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Removes all singleton dimensions from `tensor`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] with all size-1 dimensions removed.
    pub fn squeeze(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                squeezeTensor: tensor,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Removes the singleton dimension at `axis`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `axis` – Axis to remove (size must be 1).
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] with the specified axis removed.
    pub fn squeeze_axis(&self, tensor: &Tensor, axis: i64, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![self, squeezeTensor: tensor, axis: axis, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Removes singleton dimensions at multiple `axes`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Input tensor.
    /// * `axes` – Axes to remove (slice or tensor).
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] with the specified axes removed.
    pub fn squeeze_axes<'a>(
        &self,
        tensor: &Tensor,
        axes: AxesOrTensor<'a>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axes {
            AxesOrTensor::Axes(axes) => unsafe {
                msg_send![
                    self,
                    squeezeTensor: tensor,
                    axes: &*ns_number_array_from_slice(axes),
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            AxesOrTensor::Tensor(axes) => unsafe {
                msg_send![
                    self,
                    squeezeTensor: tensor,
                    axesTensor: axes,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
