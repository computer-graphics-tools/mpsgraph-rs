use crate::{Graph, ScalarOrTensor, Tensor};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

/// CumulativeOps.
impl Graph {
    /// Computes the cumulative sum of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input [`Tensor`].
    /// * `axis` - The tensor dimension where you compute the cumulative operation.
    /// * `exclusive` - If `true`, perform the exclusive cumulative operation, and the first element will be equal to zero.
    /// * `reverse` - If `true`, reverse the direction of the cumulative operation along the specified axis.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn cumulative_sum_with_exclusive_reverse<'a>(
        &self,
        tensor: &Tensor,
        axis: ScalarOrTensor<'a, i64>,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                    self,
                    cumulativeSumWithTensor: tensor,
                    axis: axis,
                    exclusive: exclusive,
                    reverse: reverse,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    cumulativeSumWithTensor: tensor,
                    axisTensor: axis,
                    exclusive: exclusive,
                    reverse: reverse,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Computes the cumulative sum of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input [`Tensor`].
    /// * `axis` - The tensor dimension where you compute the cumulative operation.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn cumulative_sum<'a>(
        &self,
        tensor: &Tensor,
        axis: ScalarOrTensor<'a, i64>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                self,
                cumulativeSumWithTensor: tensor,
                axis: axis,
                name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    cumulativeSumWithTensor: tensor,
                    axisTensor: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Computes the cumulative product of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input [`Tensor`].
    /// * `axis` - The tensor dimension where you compute the cumulative operation.
    /// * `exclusive` - If `true`, perform the exclusive cumulative operation, and the first element will be equal to one.
    /// * `reverse` - If `true`, reverse the direction of the cumulative operation along the specified axis.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn cumulative_product_with_exclusive_reverse<'a>(
        &self,
        tensor: &Tensor,
        axis: ScalarOrTensor<'a, i64>,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                    self,
                    cumulativeProductWithTensor: tensor,
                    axis: axis,
                    exclusive: exclusive,
                    reverse: reverse,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    cumulativeProductWithTensor: tensor,
                    axisTensor: axis,
                    exclusive: exclusive,
                    reverse: reverse,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Computes the cumulative product of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input [`Tensor`].
    /// * `axis` - The tensor dimension where you compute the cumulative operation.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn cumulative_product<'a>(
        &self,
        tensor: &Tensor,
        axis: ScalarOrTensor<'a, i64>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                    self,
                    cumulativeProductWithTensor: tensor,
                    axis: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    cumulativeProductWithTensor: tensor,
                    axisTensor: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Computes the cumulative minimum of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input [`Tensor`].
    /// * `axis` - The tensor dimension where you compute the cumulative operation.
    /// * `exclusive` - If `true`, perform the exclusive cumulative operation, and the first element will be equal to the largest value of the tensor data type.
    /// * `reverse` - If `true`, reverse the direction of the cumulative operation along the specified axis.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn cumulative_minimum_with_exclusive_reverse<'a>(
        &self,
        tensor: &Tensor,
        axis: ScalarOrTensor<'a, i64>,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                    self,
                    cumulativeMinimumWithTensor: tensor,
                    axis: axis,
                    exclusive: exclusive,
                    reverse: reverse,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    cumulativeMinimumWithTensor: tensor,
                    axisTensor: axis,
                    exclusive: exclusive,
                    reverse: reverse,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Computes the cumulative minimum of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input [`Tensor`].
    /// * `axis` - The tensor dimension where you compute the cumulative operation.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn cumulative_minimum<'a>(
        &self,
        tensor: &Tensor,
        axis: ScalarOrTensor<'a, i64>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                    self,
                    cumulativeMinimumWithTensor: tensor,
                    axis: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    cumulativeMinimumWithTensor: tensor,
                    axisTensor: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Computes the cumulative maximum of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input [`Tensor`].
    /// * `axis` - The tensor dimension where you compute the cumulative operation.
    /// * `exclusive` - If `true`, perform the exclusive cumulative operation, and the first element will be equal to the lowest value of the tensor data type.
    /// * `reverse` - If `true`, reverse the direction of the cumulative operation along the specified axis.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn cumulative_maximum_with_exclusive_reverse<'a>(
        &self,
        tensor: &Tensor,
        axis: ScalarOrTensor<'a, i64>,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                    self,
                    cumulativeMaximumWithTensor: tensor,
                    axis: axis,
                    exclusive: exclusive,
                    reverse: reverse,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    cumulativeMaximumWithTensor: tensor,
                    axisTensor: axis,
                    exclusive: exclusive,
                    reverse: reverse,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Computes the cumulative maximum of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input [`Tensor`].
    /// * `axis` - The tensor dimension where you compute the cumulative operation.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn cumulative_maximum<'a>(
        &self,
        tensor: &Tensor,
        axis: ScalarOrTensor<'a, i64>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                    self,
                    cumulativeMaximumWithTensor: tensor,
                    axis: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    cumulativeMaximumWithTensor: tensor,
                    axisTensor: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
