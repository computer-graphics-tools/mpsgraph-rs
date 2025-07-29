mod graph_variable_op;

pub use graph_variable_op::GraphVariableOp;

use crate::{DataType, Graph, Operation, Shape, Tensor, ns_number_array_from_slice};
use objc2::{
    extern_methods, msg_send,
    rc::{Retained, autoreleasepool},
};
use objc2_foundation::{NSData, NSString};
use std::{mem::size_of_val, slice::from_raw_parts};

/// MemoryOps.
impl Graph {
    extern_methods!();
}

impl Graph {
    /// Creates a constant operation from raw bytes.
    ///
    /// # Arguments
    ///
    /// * `data` – Raw tensor bytes (`len` must equal `sizeof(data_type) * number_of_elements`).
    /// * `shape` – Statically shaped output `[usize]`.
    /// * `data_type` – [`DataType`] of the constant tensor.
    ///
    /// # Returns
    ///
    /// A new [`Tensor`] containing the constant values.
    pub fn constant_with_ns_data(
        &self,
        data: &NSData,
        shape: &[usize],
        data_type: DataType,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                constantWithData: data,
                shape: &*ns_number_array_from_slice(shape),
                dataType: data_type
            ]
        }
    }

    /// Creates a complex constant operation that fills the tensor with a single
    /// complex scalar value.
    ///
    /// # Arguments
    ///
    /// * `real_part` – Real component of the scalar.
    /// * `imaginary_part` – Imaginary component of the scalar.
    /// * `shape` – Statically shaped output `[usize]`.
    /// * `data_type` – [`DataType`] of the constant tensor.
    ///
    /// # Returns
    ///
    /// A new [`Tensor`] containing the constant complex values.
    pub fn constant_with_real_imaginary_shape(
        &self,
        real_part: f64,
        imaginary_part: f64,
        shape: &[usize],
        data_type: DataType,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                constantWithRealPart: real_part,
                imaginaryPart: imaginary_part,
                shape: &*ns_number_array_from_slice(shape),
                dataType: data_type
            ]
        }
    }

    /// Creates a placeholder tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` – Optional shape of the tensor (`None` produces an unranked tensor).
    /// * `data_type` – [`DataType`] of the placeholder.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A new placeholder [`Tensor`].
    pub fn placeholder(
        &self,
        shape: Option<&[isize]>,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let shape = shape.map(ns_number_array_from_slice);
        autoreleasepool(|_| unsafe {
            msg_send![
                self,
                placeholderWithShape: shape.as_deref(),
                dataType: data_type,
                name: name.map(NSString::from_str).as_deref()
            ]
        })
    }

    /// Creates a constant operation from a slice of values.
    ///
    /// # Arguments
    ///
    /// * `data` – Slice containing tensor elements.
    /// * `shape` – Statically shaped output `[usize]`.
    /// * `data_type` – [`DataType`] of the constant tensor.
    ///
    /// # Returns
    ///
    /// A new [`Tensor`] containing the constant values.
    pub fn constant_with_data<T: Copy>(
        &self,
        data: &[T],
        shape: &[usize],
        data_type: DataType,
    ) -> Retained<Tensor> {
        autoreleasepool(|_| {
            let ns_data = unsafe {
                NSData::with_bytes(from_raw_parts(
                    data.as_ptr() as *const u8,
                    size_of_val(data),
                ))
            };
            unsafe {
                msg_send![
                self,
                constantWithData: &*ns_data,
                shape: &*ns_number_array_from_slice(shape),
                dataType: data_type
                ]
            }
        })
    }

    /// Creates a scalar-filled constant operation.
    ///
    /// # Arguments
    ///
    /// * `scalar` – Scalar value used to fill the tensor.
    /// * `shape_option` – Optional output shape; `None` produces a scalar tensor.
    /// * `data_type` – [`DataType`] of the constant tensor.
    ///
    /// # Returns
    ///
    /// A new [`Tensor`] containing the scalar value.
    pub fn constant_with_scalar(
        &self,
        scalar: f64,
        shape_option: Option<&[usize]>,
        data_type: DataType,
    ) -> Retained<Tensor> {
        match shape_option {
            Some(shape) => unsafe {
                msg_send![
                    self,
                    constantWithScalar: scalar,
                    shape: &*ns_number_array_from_slice(shape),
                    dataType: data_type
                ]
            },
            None => unsafe {
                msg_send![
                    self,
                    constantWithScalar: scalar,
                    dataType: data_type
                ]
            },
        }
    }

    /// Creates a complex scalar constant operation.
    ///
    /// # Arguments
    ///
    /// * `real_part` – Real component of the scalar.
    /// * `imaginary_part` – Imaginary component of the scalar.
    /// * `data_type` – Optional [`DataType`] (defaults to [`DataType::ComplexFloat32`]).
    ///
    /// # Returns
    ///
    /// A new [`Tensor`] containing the complex scalar.
    pub fn constant_with_real_imaginary(
        &self,
        real_part: f64,
        imaginary_part: f64,
        data_type: Option<DataType>,
    ) -> Retained<Tensor> {
        match data_type {
            Some(data_type) => unsafe {
                msg_send![
                    self,
                    constantWithRealPart: real_part,
                    imaginaryPart: imaginary_part,
                    dataType: data_type
                ]
            },
            None => unsafe {
                msg_send![
                    self,
                    constantWithRealPart: real_part,
                    imaginaryPart: imaginary_part
                ]
            },
        }
    }

    /// Creates a variable operation from raw bytes.
    ///
    /// # Arguments
    ///
    /// * `data` – Raw tensor bytes.
    /// * `shape` – Statically shaped output [`Shape`].
    /// * `data_type` – [`DataType`] of the variable tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A new variable [`Tensor`].
    pub fn variable_with_ns_data(
        &self,
        data: &NSData,
        shape: &[usize],
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                variableWithData: data,
                shape: &*ns_number_array_from_slice(shape),
                dataType: data_type,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Creates a variable operation from a slice of values.
    ///
    /// # Arguments
    ///
    /// * `data` – Slice containing tensor elements.
    /// * `shape` – Statically shaped output [`Shape`].
    /// * `data_type` – [`DataType`] of the variable tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A new variable [`Tensor`].
    pub fn variable_with_data<T: Copy>(
        &self,
        data: &[T],
        shape: &[usize],
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        autoreleasepool(|_| {
            let data_size = size_of_val(data);
            let ns_data = unsafe {
                NSData::with_bytes(from_raw_parts(data.as_ptr() as *const u8, data_size))
            };
            Self::variable_with_ns_data(self, &ns_data, shape, data_type, name)
        })
    }

    /// Creates a variable from another tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` – Source [`Tensor`].
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A new variable [`Tensor`].
    pub fn variable_from_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                variableFromTensorWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Reads the value of a variable tensor at this point in the graph.
    ///
    /// # Arguments
    ///
    /// * `variable` – Variable [`Tensor`] to read.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A [`Tensor`] representing the read result.
    pub fn read_variable(&self, variable: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                readVariable: variable,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Assigns `tensor` to `variable` at this point in the graph.
    ///
    /// # Arguments
    ///
    /// * `variable` – Variable [`Tensor`] to assign to.
    /// * `tensor` – Value [`Tensor`] written into the variable.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// An [`Operation`] representing the assignment.
    pub fn assign_variable(
        &self,
        variable: &Tensor,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Operation> {
        unsafe {
            msg_send![
                self,
                assignVariable: variable,
                withValueOfTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }
}
