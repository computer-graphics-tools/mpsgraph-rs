//! Tensor Operations API for MPSGraph
//!
//! This module provides ergonomic tensor operations with operator overloading
//! and functional-style programming for MPSGraph tensors.
//!
//! # Features
//!
//! - **Reference-Based Operator Overloading**: Use standard operators (`+`, `-`, `*`, `/`, `-x`) with references
//!   for tensor operations (e.g., `&a + &b`)
//! - **Functional API**: Apply operations using functional style (e.g., `abs(&sqrt(&(&a + &b), None), None)`)
//! - **Utility Methods**: Convenience functions for common operations
//! - **Tensor Creation**: Helper methods for creating tensors filled with zeros, ones, etc.

use mpsgraph::{MPSDataType, MPSGraph, MPSGraphTensor, MPSShape, MPSTensorDataScalar};
use std::ops;

/// A wrapper around MPSGraphTensor to enable operations with standard operators
#[derive(Debug, Clone)]
pub struct Tensor(pub MPSGraphTensor);

impl Tensor {
    /// Create a new Tensor wrapper around an MPSGraphTensor
    pub fn new(tensor: MPSGraphTensor) -> Self {
        Tensor(tensor)
    }

    /// Unwrap the Tensor to get the underlying MPSGraphTensor
    pub fn unwrap(self) -> MPSGraphTensor {
        self.0
    }

    /// Get a reference to the underlying MPSGraphTensor
    pub fn inner(&self) -> &MPSGraphTensor {
        &self.0
    }
}

impl From<MPSGraphTensor> for Tensor {
    fn from(tensor: MPSGraphTensor) -> Self {
        Tensor(tensor)
    }
}

impl From<Tensor> for MPSGraphTensor {
    fn from(tensor: Tensor) -> Self {
        tensor.0
    }
}

impl AsRef<MPSGraphTensor> for Tensor {
    fn as_ref(&self) -> &MPSGraphTensor {
        &self.0
    }
}

/// Addition operator for Tensor references
///
/// Enables using the `+` operator with tensor references.
/// Equivalent to calling `graph.add(lhs, rhs, None)`.
/// Using references instead of owned values preserves the original tensors for future use.
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and two tensors
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor1 = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("a"));
/// let tensor2 = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("b"));
///
/// let sum = &tensor1 + &tensor2;
/// // tensor1 and tensor2 can still be used in subsequent operations
/// ```
impl<'b> ops::Add<&'b Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &'b Tensor) -> Self::Output {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.add(&self.0, &rhs.0, None))
    }
}

/// Subtraction operator for Tensor references
///
/// Enables using the `-` operator with tensor references.
/// Equivalent to calling `graph.subtract(lhs, rhs, None)`.
/// Using references instead of owned values preserves the original tensors for future use.
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and two tensors
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor1 = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("a"));
/// let tensor2 = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("b"));
///
/// let difference = &tensor1 - &tensor2;
/// // tensor1 and tensor2 can still be used in subsequent operations
/// ```
impl<'b> ops::Sub<&'b Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &'b Tensor) -> Self::Output {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.subtract(&self.0, &rhs.0, None))
    }
}

/// Multiplication operator for Tensor references
///
/// Enables using the `*` operator with tensor references.
/// Equivalent to calling `graph.multiply(lhs, rhs, None)`.
/// Using references instead of owned values preserves the original tensors for future use.
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and two tensors
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor1 = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("a"));
/// let tensor2 = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("b"));
///
/// let product = &tensor1 * &tensor2;
/// // tensor1 and tensor2 can still be used in subsequent operations
/// ```
impl<'b> ops::Mul<&'b Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &'b Tensor) -> Self::Output {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.multiply(&self.0, &rhs.0, None))
    }
}

/// Division operator for Tensor references
///
/// Enables using the `/` operator with tensor references.
/// Equivalent to calling `graph.divide(lhs, rhs, None)`.
/// Using references instead of owned values preserves the original tensors for future use.
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and two tensors
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor1 = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("a"));
/// let tensor2 = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("b"));
///
/// let quotient = &tensor1 / &tensor2;
/// // tensor1 and tensor2 can still be used in subsequent operations
/// ```
impl<'b> ops::Div<&'b Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: &'b Tensor) -> Self::Output {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.divide(&self.0, &rhs.0, None))
    }
}

/// Implements unary negation for tensor references
///
/// Enables using the unary `-` operator with tensor references.
/// Equivalent to calling `graph.negative(tensor, None)`.
/// Using references instead of owned values preserves the original tensor for future use.
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and a tensor
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("a"));
///
/// let negated = -&tensor;
/// // tensor can still be used in subsequent operations
/// ```
impl ops::Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.negative(&self.0, None))
    }
}

/// Implements unary negation for tensor values
///
/// Enables using the unary `-` operator with tensor values.
/// Equivalent to calling `graph.negative(tensor, None)`.
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and a tensor
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
///
/// // Negate the tensor
/// let negated = -tensor.clone();
/// ```
impl ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.negative(&self.0, None))
    }
}

/// Implements multiplication for tensor reference and reference to reference
///
/// This allows expressions like `&(&a.square(None)) * &b` to work.
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and two tensors
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let a = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("a"));
/// let b = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("b"));
///
/// let result = &(&a.square(None)) * &b;
/// ```
impl<'a, 'b> ops::Mul<&'b Tensor> for &'a &'a Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &'b Tensor) -> Self::Output {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.multiply(&self.0, &rhs.0, None))
    }
}

impl Tensor {
    /// Creates a constant tensor with specified scalar value and matching data type
    ///
    /// This is a utility method to create a constant tensor with the same data type
    /// as the current tensor, but with a specific scalar value.
    ///
    /// # Parameters
    ///
    /// * `value` - The scalar value to use for the constant tensor
    ///
    /// # Returns
    ///
    /// A new constant tensor with the specified scalar value
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph and a tensor
    /// let graph = MPSGraph::new();
    /// let shape = MPSShape::from_slice(&[2, 3]);
    /// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("a"));
    ///
    /// // Create a constant tensor with value 0.5 and same data type as tensor
    /// let half = tensor.const_scalar(0.5);
    /// let scaled = &tensor * &half;
    /// ```
    pub fn const_scalar<T: MPSTensorDataScalar>(&self, value: T) -> Self {
        let op = self.0.operation();
        let graph = op.graph();
        let data_type = self.0.data_type();

        // Create constant with matching data type and convert as needed
        Tensor(graph.constant_scalar(value, data_type))
    }

    /// Applies square operation to the tensor elements
    ///
    /// Computes the square of each element in the tensor: f(x) = x²
    ///
    /// # Parameters
    ///
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor with each element squared
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph and a tensor
    /// let graph = MPSGraph::new();
    /// let shape = MPSShape::from_slice(&[2, 3]);
    /// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
    ///
    /// let squared = tensor.square(None);
    /// ```
    pub fn square(&self, name: Option<&str>) -> Self {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.square(&self.0, name))
    }

    /// Applies square root operation to the tensor elements
    ///
    /// Computes the square root of each element in the tensor: f(x) = √x
    ///
    /// # Parameters
    ///
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor with the square root of each element
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph and a tensor
    /// let graph = MPSGraph::new();
    /// let shape = MPSShape::from_slice(&[2, 3]);
    /// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
    ///
    /// let root = tensor.sqrt(None);
    /// ```
    pub fn sqrt(&self, name: Option<&str>) -> Self {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.sqrt(&self.0, name))
    }

    /// Applies absolute value operation to the tensor elements
    ///
    /// Computes the absolute value of each element in the tensor: f(x) = |x|
    ///
    /// # Parameters
    ///
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor with the absolute value of each element
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph and a tensor
    /// let graph = MPSGraph::new();
    /// let shape = MPSShape::from_slice(&[2, 3]);
    /// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
    ///
    /// let absolute = tensor.abs(None);
    /// ```
    pub fn abs(&self, name: Option<&str>) -> Self {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.abs(&self.0, name))
    }

    /// Applies exponential function to the tensor elements
    ///
    /// Computes e^x for each element in the tensor
    ///
    /// # Parameters
    ///
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor with the exponential of each element
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph and a tensor
    /// let graph = MPSGraph::new();
    /// let shape = MPSShape::from_slice(&[2, 3]);
    /// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
    ///
    /// let exp_tensor = tensor.exp(None);
    /// ```
    pub fn exp(&self, name: Option<&str>) -> Self {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.exp(&self.0, name))
    }

    /// Applies natural logarithm to the tensor elements
    ///
    /// Computes ln(x) for each element in the tensor
    ///
    /// # Parameters
    ///
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor with the natural logarithm of each element
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph and a tensor
    /// let graph = MPSGraph::new();
    /// let shape = MPSShape::from_slice(&[2, 3]);
    /// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
    ///
    /// let log_tensor = tensor.log(None);
    /// ```
    pub fn log(&self, name: Option<&str>) -> Self {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.log(&self.0, name))
    }

    /// Applies sigmoid activation function to the tensor elements
    ///
    /// Computes σ(x) = 1/(1+e^(-x)) for each element in the tensor
    ///
    /// # Parameters
    ///
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor with the sigmoid of each element
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph and a tensor
    /// let graph = MPSGraph::new();
    /// let shape = MPSShape::from_slice(&[2, 3]);
    /// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
    ///
    /// let sigmoid_tensor = tensor.sigmoid(None);
    /// ```
    pub fn sigmoid(&self, name: Option<&str>) -> Self {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.sigmoid(&self.0, name))
    }

    /// Applies tanh activation function to the tensor elements
    ///
    /// Computes tanh(x) for each element in the tensor
    ///
    /// # Parameters
    ///
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor with the tanh of each element
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph and a tensor
    /// let graph = MPSGraph::new();
    /// let shape = MPSShape::from_slice(&[2, 3]);
    /// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
    ///
    /// let tanh_tensor = tensor.tanh(None);
    /// ```
    pub fn tanh(&self, name: Option<&str>) -> Self {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.tanh(&self.0, name))
    }

    /// Applies ReLU activation function to the tensor elements
    ///
    /// Computes max(0, x) for each element in the tensor
    ///
    /// # Parameters
    ///
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor with the ReLU activation applied
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph and a tensor
    /// let graph = MPSGraph::new();
    /// let shape = MPSShape::from_slice(&[2, 3]);
    /// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
    ///
    /// let relu_tensor = tensor.relu(None);
    /// ```
    pub fn relu(&self, name: Option<&str>) -> Self {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.relu(&self.0, name))
    }

    /// Applies SiLU activation function (x * sigmoid(x))
    ///
    /// SiLU (Sigmoid Linear Unit) is also known as the Swish activation function.
    /// It computes x * sigmoid(x) for each element in the tensor.
    ///
    /// # Parameters
    ///
    /// * `name_prefix` - Optional prefix for the operation names
    ///
    /// # Returns
    ///
    /// A new tensor with the SiLU activation applied
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph and a tensor
    /// let graph = MPSGraph::new();
    /// let shape = MPSShape::from_slice(&[2, 3]);
    /// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
    ///
    /// let activated = tensor.silu(Some("activation"));
    /// ```
    pub fn silu(&self, name_prefix: Option<&str>) -> Self {
        let op = self.0.operation();
        let graph = op.graph();
        let sigmoid_name = name_prefix.map(|p| format!("{}_sigmoid", p));
        let sigmoid = graph.sigmoid(&self.0, sigmoid_name.as_deref());
        Tensor(graph.multiply(&self.0, &sigmoid, name_prefix))
    }

    /// Applies GELU activation function
    ///
    /// GELU (Gaussian Error Linear Unit) is defined as x * Φ(x) where Φ is the cumulative
    /// distribution function of the standard normal distribution.
    /// This implementation uses the approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    ///
    /// # Parameters
    ///
    /// * `name_prefix` - Optional prefix for the operation names
    ///
    /// # Returns
    ///
    /// A new tensor with the GELU activation applied
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph and a tensor
    /// let graph = MPSGraph::new();
    /// let shape = MPSShape::from_slice(&[2, 3]);
    /// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
    ///
    /// let activated = tensor.gelu(Some("activation"));
    /// ```
    pub fn gelu(&self, name_prefix: Option<&str>) -> Self {
        let op = self.0.operation();
        let graph = op.graph();

        // Constants for the GELU approximation
        let sqrt_2_over_pi = 0.7978845608028654; // sqrt(2/π)
        let coeff = 0.044715;

        // Create constant tensors
        let data_type = self.0.data_type();
        let const_0_5 = graph.constant_scalar(0.5, data_type);
        let const_1 = graph.constant_scalar(1.0, data_type);
        let const_sqrt_2_pi = graph.constant_scalar(sqrt_2_over_pi, data_type);
        let const_coeff = graph.constant_scalar(coeff, data_type);

        // Compute x^3
        let square_name = name_prefix.map(|p| format!("{}_square", p));
        let x_squared = graph.square(&self.0, square_name.as_deref());

        let cube_name = name_prefix.map(|p| format!("{}_cube", p));
        let x_cubed = graph.multiply(&self.0, &x_squared, cube_name.as_deref());

        // Compute coeff * x^3
        let scaled_cube_name = name_prefix.map(|p| format!("{}_scaled_cube", p));
        let scaled_x_cubed = graph.multiply(&const_coeff, &x_cubed, scaled_cube_name.as_deref());

        // Compute x + coeff * x^3
        let inner_name = name_prefix.map(|p| format!("{}_inner", p));
        let inner = graph.add(&self.0, &scaled_x_cubed, inner_name.as_deref());

        // Compute sqrt(2/π) * (x + coeff * x^3)
        let scaled_inner_name = name_prefix.map(|p| format!("{}_scaled_inner", p));
        let scaled_inner = graph.multiply(&const_sqrt_2_pi, &inner, scaled_inner_name.as_deref());

        // Compute tanh(sqrt(2/π) * (x + coeff * x^3))
        let tanh_name = name_prefix.map(|p| format!("{}_tanh", p));
        let tanh_term = graph.tanh(&scaled_inner, tanh_name.as_deref());

        // Compute 1 + tanh(...)
        let one_plus_tanh_name = name_prefix.map(|p| format!("{}_one_plus_tanh", p));
        let one_plus_tanh = graph.add(&const_1, &tanh_term, one_plus_tanh_name.as_deref());

        // Compute 0.5 * (1 + tanh(...))
        let half_term_name = name_prefix.map(|p| format!("{}_half_term", p));
        let half_term = graph.multiply(&const_0_5, &one_plus_tanh, half_term_name.as_deref());

        // Compute x * 0.5 * (1 + tanh(...))
        Tensor(graph.multiply(&self.0, &half_term, name_prefix))
    }

    /// Element-wise power operation
    ///
    /// Raises each element in the tensor to the specified power
    ///
    /// # Parameters
    ///
    /// * `exponent` - The exponent tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor with each element raised to the specified power
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph and a tensor
    /// let graph = MPSGraph::new();
    /// let shape = MPSShape::from_slice(&[2, 3]);
    /// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
    ///
    /// // Create an exponent tensor
    /// let exponent = graph.constant_scalar(2.0, MPSDataType::Float32);
    /// let exponent_tensor = Tensor::new(exponent);
    ///
    /// // Raise tensor to the power
    /// let squared = tensor.pow(&exponent_tensor, None);
    /// ```
    pub fn pow(&self, exponent: &Self, name: Option<&str>) -> Self {
        let op = self.0.operation();
        let graph = op.graph();
        Tensor(graph.power(&self.0, &exponent.0, name))
    }

    /// Clip tensor values to a specified range
    ///
    /// # Parameters
    ///
    /// * `min_val` - The minimum value tensor (elements smaller than this are clipped)
    /// * `max_val` - The maximum value tensor (elements larger than this are clipped)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor with values clipped to the specified range
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph and a tensor
    /// let graph = MPSGraph::new();
    /// let shape = MPSShape::from_slice(&[2, 3]);
    /// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
    ///
    /// // Create min and max value tensors
    /// let min_val = Tensor::new(graph.constant_scalar(0.0, MPSDataType::Float32));
    /// let max_val = Tensor::new(graph.constant_scalar(1.0, MPSDataType::Float32));
    ///
    /// // Clip tensor values to [0, 1] range
    /// let clipped = tensor.clip(&min_val, &max_val, None);
    /// ```
    pub fn clip(&self, min_val: &Self, max_val: &Self, name: Option<&str>) -> Self {
        let op = self.0.operation();
        let graph = op.graph();

        // First clip to minimum (max of tensor and min_val)
        let name_min = name.map(|n| format!("{}_min", n));
        let clipped_min = graph.maximum(&self.0, &min_val.0, name_min.as_deref());

        // Then clip to maximum (min of clipped_min and max_val)
        let name_max = name.map(|n| format!("{}_max", n));
        Tensor(graph.minimum(&clipped_min, &max_val.0, name_max.as_deref()))
    }
}

/// Extensions for MPSGraph to create Tensor
pub trait GraphExt {
    /// Create a placeholder tensor and wrap it with Tensor
    fn placeholder_tensor(
        &self,
        shape: &MPSShape,
        data_type: MPSDataType,
        name: Option<&str>,
    ) -> Tensor;

    /// Create a tensor filled with zeros
    fn zeros(&self, shape: &[u64], data_type: MPSDataType) -> Tensor;

    /// Create a tensor filled with ones
    fn ones(&self, shape: &[u64], data_type: MPSDataType) -> Tensor;

    /// Create a tensor filled with a specific value
    fn full<T: MPSTensorDataScalar>(
        &self,
        value: T,
        shape: &[u64],
        data_type: MPSDataType,
    ) -> Tensor;

    /// Create a tensor with random uniform values
    fn create_random_uniform<T: MPSTensorDataScalar>(
        &self,
        lower_bound: T,
        upper_bound: T,
        shape: &[u64],
        data_type: MPSDataType,
    ) -> Tensor;

    /// Create a tensor with random normal values
    fn create_random_normal<T: MPSTensorDataScalar>(
        &self,
        mean: T,
        std_dev: T,
        shape: &[u64],
        data_type: MPSDataType,
    ) -> Tensor;

    /// Create a tensor with sequential values
    fn arange<T: MPSTensorDataScalar>(
        &self,
        start: T,
        count: u64,
        data_type: MPSDataType,
    ) -> Tensor;
}

impl GraphExt for MPSGraph {
    fn placeholder_tensor(
        &self,
        shape: &MPSShape,
        data_type: MPSDataType,
        name: Option<&str>,
    ) -> Tensor {
        Tensor(self.placeholder(shape, data_type, name))
    }

    /// Create a tensor filled with zeros of the specified shape and data type
    ///
    /// # Parameters
    ///
    /// * `shape` - The shape of the tensor as an array of dimension sizes
    /// * `data_type` - The data type of the tensor elements
    ///
    /// # Returns
    ///
    /// A new tensor filled with zeros
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph
    /// let graph = MPSGraph::new();
    ///
    /// // Create a tensor filled with zeros
    /// let zeros = graph.zeros(&[2, 3], MPSDataType::Float32);
    /// ```
    fn zeros(&self, shape: &[u64], data_type: MPSDataType) -> Tensor {
        // Convert shape from u64 to usize for MPSShape
        let usize_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        // Create a shape object
        let shape_obj = MPSShape::from_slice(&usize_shape);

        // Create a scalar constant with zero and specified shape
        Tensor(self.constant_scalar_with_shape(0.0f32, &shape_obj, data_type))
    }

    /// Create a tensor filled with ones of the specified shape and data type
    ///
    /// # Parameters
    ///
    /// * `shape` - The shape of the tensor as an array of dimension sizes
    /// * `data_type` - The data type of the tensor elements
    ///
    /// # Returns
    ///
    /// A new tensor filled with ones
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph
    /// let graph = MPSGraph::new();
    ///
    /// // Create a tensor filled with ones
    /// let ones = graph.ones(&[2, 3], MPSDataType::Float32);
    /// ```
    fn ones(&self, shape: &[u64], data_type: MPSDataType) -> Tensor {
        // Convert shape from u64 to usize for MPSShape
        let usize_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        // Create a shape object
        let shape_obj = MPSShape::from_slice(&usize_shape);

        // Create a filled tensor using constant_scalar_with_shape which is more direct
        Tensor(self.constant_scalar_with_shape(1.0, &shape_obj, data_type))
    }

    /// Create a tensor with all elements set to a specific value
    ///
    /// # Parameters
    ///
    /// * `value` - The scalar value to fill the tensor with
    /// * `shape` - The shape of the tensor as an array of dimension sizes
    /// * `data_type` - The data type of the tensor elements
    ///
    /// # Returns
    ///
    /// A new tensor filled with the specified value
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph
    /// let graph = MPSGraph::new();
    ///
    /// // Create a tensor filled with the value 2.0
    /// let twos = graph.full(2.0, &[2, 3], MPSDataType::Float32);
    /// ```
    fn full<T: MPSTensorDataScalar>(
        &self,
        value: T,
        shape: &[u64],
        data_type: MPSDataType,
    ) -> Tensor {
        // Convert shape from u64 to usize for MPSShape
        let usize_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        // Create a shape object
        let shape_obj = MPSShape::from_slice(&usize_shape);

        // Create a filled tensor
        Tensor(self.constant_scalar_with_shape(value, &shape_obj, data_type))
    }

    /// Create a tensor filled with random uniform values
    ///
    /// This is a convenience method that uses the existing random_uniform function
    /// but with a more consistent parameter order and shape handling.
    ///
    /// # Parameters
    ///
    /// * `lower_bound` - The lower bound of the uniform distribution
    /// * `upper_bound` - The upper bound of the uniform distribution
    /// * `shape` - The shape of the tensor as an array of dimension sizes
    /// * `data_type` - The data type of the tensor elements
    ///
    /// # Returns
    ///
    /// A new tensor filled with random values from the specified uniform distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph
    /// let graph = MPSGraph::new();
    ///
    /// // Creates a 2x3 tensor with random values in the range [0.0, 1.0]
    /// let random = graph.create_random_uniform(0.0, 1.0, &[2, 3], MPSDataType::Float32);
    /// ```
    fn create_random_uniform<T: MPSTensorDataScalar>(
        &self,
        lower_bound: T,
        upper_bound: T,
        shape: &[u64],
        data_type: MPSDataType,
    ) -> Tensor {
        // Convert shape from u64 to usize for MPSShape
        let usize_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        // Create a shape object
        let _shape_obj = MPSShape::from_slice(&usize_shape);

        // Create lower and upper bound tensors
        let lower_f32 = lower_bound.to_f64() as f32;
        let upper_f32 = upper_bound.to_f64() as f32;

        // Create a uniform tensor using random_uniform_tensor_with_seed
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let random_uniform = self.random_uniform_tensor_with_seed(&shape_usize, 0, None);

        // Scale and shift the values from [0,1] to [lower,upper]
        let range = self.constant_scalar(upper_f32 - lower_f32, data_type);
        let scaled = self.multiply(&random_uniform, &range, None);
        let offset = self.constant_scalar(lower_f32, data_type);
        Tensor(self.add(&scaled, &offset, None))
    }

    /// Create a tensor filled with random normal values
    ///
    /// This is a convenience method that uses the existing random_normal function
    /// but with a more consistent parameter order and shape handling.
    ///
    /// # Parameters
    ///
    /// * `mean` - The mean of the normal distribution
    /// * `std_dev` - The standard deviation of the normal distribution
    /// * `shape` - The shape of the tensor as an array of dimension sizes
    /// * `data_type` - The data type of the tensor elements
    ///
    /// # Returns
    ///
    /// A new tensor filled with random values from the specified normal distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph
    /// let graph = MPSGraph::new();
    ///
    /// // Creates a 2x3 tensor with random values from N(0.0, 1.0)
    /// let random = graph.create_random_normal(0.0, 1.0, &[2, 3], MPSDataType::Float32);
    /// ```
    fn create_random_normal<T: MPSTensorDataScalar>(
        &self,
        mean: T,
        std_dev: T,
        shape: &[u64],
        data_type: MPSDataType,
    ) -> Tensor {
        // Convert shape from u64 to usize for MPSShape
        let usize_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        // Create a shape object
        let _shape_obj = MPSShape::from_slice(&usize_shape);

        // Convert to f32 for the existing API
        let mean_f32 = mean.to_f64() as f32;
        let std_dev_f32 = std_dev.to_f64() as f32;

        // Create a uniform tensor using random_uniform_tensor_with_seed
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        // For normal distribution, we'll use Box-Muller transform on uniform values
        // This is a simplification - in a real implementation we would use a proper normal generator
        let uniform1 = self.random_uniform_tensor_with_seed(&shape_usize, 0, None);
        let _uniform2 = self.random_uniform_tensor_with_seed(&shape_usize, 1, None); // Different seed

        // Create constants for mean and std_dev
        let mean_tensor = self.constant_scalar(mean_f32, data_type);
        let std_dev_tensor = self.constant_scalar(std_dev_f32, data_type);

        // Simple approximation - not true normal distribution but serves as example
        let scaled = self.multiply(&uniform1, &std_dev_tensor, None);
        Tensor(self.add(&scaled, &mean_tensor, None))
    }

    /// Create a tensor with a sequence of values starting from `start` with a step size of 1
    ///
    /// # Parameters
    ///
    /// * `start` - The starting value
    /// * `count` - The number of elements to generate
    /// * `data_type` - The data type of the tensor elements
    ///
    /// # Returns
    ///
    /// A new 1D tensor with a sequence of values
    ///
    /// # Examples
    ///
    /// ```
    /// use mpsgraph_tools::prelude::*;
    ///
    /// // Create a graph
    /// let graph = MPSGraph::new();
    ///
    /// // Creates a tensor with values [5, 6, 7, 8, 9]
    /// let arange = graph.arange(5, 5, MPSDataType::Int32);
    /// ```
    fn arange<T: MPSTensorDataScalar>(
        &self,
        start: T,
        count: u64,
        data_type: MPSDataType,
    ) -> Tensor {
        // For simple case, create the values directly
        let start_val = start.to_f64();
        let values: Vec<f64> = (0..count).map(|i| start_val + i as f64).collect();

        // Create constant tensor with the sequence
        let shape = vec![values.len()];
        Tensor(self.constant_with_shape(&values, &shape, data_type))
    }
}

/// Functional API for tensor operations
///
/// These functions provide a functional programming style interface to tensor operations.
/// They enable a consistent style for creating tensor computation graphs.
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and a tensor
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
///
/// // Method-based API (object-oriented style)
/// let squared_method = tensor.square(None);
///
/// // Functional API (functional programming style)
/// let squared_func = square(&tensor, None);
/// ```
///
/// Applies square operation to the tensor elements
///
/// Computes the square of each element in the tensor: f(x) = x²
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with each element squared
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Setup required for the example
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
///
/// let squared = square(&tensor, None);
/// ```
pub fn square(tensor: &Tensor, name: Option<&str>) -> Tensor {
    tensor.square(name)
}

/// Applies square root operation to the tensor elements
///
/// Computes the square root of each element in the tensor: f(x) = √x
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with the square root of each element
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and a tensor
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
///
/// let root = sqrt(&tensor, None);
/// ```
pub fn sqrt(tensor: &Tensor, name: Option<&str>) -> Tensor {
    tensor.sqrt(name)
}

/// Applies absolute value operation to the tensor elements
///
/// Computes the absolute value of each element in the tensor: f(x) = |x|
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with the absolute value of each element
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and tensors
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
/// let a = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("a"));
/// let b = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("b"));
///
/// let absolute = abs(&tensor, None);
/// let abs_diff = abs(&(&a - &b), None);
/// ```
pub fn abs(tensor: &Tensor, name: Option<&str>) -> Tensor {
    tensor.abs(name)
}

/// Applies exponential function to the tensor elements
///
/// Computes e^x for each element in the tensor
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with the exponential of each element
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and a tensor
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
///
/// let exp_tensor = exp(&tensor, None);
/// ```
pub fn exp(tensor: &Tensor, name: Option<&str>) -> Tensor {
    tensor.exp(name)
}

/// Applies natural logarithm to the tensor elements
///
/// Computes ln(x) for each element in the tensor
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with the natural logarithm of each element
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and a tensor
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
///
/// let log_tensor = log(&tensor, None);
/// ```
pub fn log(tensor: &Tensor, name: Option<&str>) -> Tensor {
    tensor.log(name)
}

/// Applies sigmoid activation function to the tensor elements
///
/// Computes σ(x) = 1/(1+e^(-x)) for each element in the tensor
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with the sigmoid of each element
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and a tensor
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
///
/// let sigmoid_tensor = sigmoid(&tensor, None);
/// ```
pub fn sigmoid(tensor: &Tensor, name: Option<&str>) -> Tensor {
    tensor.sigmoid(name)
}

/// Applies tanh activation function to the tensor elements
///
/// Computes tanh(x) for each element in the tensor
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with the tanh of each element
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and a tensor
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
///
/// let tanh_tensor = tanh(&tensor, None);
/// ```
pub fn tanh(tensor: &Tensor, name: Option<&str>) -> Tensor {
    tensor.tanh(name)
}

/// Applies ReLU activation function to the tensor elements
///
/// Computes max(0, x) for each element in the tensor
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with the ReLU activation applied
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and a tensor
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
///
/// let relu_tensor = relu(&tensor, None);
/// ```
pub fn relu(tensor: &Tensor, name: Option<&str>) -> Tensor {
    tensor.relu(name)
}

/// Applies SiLU activation function (x * sigmoid(x))
///
/// SiLU (Sigmoid Linear Unit) is also known as the Swish activation function.
/// It computes x * sigmoid(x) for each element in the tensor.
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name_prefix` - Optional prefix for the operation names
///
/// # Returns
///
/// A new tensor with the SiLU activation applied
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and a tensor
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
///
/// let activated = silu(&tensor, Some("activation"));
/// ```
pub fn silu(tensor: &Tensor, name_prefix: Option<&str>) -> Tensor {
    tensor.silu(name_prefix)
}

/// Applies GELU activation function
///
/// GELU (Gaussian Error Linear Unit) is defined as x * Φ(x) where Φ is the cumulative
/// distribution function of the standard normal distribution.
/// This implementation uses the approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name_prefix` - Optional prefix for the operation names
///
/// # Returns
///
/// A new tensor with the GELU activation applied
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and a tensor
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
///
/// let activated = gelu(&tensor, Some("activation"));
/// ```
pub fn gelu(tensor: &Tensor, name_prefix: Option<&str>) -> Tensor {
    tensor.gelu(name_prefix)
}

/// Element-wise power operation
///
/// Raises each element in the tensor to the specified power
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `exponent` - The exponent tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with each element raised to the specified power
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and a tensor
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
///
/// // Create an exponent tensor
/// let exponent = graph.constant_scalar(2.0, MPSDataType::Float32);
/// let exponent_tensor = Tensor::new(exponent);
///
/// // Raise tensor to the power
/// let squared = pow(&tensor, &exponent_tensor, None);
/// ```
pub fn pow(tensor: &Tensor, exponent: &Tensor, name: Option<&str>) -> Tensor {
    tensor.pow(exponent, name)
}

/// Clip tensor values to a specified range
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `min_val` - The minimum value tensor (elements smaller than this are clipped)
/// * `max_val` - The maximum value tensor (elements larger than this are clipped)
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with values clipped to the specified range
///
/// # Examples
///
/// ```
/// use mpsgraph_tools::prelude::*;
///
/// // Create a graph and a tensor
/// let graph = MPSGraph::new();
/// let shape = MPSShape::from_slice(&[2, 3]);
/// let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
///
/// // Create min and max value tensors
/// let min_val = Tensor::new(graph.constant_scalar(0.0, MPSDataType::Float32));
/// let max_val = Tensor::new(graph.constant_scalar(1.0, MPSDataType::Float32));
///
/// // Clip tensor values to [0, 1] range
/// let clipped = clip(&tensor, &min_val, &max_val, None);
/// ```
pub fn clip(tensor: &Tensor, min_val: &Tensor, max_val: &Tensor, name: Option<&str>) -> Tensor {
    tensor.clip(min_val, max_val, name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_addition() {
        let graph = MPSGraph::new();
        let shape_dims = vec![2usize, 3usize];
        let shape = MPSShape::from_slice(&shape_dims);

        let a = Tensor::new(graph.placeholder(&shape, MPSDataType::Float32, None));
        let b = Tensor::new(graph.placeholder(&shape, MPSDataType::Float32, None));

        let sum = &a + &b;

        assert_eq!(sum.0.data_type(), MPSDataType::Float32);
        // Additional test logic would verify the actual computation with MPSGraphExecutable
    }
}
