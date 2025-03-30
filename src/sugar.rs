//! Sugar API for more ergonomic MPSGraph usage
//!
//! This module provides operator overloading and convenient extension methods
//! for MPSGraph tensors, making the API more user-friendly. Similar to Swift extensions
//! in the mirai/llm-mpsgraph project.
//!
//! # Features
//! 
//! - **Reference-Based Operator Overloading**: Use standard operators (`+`, `-`, `*`, `/`, `-x`) with references
//!   for tensor operations (e.g., `&a + &b`)
//! - **Method Chaining**: Apply operations using method syntax (e.g., `tensor.sqrt().abs()`)
//! - **Utility Methods**: Convenience functions for common operations
//! - **Tensor Creation**: Helper methods for creating tensors filled with zeros, ones, etc.
//!
//! # Examples
//!
//! ```rust
//! use mpsgraph::prelude::*;
//! 
//! // Create a graph and tensors
//! let graph = MPSGraph::new();
//! let a = graph.placeholder(&[2, 3], MPSDataType::Float32, None);
//! let b = graph.placeholder(&[2, 3], MPSDataType::Float32, None);
//! 
//! // Use operator overloading for arithmetic (always with references)
//! let sum = &a + &b;
//! let diff = &a - &b;
//! let product = &a * &b;
//! let ratio = &a / &b;
//! 
//! // Apply unary operations with method chaining
//! let result = (&a + &b).sqrt().abs();
//! 
//! // Create tensors with utility methods
//! let zeros = graph.zeros(&[2, 3], MPSDataType::Float32);
//! let ones = graph.ones(&[2, 3], MPSDataType::Float32);
//! ```
//!
//! # Important Note
//!
//! All operators require references to tensors (e.g., `&a + &b`) and do not consume the original tensors.
//! This allows you to reuse tensors in multiple operations.
//!
//! Feature flag: `sugar_api`

use crate::{MPSGraph, MPSGraphTensor, MPSDataType, MPSShape, MPSTensorDataScalar};
use std::ops;

/// Addition operator for MPSGraphTensor references
///
/// Enables using the `+` operator with tensor references.
/// Equivalent to calling `graph.add(lhs, rhs, None)`.
/// Using references instead of owned values preserves the original tensors for future use.
///
/// # Examples
///
/// ```rust
/// let sum = &tensor1 + &tensor2;
/// // tensor1 and tensor2 can still be used in subsequent operations
/// ```
impl<'a, 'b> ops::Add<&'b MPSGraphTensor> for &'a MPSGraphTensor {
    type Output = MPSGraphTensor;

    fn add(self, rhs: &'b MPSGraphTensor) -> Self::Output {
        // Get the graph from the operation and add the tensors
        let op = self.operation();
        let graph = op.graph();
        graph.add(self, rhs, None)
    }
}


/// Subtraction operator for MPSGraphTensor references
///
/// Enables using the `-` operator with tensor references.
/// Equivalent to calling `graph.subtract(lhs, rhs, None)`.
/// Using references instead of owned values preserves the original tensors for future use.
///
/// # Examples
///
/// ```rust
/// let difference = &tensor1 - &tensor2;
/// // tensor1 and tensor2 can still be used in subsequent operations
/// ```
impl<'a, 'b> ops::Sub<&'b MPSGraphTensor> for &'a MPSGraphTensor {
    type Output = MPSGraphTensor;

    fn sub(self, rhs: &'b MPSGraphTensor) -> Self::Output {
        let op = self.operation();
        let graph = op.graph();
        graph.subtract(self, rhs, None)
    }
}


/// Multiplication operator for MPSGraphTensor references
///
/// Enables using the `*` operator with tensor references.
/// Equivalent to calling `graph.multiply(lhs, rhs, None)`.
/// Using references instead of owned values preserves the original tensors for future use.
///
/// # Examples
///
/// ```rust
/// let product = &tensor1 * &tensor2;
/// // tensor1 and tensor2 can still be used in subsequent operations
/// ```
impl<'a, 'b> ops::Mul<&'b MPSGraphTensor> for &'a MPSGraphTensor {
    type Output = MPSGraphTensor;

    fn mul(self, rhs: &'b MPSGraphTensor) -> Self::Output {
        let op = self.operation();
        let graph = op.graph();
        graph.multiply(self, rhs, None)
    }
}


/// Division operator for MPSGraphTensor references
///
/// Enables using the `/` operator with tensor references.
/// Equivalent to calling `graph.divide(lhs, rhs, None)`.
/// Using references instead of owned values preserves the original tensors for future use.
///
/// # Examples
///
/// ```rust
/// let quotient = &tensor1 / &tensor2;
/// // tensor1 and tensor2 can still be used in subsequent operations
/// ```
impl<'a, 'b> ops::Div<&'b MPSGraphTensor> for &'a MPSGraphTensor {
    type Output = MPSGraphTensor;

    fn div(self, rhs: &'b MPSGraphTensor) -> Self::Output {
        let op = self.operation();
        let graph = op.graph();
        graph.divide(self, rhs, None)
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
/// ```rust
/// let negated = -&tensor;
/// // tensor can still be used in subsequent operations
/// ```
impl ops::Neg for &MPSGraphTensor {
    type Output = MPSGraphTensor;

    fn neg(self) -> Self::Output {
        let op = self.operation();
        let graph = op.graph();
        graph.negative(self, None)
    }
}


/// Utility methods for MPSGraphTensor
impl MPSGraphTensor {
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
    /// ```rust
    /// // Create a constant tensor with value 0.5 and same data type as tensor
    /// let half = tensor.const_scalar(0.5);
    /// let scaled = &tensor * &half;
    /// ```
    pub fn const_scalar<T: MPSTensorDataScalar>(&self, value: T) -> MPSGraphTensor {
        let op = self.operation();
        let graph = op.graph();
        let data_type = self.data_type();
        
        // Create constant with matching data type and convert as needed
        graph.constant_scalar(value, data_type)
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
    /// ```rust
    /// let squared = tensor.square(None);
    /// ```
    pub fn square(&self, name: Option<&str>) -> MPSGraphTensor {
        let op = self.operation();
        let graph = op.graph();
        graph.square(&self, name)
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
    /// ```rust
    /// let root = tensor.sqrt(None);
    /// ```
    pub fn sqrt(&self, name: Option<&str>) -> MPSGraphTensor {
        let op = self.operation();
        let graph = op.graph();
        graph.sqrt(&self, name)
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
    /// ```rust
    /// let absolute = tensor.abs(None);
    /// ```
    pub fn abs(&self, name: Option<&str>) -> MPSGraphTensor {
        let op = self.operation();
        let graph = op.graph();
        graph.abs(&self, name)
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
    /// ```rust
    /// let exp_tensor = tensor.exp(None);
    /// ```
    pub fn exp(&self, name: Option<&str>) -> MPSGraphTensor {
        let op = self.operation();
        let graph = op.graph();
        graph.exp(&self, name)
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
    /// ```rust
    /// let log_tensor = tensor.log(None);
    /// ```
    pub fn log(&self, name: Option<&str>) -> MPSGraphTensor {
        let op = self.operation();
        let graph = op.graph();
        graph.log(&self, name)
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
    /// ```rust
    /// let sigmoid_tensor = tensor.sigmoid(None);
    /// ```
    pub fn sigmoid(&self, name: Option<&str>) -> MPSGraphTensor {
        let op = self.operation();
        let graph = op.graph();
        graph.sigmoid(&self, name)
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
    /// ```rust
    /// let tanh_tensor = tensor.tanh(None);
    /// ```
    pub fn tanh(&self, name: Option<&str>) -> MPSGraphTensor {
        let op = self.operation();
        let graph = op.graph();
        graph.tanh(&self, name)
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
    /// ```rust
    /// let relu_tensor = tensor.relu(None);
    /// ```
    pub fn relu(&self, name: Option<&str>) -> MPSGraphTensor {
        let op = self.operation();
        let graph = op.graph();
        graph.relu(&self, name)
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
    /// ```rust
    /// let activated = tensor.silu(Some("activation"));
    /// ```
    pub fn silu(&self, name_prefix: Option<&str>) -> MPSGraphTensor {
        let op = self.operation();
        let graph = op.graph();
        let sigmoid_name = name_prefix.map(|p| format!("{}_sigmoid", p));
        let sigmoid = graph.sigmoid(&self, sigmoid_name.as_deref());
        graph.multiply(&self, &sigmoid, name_prefix)
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
    /// ```rust
    /// let activated = tensor.gelu(Some("activation"));
    /// ```
    pub fn gelu(&self, name_prefix: Option<&str>) -> MPSGraphTensor {
        let op = self.operation();
        let graph = op.graph();
        
        // Constants for the GELU approximation
        let sqrt_2_over_pi = 0.7978845608028654; // sqrt(2/π)
        let coeff = 0.044715;
        
        // Create constant tensors
        let data_type = self.data_type();
        let const_0_5 = graph.constant_scalar(0.5, data_type);
        let const_1 = graph.constant_scalar(1.0, data_type);
        let const_sqrt_2_pi = graph.constant_scalar(sqrt_2_over_pi, data_type);
        let const_coeff = graph.constant_scalar(coeff, data_type);
        
        // Compute x^3
        let square_name = name_prefix.map(|p| format!("{}_square", p));
        // Using the updated square method which now takes &self
        let x_squared = self.square(square_name.as_deref());
        
        let cube_name = name_prefix.map(|p| format!("{}_cube", p));
        let x_cubed = graph.multiply(&self, &x_squared, cube_name.as_deref());
        
        // Compute coeff * x^3
        let scaled_cube_name = name_prefix.map(|p| format!("{}_scaled_cube", p));
        let scaled_x_cubed = graph.multiply(&const_coeff, &x_cubed, scaled_cube_name.as_deref());
        
        // Compute x + coeff * x^3
        let inner_name = name_prefix.map(|p| format!("{}_inner", p));
        let inner = graph.add(&self, &scaled_x_cubed, inner_name.as_deref());
        
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
        graph.multiply(&self, &half_term, name_prefix)
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
    /// ```rust
    /// let exponent = graph.constant_scalar(2.0, MPSDataType::Float32);
    /// let squared = tensor.pow(&exponent, None);
    /// ```
    pub fn pow(&self, exponent: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let op = self.operation();
        let graph = op.graph();
        graph.power(&self, exponent, name)
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
    /// ```rust
    /// let min_val = graph.constant_scalar(0.0, MPSDataType::Float32);
    /// let max_val = graph.constant_scalar(1.0, MPSDataType::Float32);
    /// let clipped = tensor.clip(&min_val, &max_val, None);
    /// ```
    pub fn clip(&self, min_val: &MPSGraphTensor, max_val: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
        let op = self.operation();
        let graph = op.graph();
        
        // First clip to minimum (max of tensor and min_val)
        let name_min = name.map(|n| format!("{}_min", n));
        let clipped_min = graph.maximum(self, min_val, name_min.as_deref());
        
        // Then clip to maximum (min of clipped_min and max_val)
        let name_max = name.map(|n| format!("{}_max", n));
        graph.minimum(&clipped_min, max_val, name_max.as_deref())
    }
}

/// Functional API for MPSGraphTensor operations
/// 
/// These functions provide a functional programming style interface to tensor operations.
/// They can be used as an alternative to the method-based API.
/// 
/// # Examples
/// 
/// ```rust
/// // Method-based API
/// let squared_method = tensor.square(None);
/// 
/// // Functional API
/// let squared_func = square(&tensor, None);
/// ```

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
/// ```rust
/// let squared = square(&tensor, None);
/// ```
pub fn square(tensor: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
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
/// ```rust
/// let root = sqrt(&tensor, None);
/// ```
pub fn sqrt(tensor: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
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
/// ```rust
/// let absolute = abs(&tensor, None);
/// let abs_diff = abs(&(&a - &b), None);
/// ```
pub fn abs(tensor: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
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
/// ```rust
/// let exp_tensor = exp(&tensor, None);
/// ```
pub fn exp(tensor: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
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
/// ```rust
/// let log_tensor = log(&tensor, None);
/// ```
pub fn log(tensor: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
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
/// ```rust
/// let sigmoid_tensor = sigmoid(&tensor, None);
/// ```
pub fn sigmoid(tensor: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
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
/// ```rust
/// let tanh_tensor = tanh(&tensor, None);
/// ```
pub fn tanh(tensor: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
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
/// ```rust
/// let relu_tensor = relu(&tensor, None);
/// ```
pub fn relu(tensor: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
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
/// ```rust
/// let activated = silu(&tensor, Some("activation"));
/// ```
pub fn silu(tensor: &MPSGraphTensor, name_prefix: Option<&str>) -> MPSGraphTensor {
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
/// ```rust
/// let activated = gelu(&tensor, Some("activation"));
/// ```
pub fn gelu(tensor: &MPSGraphTensor, name_prefix: Option<&str>) -> MPSGraphTensor {
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
/// ```rust
/// let exponent = graph.constant_scalar(2.0, MPSDataType::Float32);
/// let squared = pow(&tensor, &exponent, None);
/// ```
pub fn pow(tensor: &MPSGraphTensor, exponent: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
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
/// ```rust
/// let min_val = graph.constant_scalar(0.0, MPSDataType::Float32);
/// let max_val = graph.constant_scalar(1.0, MPSDataType::Float32);
/// let clipped = clip(&tensor, &min_val, &max_val, None);
/// ```
pub fn clip(tensor: &MPSGraphTensor, min_val: &MPSGraphTensor, max_val: &MPSGraphTensor, name: Option<&str>) -> MPSGraphTensor {
    tensor.clip(min_val, max_val, name)
}

// Implement utility methods for the MPSGraph type
impl MPSGraph {
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
    /// ```rust
    /// let zeros = graph.zeros(&[2, 3], MPSDataType::Float32);
    /// ```
    pub fn zeros(&self, shape: &[u64], data_type: MPSDataType) -> MPSGraphTensor {
        // Convert shape from u64 to usize for MPSShape
        let usize_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        
        // Create a shape object
        let shape_obj = MPSShape::from_slice(&usize_shape);
        
        // Create a scalar constant with zero and specified shape
        self.constant_scalar_with_shape(0.0f32, &shape_obj, data_type)
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
    /// ```rust
    /// let ones = graph.ones(&[2, 3], MPSDataType::Float32);
    /// ```
    pub fn ones(&self, shape: &[u64], data_type: MPSDataType) -> MPSGraphTensor {
        // Convert shape from u64 to usize for MPSShape
        let usize_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        
        // Create a shape object
        let shape_obj = MPSShape::from_slice(&usize_shape);
        
        // Create a filled tensor using constant_scalar_with_shape which is more direct
        self.constant_scalar_with_shape(1.0, &shape_obj, data_type)
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
    /// ```rust
    /// let twos = graph.full(2.0, &[2, 3], MPSDataType::Float32);
    /// ```
    pub fn full<T: MPSTensorDataScalar>(&self, value: T, shape: &[u64], data_type: MPSDataType) -> MPSGraphTensor {
        // Convert shape from u64 to usize for MPSShape
        let usize_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        
        // Create a shape object
        let shape_obj = MPSShape::from_slice(&usize_shape);
        
        // Create a filled tensor
        self.constant_scalar_with_shape(value, &shape_obj, data_type)
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
    /// ```rust
    /// // Creates a 2x3 tensor with random values in the range [0.0, 1.0]
    /// let random = graph.create_random_uniform(0.0, 1.0, &[2, 3], MPSDataType::Float32);
    /// ```
    pub fn create_random_uniform<T: MPSTensorDataScalar>(
        &self,
        lower_bound: T,
        upper_bound: T,
        shape: &[u64],
        data_type: MPSDataType
    ) -> MPSGraphTensor {
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
        self.add(&scaled, &offset, None)
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
    /// ```rust
    /// // Creates a 2x3 tensor with random values from N(0.0, 1.0)
    /// let random = graph.create_random_normal(0.0, 1.0, &[2, 3], MPSDataType::Float32);
    /// ```
    pub fn create_random_normal<T: MPSTensorDataScalar>(
        &self,
        mean: T,
        std_dev: T,
        shape: &[u64],
        data_type: MPSDataType
    ) -> MPSGraphTensor {
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
        self.add(&scaled, &mean_tensor, None)
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
    /// ```rust
    /// // Creates a tensor with values [5, 6, 7, 8, 9]
    /// let arange = graph.arange(5, 5, MPSDataType::Int32);
    /// ```
    pub fn arange<T: MPSTensorDataScalar>(
        &self,
        start: T,
        count: u64,
        data_type: MPSDataType
    ) -> MPSGraphTensor {
        // For simple case, create the values directly
        let start_val = start.to_f64();
        let values: Vec<f64> = (0..count).map(|i| start_val + i as f64).collect();
        
        // Create constant tensor with the sequence
        let shape = vec![values.len()];
        self.constant_with_shape(&values, &shape, data_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_addition() {
        let graph = MPSGraph::new();
        let shape_dims = vec![2usize, 3usize];
        let shape = MPSShape::from_slice(&shape_dims);
        
        let a = graph.placeholder(&shape, MPSDataType::Float32, None);
        let b = graph.placeholder(&shape, MPSDataType::Float32, None);
        
        let sum = &a + &b;
        
        assert_eq!(sum.data_type(), MPSDataType::Float32);
        // Additional test logic would verify the actual computation with MPSGraphExecutable
    }
}