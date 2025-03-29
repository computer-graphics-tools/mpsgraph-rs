# MPSGraph Convenience Extensions

This document describes convenience extensions from the mirai/llm-mpsgraph Swift codebase that can be ported to the mpsgraph-rs Rust project to improve developer experience and code readability.

## Overview

The mirai/llm-mpsgraph project implements several Swift extension types to make working with MPSGraph more ergonomic. These include:

1. **Operator Overloading** - Using standard operators like `+`, `-`, `*`, `/` for tensor operations
2. **Scalar-Tensor Operations** - Mixing scalars and tensors in operations
3. **Method Chaining** - Using dot notation for common operations
4. **Specialized Utility Functions** - Convenience wrappers for common ML operations

## Extensions to Port

### 1. Basic Arithmetic Operator Overloads

```rust
impl std::ops::Add for MPSGraphTensor {
    type Output = MPSGraphTensor;

    fn add(self, rhs: Self) -> Self::Output {
        // Ensure data types match
        // Return graph.addition(lhs, rhs)
    }
}

// Similar implementations for:
// - std::ops::Sub
// - std::ops::Mul
// - std::ops::Div
// - std::ops::Rem (modulo)
```

### 2. Comparison Operator Overloads

```rust
impl std::cmp::PartialEq for MPSGraphTensor {
    fn eq(&self, other: &Self) -> bool {
        // Return graph.equal(self, other) - returns a new tensor
    }
}

// Similar implementations for:
// - std::cmp::PartialOrd (for <, <=, >, >=)
```

### 3. Scalar-Tensor Operations

```rust
// Add scalar to tensor
impl std::ops::Add<f32> for MPSGraphTensor {
    type Output = MPSGraphTensor;

    fn add(self, rhs: f32) -> Self::Output {
        if rhs != 0.0 {
            let graph = self.operation.graph;
            let const_tensor = graph.constant(rhs, dataType: self.dataType);
            graph.addition(self, const_tensor, name: nil)
        } else {
            self
        }
    }
}

// Add tensor to scalar (reverse)
impl std::ops::Add<MPSGraphTensor> for f32 {
    type Output = MPSGraphTensor;

    fn add(self, rhs: MPSGraphTensor) -> Self::Output {
        rhs + self
    }
}

// Similar implementations for:
// - Subtraction
// - Multiplication
// - Division
// - Comparisons
```

### 4. Unary Operations as Methods

```rust
impl MPSGraphTensor {
    // Applies the natural exponent to the tensor elements
    pub fn exponent(&self, name: Option<&str>) -> MPSGraphTensor {
        self.operation.graph.exponent(with: self, name: name)
    }
    
    // Square
    pub fn square(&self, name: Option<&str>) -> MPSGraphTensor {
        self.operation.graph.square(with: self, name: name)
    }
    
    // Similar methods for:
    // - log, log2, log10
    // - sqrt, reciprocal_sqrt
    // - abs, neg
    // - sin, cos, tan
    // - etc.
}
```

### 5. Specialized ML Operations

```rust
impl MPSGraphTensor {
    // SiLU activation (Swish): x * sigmoid(x)
    pub fn silu(&self, name_prefix: Option<&str>) -> MPSGraphTensor {
        let sigmoid = self.operation.graph.sigmoid(with: self, 
                                            name: name_prefix.map(|s| format!("{}_sigmoid", s)));
        self.operation.graph.multiplication(self, sigmoid, name: nil)
    }
    
    // RMS normalization factor
    pub fn compute_rms_normalization_factor(
        &self, 
        epsilon: f32,
        accumulation_data_type: MPSDataType,
        name_prefix: Option<&str>
    ) -> MPSGraphTensor {
        // Implementation
    }
}
```

### 6. Shape Operations

```rust
impl MPSGraphTensor {
    // Get shape as a tensor
    pub fn shape_tensor(&self, name: Option<&str>) -> MPSGraphTensor {
        self.operation.graph.shape_of(self, name: name)
    }
    
    // Reshape with more ergonomic API
    pub fn reshape_to(&self, shape: &[i64], name: Option<&str>) -> MPSGraphTensor {
        // Implementation
    }
}
```

### 7. Type Conversion

```rust
impl MPSGraphTensor {
    // Cast only if needed (returns self if already the right type)
    pub fn cast_if_needed(&self, data_type: MPSDataType, name: Option<&str>) -> MPSGraphTensor {
        if self.data_type == data_type {
            return self.clone();
        }
        self.operation.graph.cast(self, to: data_type, name: name)
    }
}
```

### 8. Tensor Creation Helpers

```rust
impl MPSGraph {
    // Create constant tensor from scalar value with specific data type
    pub fn constant<T: Into<f64>>(&self, value: T, data_type: MPSDataType) -> MPSGraphTensor {
        // Implementation
    }
    
    // Create zeros/ones tensors with given shape
    pub fn zeros(&self, shape: &[i64], data_type: MPSDataType) -> MPSGraphTensor {
        // Implementation
    }
    
    pub fn ones(&self, shape: &[i64], data_type: MPSDataType) -> MPSGraphTensor {
        // Implementation
    }
}
```

## Implementation Considerations

1. **Data Types**: Swift has strong support for generics. In Rust, we may need to implement separate functions for different data types or use traits to constrain generic type parameters.

2. **Error Handling**: Swift uses optional return types (`?`) while Rust uses `Result<T, E>`. Consider using `Result` for operations that might fail.

3. **Memory Management**: Ensure proper clone/drop semantics for tensor operations.

4. **Method vs. Function Style**: Swift extensions tend to use method syntax. In Rust, consider if extension traits or impl blocks would be more idiomatic.

5. **Operator Overloading**: While Rust has good support for operator overloading, use it judiciously to maintain code clarity.

## Benefits

Implementing these convenience extensions will:

1. Reduce boilerplate code
2. Make tensor operations more concise and readable
3. Allow more natural mathematical expressions with tensors
4. Bring Rust API closer to what ML practitioners expect from a tensor library
5. Improve developer experience working with MPSGraph in Rust

## Example Usage (After Implementation)

```rust
// Before
let sum = graph.addition(tensor1, tensor2, None);
let scaled = graph.multiplication(sum, graph.constant(0.5, MPSDataType::Float32), None);
let activated = graph.tanh(with: scaled, None);

// After
let activated = (tensor1 + tensor2) * 0.5.into_tensor(graph, MPSDataType::Float32).tanh();
```

This shows how the convenience extensions can dramatically improve code readability while maintaining the same functionality.