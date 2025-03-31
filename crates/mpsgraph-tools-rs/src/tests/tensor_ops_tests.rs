use crate::tensor_ops::{abs, clip, exp, gelu, log, pow, relu, sigmoid, silu, sqrt, square, tanh};
use crate::tensor_ops::{GraphExt, Tensor};
use mpsgraph::{MPSDataType, MPSGraph, MPSGraphTensor, MPSShape};

#[test]
fn test_tensor_creation() {
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[2usize, 3usize]);

    // Test creating a placeholder tensor
    let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, None);
    assert_eq!(tensor.inner().data_type(), MPSDataType::Float32);

    // Test creating a zero-filled tensor
    let zeros = graph.zeros(&[2, 3], MPSDataType::Float32);
    assert_eq!(zeros.inner().data_type(), MPSDataType::Float32);

    // Test creating a ones-filled tensor
    let ones = graph.ones(&[2, 3], MPSDataType::Float32);
    assert_eq!(ones.inner().data_type(), MPSDataType::Float32);

    // Test creating a full tensor with specific value
    let full = graph.full(5.0f32, &[2, 3], MPSDataType::Float32);
    assert_eq!(full.inner().data_type(), MPSDataType::Float32);

    // Test creating a sequence
    let sequence = graph.arange(0.0f32, 5, MPSDataType::Float32);
    assert_eq!(sequence.inner().data_type(), MPSDataType::Float32);
}

#[test]
fn test_tensor_arithmetic_operators() {
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[2usize, 3usize]);

    let a = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("a"));
    let b = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("b"));

    // Test addition
    let sum = &a + &b;
    assert_eq!(sum.inner().data_type(), MPSDataType::Float32);

    // Test subtraction
    let diff = &a - &b;
    assert_eq!(diff.inner().data_type(), MPSDataType::Float32);

    // Test multiplication
    let product = &a * &b;
    assert_eq!(product.inner().data_type(), MPSDataType::Float32);

    // Test division
    let quotient = &a / &b;
    assert_eq!(quotient.inner().data_type(), MPSDataType::Float32);

    // Test negation
    let negated = -&a;
    assert_eq!(negated.inner().data_type(), MPSDataType::Float32);

    // Test chained operations
    let chain = &(&a + &b) * &(&a - &b);
    assert_eq!(chain.inner().data_type(), MPSDataType::Float32);
}

#[test]
fn test_tensor_unary_operations() {
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[2usize, 3usize]);
    let a = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("a"));

    // Test square
    let squared = square(&a, None);
    assert_eq!(squared.inner().data_type(), MPSDataType::Float32);
    let squared_method = a.square(None);
    assert_eq!(squared_method.inner().data_type(), MPSDataType::Float32);

    // Test sqrt
    let sqrt_a = sqrt(&a, None);
    assert_eq!(sqrt_a.inner().data_type(), MPSDataType::Float32);
    let sqrt_method = a.sqrt(None);
    assert_eq!(sqrt_method.inner().data_type(), MPSDataType::Float32);

    // Test abs
    let abs_a = abs(&a, None);
    assert_eq!(abs_a.inner().data_type(), MPSDataType::Float32);
    let abs_method = a.abs(None);
    assert_eq!(abs_method.inner().data_type(), MPSDataType::Float32);

    // Test exp
    let exp_a = exp(&a, None);
    assert_eq!(exp_a.inner().data_type(), MPSDataType::Float32);
    let exp_method = a.exp(None);
    assert_eq!(exp_method.inner().data_type(), MPSDataType::Float32);

    // Test log
    let log_a = log(&a, None);
    assert_eq!(log_a.inner().data_type(), MPSDataType::Float32);
    let log_method = a.log(None);
    assert_eq!(log_method.inner().data_type(), MPSDataType::Float32);
}

#[test]
fn test_activation_functions() {
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[2usize, 3usize]);
    let a = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("a"));

    // Test sigmoid
    let sigmoid_a = sigmoid(&a, None);
    assert_eq!(sigmoid_a.inner().data_type(), MPSDataType::Float32);
    let sigmoid_method = a.sigmoid(None);
    assert_eq!(sigmoid_method.inner().data_type(), MPSDataType::Float32);

    // Test tanh
    let tanh_a = tanh(&a, None);
    assert_eq!(tanh_a.inner().data_type(), MPSDataType::Float32);
    let tanh_method = a.tanh(None);
    assert_eq!(tanh_method.inner().data_type(), MPSDataType::Float32);

    // Test relu
    let relu_a = relu(&a, None);
    assert_eq!(relu_a.inner().data_type(), MPSDataType::Float32);
    let relu_method = a.relu(None);
    assert_eq!(relu_method.inner().data_type(), MPSDataType::Float32);

    // Test silu
    let silu_a = silu(&a, None);
    assert_eq!(silu_a.inner().data_type(), MPSDataType::Float32);
    let silu_method = a.silu(None);
    assert_eq!(silu_method.inner().data_type(), MPSDataType::Float32);

    // Test gelu
    let gelu_a = gelu(&a, None);
    assert_eq!(gelu_a.inner().data_type(), MPSDataType::Float32);
    let gelu_method = a.gelu(None);
    assert_eq!(gelu_method.inner().data_type(), MPSDataType::Float32);
}

#[test]
fn test_binary_operations() {
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[2usize, 3usize]);
    let a = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("a"));

    // Test power operation
    let exponent = Tensor::new(graph.constant_scalar(2.0f32, MPSDataType::Float32));
    let powered = pow(&a, &exponent, None);
    assert_eq!(powered.inner().data_type(), MPSDataType::Float32);
    let powered_method = a.pow(&exponent, None);
    assert_eq!(powered_method.inner().data_type(), MPSDataType::Float32);

    // Test clip operation
    let min_val = Tensor::new(graph.constant_scalar(0.0f32, MPSDataType::Float32));
    let max_val = Tensor::new(graph.constant_scalar(1.0f32, MPSDataType::Float32));
    let clipped = clip(&a, &min_val, &max_val, None);
    assert_eq!(clipped.inner().data_type(), MPSDataType::Float32);
    let clipped_method = a.clip(&min_val, &max_val, None);
    assert_eq!(clipped_method.inner().data_type(), MPSDataType::Float32);
}

#[test]
fn test_tensor_conversion() {
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[2usize, 3usize]);

    // Test MPSGraphTensor to Tensor conversion (From trait)
    let tensor_obj = graph.placeholder(&shape, MPSDataType::Float32, None);
    let tensor = Tensor::from(tensor_obj.clone());
    assert_eq!(tensor.inner().data_type(), MPSDataType::Float32);

    // Test Tensor to MPSGraphTensor conversion (into() from From trait)
    let tensor_back: MPSGraphTensor = tensor.clone().into();
    assert_eq!(tensor_back.data_type(), MPSDataType::Float32);

    // Test unwrap method
    let tensor_unwrapped = tensor.unwrap();
    assert_eq!(tensor_unwrapped.data_type(), MPSDataType::Float32);
}

#[test]
fn test_const_scalar() {
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[2usize, 3usize]);
    let tensor = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("a"));

    // Test creating a constant scalar with matching data type
    let half = tensor.const_scalar(0.5f32);
    assert_eq!(half.inner().data_type(), MPSDataType::Float32);

    // Test using the constant in an operation
    let scaled = &tensor * &half;
    assert_eq!(scaled.inner().data_type(), MPSDataType::Float32);
}

#[test]
fn test_complex_expressions() {
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[2usize, 3usize]);
    let a = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("a"));
    let b = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("b"));

    // Test nested function calls
    let nested = abs(&sqrt(&(&a + &b), None), None);
    assert_eq!(nested.inner().data_type(), MPSDataType::Float32);

    // Test mixed operation styles
    let square_a = a.square(None);
    let diff_abs = abs(&(&a - &b), None);
    let mixed = &square_a * &diff_abs;
    assert_eq!(mixed.inner().data_type(), MPSDataType::Float32);

    // Test complex activation functions
    let custom_activation = &(relu(&a, None)) + &tanh(&b, None);
    assert_eq!(custom_activation.inner().data_type(), MPSDataType::Float32);
}

#[test]
fn test_random_tensors() {
    let graph = MPSGraph::new();

    // Test random uniform
    let random_uniform = graph.create_random_uniform(0.0f32, 1.0f32, &[2, 3], MPSDataType::Float32);
    assert_eq!(random_uniform.inner().data_type(), MPSDataType::Float32);

    // Test random normal
    let random_normal = graph.create_random_normal(0.0f32, 1.0f32, &[2, 3], MPSDataType::Float32);
    assert_eq!(random_normal.inner().data_type(), MPSDataType::Float32);
}
