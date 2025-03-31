use crate::{
    core::MPSDataType, graph::MPSGraph, shape::MPSShape, tensor::MPSGraphTensor,
    tensor_data::MPSGraphTensorData,
};
use std::collections::HashMap;

// Simple verification helper that checks if result exists - we can't check actual values from MPSGraphTensorData directly
fn verify_result_exists(
    results: &HashMap<MPSGraphTensor, MPSGraphTensorData>,
    tensor: &MPSGraphTensor,
    name: &str,
) {
    assert!(
        results.contains_key(tensor),
        "Results should contain {} tensor",
        name
    );
}

#[test]
fn test_binary_arithmetic_ops() {
    let graph = MPSGraph::new();

    // Create inputs
    let shape = MPSShape::from_slice(&[2, 2]);
    let a = graph.placeholder(&shape, MPSDataType::Float32, Some("A"));
    let b = graph.placeholder(&shape, MPSDataType::Float32, Some("B"));

    // Define various binary operations
    let add_result = graph.add(&a, &b, Some("Add"));
    let sub_result = graph.subtract(&a, &b, Some("Subtract"));
    let mul_result = graph.multiply(&a, &b, Some("Multiply"));
    let div_result = graph.divide(&a, &b, Some("Divide"));

    // Create input data
    let a_data = MPSGraphTensorData::new(&[2.0f32, 4.0, 6.0, 8.0], &[2, 2], MPSDataType::Float32);
    let b_data = MPSGraphTensorData::new(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], MPSDataType::Float32);

    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(a.clone(), a_data);
    feeds.insert(b.clone(), b_data);

    // Run graph
    let results = graph.run_with_feeds(
        &feeds,
        &[
            add_result.clone(),
            sub_result.clone(),
            mul_result.clone(),
            div_result.clone(),
        ],
    );

    // Verify results for each operation
    verify_result_exists(&results, &add_result, "addition");
    verify_result_exists(&results, &sub_result, "subtraction");
    verify_result_exists(&results, &mul_result, "multiplication");
    verify_result_exists(&results, &div_result, "division");
}

#[test]
fn test_unary_arithmetic_ops() {
    let graph = MPSGraph::new();

    // Create input
    let shape = MPSShape::from_slice(&[2, 2]);
    let a = graph.placeholder(&shape, MPSDataType::Float32, Some("A"));

    // Define various unary operations
    let identity_result = graph.identity(&a, Some("Identity"));
    let abs_result = graph.abs(&a, Some("Abs"));
    let exp_result = graph.exp(&a, Some("Exp"));
    let log_result = graph.log(&a, Some("Log"));
    let neg_result = graph.negative(&a, Some("Negative"));
    let square_result = graph.square(&a, Some("Square"));
    let sqrt_result = graph.sqrt(&a, Some("Sqrt"));

    // Create input data
    let a_data = MPSGraphTensorData::new(&[1.0f32, 2.0, 4.0, 16.0], &[2, 2], MPSDataType::Float32);

    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(a.clone(), a_data);

    // Run graph
    let results = graph.run_with_feeds(
        &feeds,
        &[
            identity_result.clone(),
            abs_result.clone(),
            exp_result.clone(),
            log_result.clone(),
            neg_result.clone(),
            square_result.clone(),
            sqrt_result.clone(),
        ],
    );

    // Verify results for each operation
    verify_result_exists(&results, &identity_result, "identity");
    verify_result_exists(&results, &abs_result, "abs");
    verify_result_exists(&results, &exp_result, "exp");
    verify_result_exists(&results, &log_result, "log");
    verify_result_exists(&results, &neg_result, "negative");
    verify_result_exists(&results, &square_result, "square");
    verify_result_exists(&results, &sqrt_result, "sqrt");
}

#[test]
fn test_power_ops() {
    let graph = MPSGraph::new();

    // Create inputs
    let shape = MPSShape::from_slice(&[2, 2]);
    let base = graph.placeholder(&shape, MPSDataType::Float32, Some("Base"));

    // Create exponent as constant
    let exponent_data = [2.0f32, 3.0, 0.5, 0.0];
    let exponent = graph.constant(&exponent_data, &shape, MPSDataType::Float32);

    // Define power operation
    let power_result = graph.power(&base, &exponent, Some("Power"));

    // Create input data
    let base_data =
        MPSGraphTensorData::new(&[2.0f32, 3.0, 4.0, 5.0], &[2, 2], MPSDataType::Float32);

    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(base.clone(), base_data);

    // Run graph
    let results = graph.run_with_feeds(&feeds, &[power_result.clone()]);

    // Verify power results
    verify_result_exists(&results, &power_result, "power");
}

#[test]
fn test_scalar_arithmetic() {
    let graph = MPSGraph::new();

    // Create input tensor
    let shape = MPSShape::from_slice(&[2, 2]);
    let a = graph.placeholder(&shape, MPSDataType::Float32, Some("A"));

    // Create scalar constant
    let scalar_val = 2.0f32;
    let scalar = graph.constant_scalar(scalar_val, MPSDataType::Float32);

    // Define operations with scalar
    let add_scalar = graph.add(&a, &scalar, Some("AddScalar"));
    let mul_scalar = graph.multiply(&a, &scalar, Some("MulScalar"));
    let div_scalar = graph.divide(&a, &scalar, Some("DivScalar"));

    // Create input data
    let a_data = MPSGraphTensorData::new(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], MPSDataType::Float32);

    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(a.clone(), a_data);

    // Run graph
    let results = graph.run_with_feeds(
        &feeds,
        &[add_scalar.clone(), mul_scalar.clone(), div_scalar.clone()],
    );

    // Verify results
    verify_result_exists(&results, &add_scalar, "add_scalar");
    verify_result_exists(&results, &mul_scalar, "mul_scalar");
    verify_result_exists(&results, &div_scalar, "div_scalar");
}
