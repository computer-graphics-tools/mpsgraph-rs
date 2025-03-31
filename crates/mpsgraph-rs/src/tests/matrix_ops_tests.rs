use crate::{
    core::MPSDataType, graph::MPSGraph, shape::MPSShape, tensor::MPSGraphTensor,
    tensor_data::MPSGraphTensorData,
};
use std::collections::HashMap;

// Simple verification helper that checks if result exists
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
fn test_matrix_multiplication() {
    let graph = MPSGraph::new();

    // Create input matrices
    // Matrix A: 2x3
    let shape_a = MPSShape::from_slice(&[2, 3]);
    let a = graph.placeholder(&shape_a, MPSDataType::Float32, Some("A"));

    // Matrix B: 3x2
    let shape_b = MPSShape::from_slice(&[3, 2]);
    let b = graph.placeholder(&shape_b, MPSDataType::Float32, Some("B"));

    // Define matrix multiplication operation: C = A * B
    let matmul_result = graph.matmul(&a, &b, Some("MatMul"));

    // Create input data
    // A: [1, 2, 3]
    //    [4, 5, 6]
    let a_data = MPSGraphTensorData::new(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        MPSDataType::Float32,
    );

    // B: [7, 8]
    //    [9, 10]
    //    [11, 12]
    let b_data = MPSGraphTensorData::new(
        &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0],
        &[3, 2],
        MPSDataType::Float32,
    );

    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(a.clone(), a_data);
    feeds.insert(b.clone(), b_data);

    // Run graph
    let results = graph.run_with_feeds(&feeds, &[matmul_result.clone()]);

    // Verify result exists
    verify_result_exists(&results, &matmul_result, "matrix multiplication");
}

#[test]
fn test_transpose() {
    let graph = MPSGraph::new();

    // Create input matrix
    // Matrix A: 2x3
    let shape_a = MPSShape::from_slice(&[2, 3]);
    let a = graph.placeholder(&shape_a, MPSDataType::Float32, Some("A"));

    // Define transpose operation: B = A^T
    // Permutation [1, 0] means swap dimensions 0 and 1
    let perm_data = [1usize, 0];
    let transpose_result = graph.transpose(&a, &perm_data, Some("Transpose"));

    // Create input data
    // A: [1, 2, 3]
    //    [4, 5, 6]
    let a_data = MPSGraphTensorData::new(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        MPSDataType::Float32,
    );

    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(a.clone(), a_data);

    // Run graph
    let results = graph.run_with_feeds(&feeds, &[transpose_result.clone()]);

    // Verify result exists
    verify_result_exists(&results, &transpose_result, "transpose");
}

#[test]
fn test_batch_matrix_multiplication() {
    let graph = MPSGraph::new();

    // Create input tensors
    // Tensor A: 2x2x3 (batch of 2 matrices, each 2x3)
    let shape_a = MPSShape::from_slice(&[2, 2, 3]);
    let a = graph.placeholder(&shape_a, MPSDataType::Float32, Some("A"));

    // Tensor B: 2x3x2 (batch of 2 matrices, each 3x2)
    let shape_b = MPSShape::from_slice(&[2, 3, 2]);
    let b = graph.placeholder(&shape_b, MPSDataType::Float32, Some("B"));

    // Define batch matrix multiplication operation: C = A * B
    let batch_matmul_result = graph.batch_matmul(&a, &b, Some("BatchMatMul"));

    // Create input data
    // Batch 1: A1 = [1, 2, 3, 4, 5, 6], B1 = [7, 8, 9, 10, 11, 12]
    // Batch 2: A2 = [13, 14, 15, 16, 17, 18], B2 = [19, 20, 21, 22, 23, 24]
    let a_data = MPSGraphTensorData::new(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        &[2, 2, 3],
        MPSDataType::Float32,
    );

    let b_data = MPSGraphTensorData::new(
        &[
            7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ],
        &[2, 3, 2],
        MPSDataType::Float32,
    );

    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(a.clone(), a_data);
    feeds.insert(b.clone(), b_data);

    // Run graph
    let results = graph.run_with_feeds(&feeds, &[batch_matmul_result.clone()]);

    // Verify result exists
    verify_result_exists(
        &results,
        &batch_matmul_result,
        "batch matrix multiplication",
    );
}

#[test]
fn test_reshape_and_matmul() {
    let graph = MPSGraph::new();

    // Create input tensor: 6 elements
    let shape_in = MPSShape::from_slice(&[6]);
    let input = graph.placeholder(&shape_in, MPSDataType::Float32, Some("Input"));

    // Reshape to 2x3 matrix - use i64 array since reshape takes &[i64]
    let shape_dims = &[2i64, 3];
    let a = graph.reshape(&input, shape_dims, Some("A"));

    // Create weight matrix: 3x2
    let shape_b = MPSShape::from_slice(&[3, 2]);
    let b_data = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let b = graph.constant(&b_data, &shape_b, MPSDataType::Float32);

    // Define matrix multiplication: C = A * B
    let matmul_result = graph.matmul(&a, &b, Some("MatMul"));

    // Create input data
    let input_data = MPSGraphTensorData::new(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[6],
        MPSDataType::Float32,
    );

    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(input.clone(), input_data);

    // Run graph
    let results = graph.run_with_feeds(&feeds, &[matmul_result.clone()]);

    // Verify result exists
    verify_result_exists(&results, &matmul_result, "reshaped matrix multiplication");
}
