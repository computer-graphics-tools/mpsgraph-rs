use mpsgraph::{MPSGraph, MPSGraphTensorData, MPSDataType, MPSShape};

fn main() {
    // Create a new graph
    let graph = MPSGraph::new();
    
    // Create a 3x3 input tensor with values 1-9
    let shape = MPSShape::from_slice(&[3, 3]);
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let input_tensor = graph.placeholder(&shape, MPSDataType::Float32, None);
    
    // Create a 1x3 tensor for matrix multiplication
    let lhs_shape = MPSShape::from_slice(&[1, 3]);
    let lhs_data = vec![1.0f32, 2.0, 3.0];
    let lhs_tensor = graph.placeholder(&lhs_shape, MPSDataType::Float32, None);
    
    // 1. Matrix multiplication
    let matmul_tensor = graph.matmul(&lhs_tensor, &input_tensor, Some("matmul"));
    
    // 2. Band part operation (keep only main diagonal)
    let band_part_tensor = graph.band_part(&input_tensor, 0, 0, Some("band_part"));
    
    // 3. Matrix inverse
    let invertible_tensor = graph.placeholder(&shape, MPSDataType::Float32, None);
    let invertible_data = vec![
        4.0f32, 7.0, 1.0,
        2.0, 6.0, 5.0,
        3.0, 4.0, 8.0
    ];
    let inverse_tensor = graph.matrix_inverse(&invertible_tensor, Some("inverse"));
    
    // Create feed dictionary for inputs - need to use references to the tensors
    let mut feed_dict = std::collections::HashMap::new();
    feed_dict.insert(&input_tensor, MPSGraphTensorData::new(&input_data, &[3, 3], MPSDataType::Float32));
    feed_dict.insert(&lhs_tensor, MPSGraphTensorData::new(&lhs_data, &[1, 3], MPSDataType::Float32));
    feed_dict.insert(&invertible_tensor, MPSGraphTensorData::new(&invertible_data, &[3, 3], MPSDataType::Float32));
    
    // Run the graph - need to use references to the tensors
    let results = graph.run(feed_dict, &[&matmul_tensor, &band_part_tensor, &inverse_tensor]);
    
    // Extract and print results
    println!("Input Matrix:");
    print_matrix(&input_data, 3, 3);
    
    println!("\nMatrix Multiplication Result:");
    let matmul_result = results[&matmul_tensor].to_vec::<f32>();
    print_matrix(&matmul_result, 1, 3);
    
    println!("\nBand Part Result (main diagonal only):");
    let band_part_result = results[&band_part_tensor].to_vec::<f32>();
    print_matrix(&band_part_result, 3, 3);
    
    println!("\nOriginal Matrix for Inverse:");
    print_matrix(&invertible_data, 3, 3);
    
    println!("\nMatrix Inverse Result:");
    let inverse_result = results[&inverse_tensor].to_vec::<f32>();
    print_matrix(&inverse_result, 3, 3);
    
    // Verify the inverse by multiplying original * inverse = identity
    let verification_tensor = graph.matmul(&invertible_tensor, &inverse_tensor, Some("verification"));
    let mut verification_feed_dict = std::collections::HashMap::new();
    verification_feed_dict.insert(&invertible_tensor, MPSGraphTensorData::new(&invertible_data, &[3, 3], MPSDataType::Float32));
    verification_feed_dict.insert(&inverse_tensor, results[&inverse_tensor].clone());
    let verification_results = graph.run(verification_feed_dict, &[&verification_tensor]);
    let verification_data = verification_results[&verification_tensor].to_vec::<f32>();
    
    println!("\nVerification (Original * Inverse ≈ Identity):");
    print_matrix(&verification_data, 3, 3);
}

fn print_matrix(data: &[f32], rows: usize, cols: usize) {
    for i in 0..rows {
        print!("[");
        for j in 0..cols {
            let val = data[i * cols + j];
            // Format to 4 decimal places for clarity
            if j < cols - 1 {
                print!("{:.4}, ", val);
            } else {
                print!("{:.4}", val);
            }
        }
        println!("]");
    }
}