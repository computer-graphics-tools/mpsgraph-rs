use crate::{
    core::MPSDataType, graph::MPSGraph, shape::MPSShape, tensor::MPSGraphTensor,
    tensor_data::MPSGraphTensorData,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

// These tests ensure that memory management works correctly for Objective-C objects

#[test]
fn test_graph_cloning_and_dropping() {
    // Create a new graph
    let graph = MPSGraph::new();

    // Clone it multiple times
    let clones: Vec<_> = (0..10).map(|_| graph.clone()).collect();

    // Add operations to each clone
    for (i, clone) in clones.iter().enumerate() {
        let shape = MPSShape::from_slice(&[2, 2]);
        let _tensor =
            clone.placeholder(&shape, MPSDataType::Float32, Some(&format!("tensor_{}", i)));
    }

    // Drop the clones one by one
    for clone in clones {
        drop(clone);
    }

    // Original should still be valid
    let shape = MPSShape::from_slice(&[2, 2]);
    let _tensor = graph.placeholder(&shape, MPSDataType::Float32, Some("final_tensor"));
}

#[test]
fn test_tensor_cloning_and_dropping() {
    let graph = MPSGraph::new();

    // Create a tensor
    let shape = MPSShape::from_slice(&[2, 2]);
    let tensor = graph.placeholder(&shape, MPSDataType::Float32, Some("original"));

    // Clone it multiple times
    let clones: Vec<_> = (0..10).map(|_| tensor.clone()).collect();

    // Use each clone
    for clone in &clones {
        let _name = clone.name();
        let _dtype = clone.data_type();
    }

    // Drop the clones one by one
    for clone in clones {
        drop(clone);
    }

    // Original should still be valid
    assert_eq!(tensor.name(), "original");
}

#[test]
fn test_tensor_data_cloning_and_dropping() {
    // Create tensor data
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let shape = [2, 2];
    let tensor_data = MPSGraphTensorData::new(&data, &shape, MPSDataType::Float32);

    // Clone it multiple times
    let clones: Vec<_> = (0..10).map(|_| tensor_data.clone()).collect();

    // Use each clone
    for clone in &clones {
        let _shape = clone.shape();
        let _dtype = clone.data_type();
    }

    // Drop the clones one by one
    for clone in clones {
        drop(clone);
    }

    // Original should still be valid
    let _shape = tensor_data.shape();
    let _dtype = tensor_data.data_type();
}

#[test]
fn test_multithreaded_tensor_access() {
    let graph = Arc::new(MPSGraph::new());

    // Create a shared tensor
    let shape = MPSShape::from_slice(&[2, 2]);
    let tensor = Arc::new(Mutex::new(graph.placeholder(
        &shape,
        MPSDataType::Float32,
        Some("shared"),
    )));

    // Spawn multiple threads accessing the tensor
    let handles: Vec<_> = (0..5)
        .map(|_| {
            let graph_clone = Arc::clone(&graph);
            let tensor_clone = Arc::clone(&tensor);

            thread::spawn(move || {
                // Access the tensor
                let t = tensor_clone.lock().unwrap();
                let name = t.name();
                let dtype = t.data_type();

                // Create a new tensor in this thread
                let shape = MPSShape::from_slice(&[3, 3]);
                let _new_tensor = graph_clone.placeholder(&shape, MPSDataType::Float32, None);

                (name, dtype)
            })
        })
        .collect();

    // Join all threads
    for handle in handles {
        let (name, dtype) = handle.join().unwrap();
        assert_eq!(name, "shared");
        assert_eq!(dtype, MPSDataType::Float32);
    }
}

#[test]
fn test_large_number_of_tensors() {
    let graph = MPSGraph::new();

    // Create and drop a large number of tensors
    let shape = MPSShape::from_slice(&[2, 2]);

    // Using a scope to control lifetimes
    {
        // Create 100 tensors
        let tensors: Vec<_> = (0..100)
            .map(|i| {
                graph.placeholder(&shape, MPSDataType::Float32, Some(&format!("tensor_{}", i)))
            })
            .collect();

        // Create some operations between them
        let mut results = Vec::new();
        for i in 0..99 {
            let result = graph.add(&tensors[i], &tensors[i + 1], None);
            results.push(result);
        }

        // Verify a few tensors
        assert_eq!(tensors[0].name(), "tensor_0");
        assert_eq!(tensors[50].name(), "tensor_50");
        assert_eq!(tensors[99].name(), "tensor_99");

        // Let them all drop at the end of this scope
    }

    // Graph should still be valid
    let new_tensor = graph.placeholder(&shape, MPSDataType::Float32, Some("after_drop"));
    assert_eq!(new_tensor.name(), "after_drop");
}

#[test]
fn test_retain_release_cycle() {
    let graph = MPSGraph::new();

    // Create tensors that refer to each other
    let shape = MPSShape::from_slice(&[2, 2]);
    let a = graph.placeholder(&shape, MPSDataType::Float32, Some("A"));
    let b = graph.placeholder(&shape, MPSDataType::Float32, Some("B"));

    // Create operations that connect them
    let a_plus_b = graph.add(&a, &b, Some("A+B"));

    // Create a HashMap that stores the tensors
    let mut tensors: HashMap<String, MPSGraphTensor> = HashMap::new();
    tensors.insert("A".to_string(), a.clone());
    tensors.insert("B".to_string(), b.clone());
    tensors.insert("A+B".to_string(), a_plus_b.clone());

    // Add and remove tensors from the HashMap
    for i in 0..10 {
        let temp = graph.placeholder(&shape, MPSDataType::Float32, Some(&format!("temp_{}", i)));
        let key = format!("temp_{}", i);
        tensors.insert(key.clone(), temp.clone());

        if i % 2 == 0 {
            tensors.remove(&key);
        }
    }

    // Original tensors should still be valid
    assert_eq!(a.name(), "A");
    assert_eq!(b.name(), "B");
    assert_eq!(a_plus_b.name(), "A+B");

    // Create some more operations
    let c = graph.multiply(&a, &b, Some("A*B"));
    tensors.insert("A*B".to_string(), c.clone());

    // Drop the HashMap
    drop(tensors);

    // Original tensors should still be valid
    assert_eq!(a.name(), "A");
    assert_eq!(b.name(), "B");
    assert_eq!(a_plus_b.name(), "A+B");
    assert_eq!(c.name(), "A*B");
}
