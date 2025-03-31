use crate::{core::MPSDataType, shape::MPSShape, tensor_data::MPSGraphTensorData};

#[test]
fn test_tensor_data_creation() {
    // Create tensor data with different types and shapes

    // Float32
    let data_f32 = [1.0f32, 2.0, 3.0, 4.0];
    let shape_f32 = [2, 2];
    let tensor_data_f32 = MPSGraphTensorData::new(&data_f32, &shape_f32, MPSDataType::Float32);

    // Int32
    let data_i32 = [5i32, 6, 7, 8];
    let shape_i32 = [2, 2];
    let tensor_data_i32 = MPSGraphTensorData::new(&data_i32, &shape_i32, MPSDataType::Int32);

    // Float16 (test with f32 data, internally converted)
    let data_f16 = [0.5f32, 1.5, 2.5, 3.5];
    let shape_f16 = [4];
    let tensor_data_f16 = MPSGraphTensorData::new(&data_f16, &shape_f16, MPSDataType::Float16);

    // Int8
    let data_i8 = [1i8, 2, 3, 4];
    let shape_i8 = [2, 2];
    let tensor_data_i8 = MPSGraphTensorData::new(&data_i8, &shape_i8, MPSDataType::Int8);

    // Verify properties
    assert_eq!(tensor_data_f32.data_type(), MPSDataType::Float32);
    assert_eq!(tensor_data_i32.data_type(), MPSDataType::Int32);
    assert_eq!(tensor_data_f16.data_type(), MPSDataType::Float16);
    assert_eq!(tensor_data_i8.data_type(), MPSDataType::Int8);

    // Check shape
    let shape_obj_f32 = tensor_data_f32.shape();
    assert_eq!(shape_obj_f32.dimensions(), vec![2, 2]);

    let shape_obj_f16 = tensor_data_f16.shape();
    assert_eq!(shape_obj_f16.dimensions(), vec![4]);
}

#[test]
fn test_tensor_data_metal_device() {
    // Skip if Metal device not available
    if let Some(metal_device) = metal::Device::system_default() {
        // Create a Metal buffer with data
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let size = std::mem::size_of_val(&data) as u64;
        let buffer = metal_device.new_buffer_with_data(
            data.as_ptr() as *const _,
            size,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create tensor data from Metal buffer
        let shape = MPSShape::from_slice(&[2, 2]);
        let tensor_data = MPSGraphTensorData::from_buffer(&buffer, &shape, MPSDataType::Float32);

        // Verify properties
        assert_eq!(tensor_data.data_type(), MPSDataType::Float32);

        let shape_obj = tensor_data.shape();
        assert_eq!(shape_obj.dimensions(), vec![2, 2]);
    }
}

#[test]
fn test_tensor_data_clone() {
    // Create tensor data
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let shape = [2, 2];
    let tensor_data = MPSGraphTensorData::new(&data, &shape, MPSDataType::Float32);

    // Clone tensor data
    let cloned_data = tensor_data.clone();

    // Verify properties of clone match original
    assert_eq!(cloned_data.data_type(), tensor_data.data_type());

    let original_shape = tensor_data.shape();
    let cloned_shape = cloned_data.shape();
    assert_eq!(cloned_shape.dimensions(), original_shape.dimensions());
}

#[test]
fn test_tensor_data_different_shapes() {
    // Test various shape configurations

    // 1D shape
    let data_1d = [1.0f32, 2.0, 3.0, 4.0];
    let shape_1d = [4];
    let tensor_data_1d = MPSGraphTensorData::new(&data_1d, &shape_1d, MPSDataType::Float32);

    let shape_obj_1d = tensor_data_1d.shape();
    assert_eq!(shape_obj_1d.dimensions(), vec![4]);

    // 3D shape
    let data_3d = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let shape_3d = [2, 2, 2];
    let tensor_data_3d = MPSGraphTensorData::new(&data_3d, &shape_3d, MPSDataType::Float32);

    let shape_obj_3d = tensor_data_3d.shape();
    assert_eq!(shape_obj_3d.dimensions(), vec![2, 2, 2]);

    // 4D shape
    let mut data_4d = Vec::new();
    for i in 0..16 {
        data_4d.push(i as f32);
    }
    let shape_4d = [2, 2, 2, 2];
    let tensor_data_4d = MPSGraphTensorData::new(&data_4d, &shape_4d, MPSDataType::Float32);

    let shape_obj_4d = tensor_data_4d.shape();
    assert_eq!(shape_obj_4d.dimensions(), vec![2, 2, 2, 2]);
}
