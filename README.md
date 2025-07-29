# mpsgraph-rs

Modern Rust bindings for Apple's Metal Performance Shaders Graph framework (MPSGraph).

## Overview

This crate provides Rust bindings for the Metal Performance Shaders Graph framework, a high-level graph-based API for defining neural network models that run on the GPU. It uses:

- More idiomatic Rust API with improved type safety
- Cleaner FFI boundary between Rust and Objective-C
- Automatic memory management with `Retained<T>` instead of manual retain/release

## Requirements

- macOS 10.15 or newer
- Rust 1.56 or newer
- Metal-compatible GPU

## Usage Example

```rust
use mpsgraph::{DataType, Device, Graph, ShapedType, TensorData};
use metal::{self as mtl, MTLResourceOptions};
use std::collections::HashMap;

fn main() {
    let mtl_device = MTLDevice::system_default().expect("No Metal device found");
    let device = Device::with_device(&mtl_device);
    let graph = Graph::new();

    let a = graph.placeholder(Some(&[2_isize, 2]), DataType::Float32, None);
    let b = graph.placeholder(Some(&[2_isize, 2]), DataType::Float32, None);
    let c = graph.matrix_multiplication(&a, &b, None);

    let a_values: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let b_values: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
    let bytes = (a_values.len() * std::mem::size_of::<f32>()) as u64;

    let a_buffer = mtl_device.new_buffer_with_data(
        a_values.as_ptr() as *const std::ffi::c_void,
        bytes,
        MTLResourceOptions::empty(),
    );
    let b_buffer = mtl_device.new_buffer_with_data(
        b_values.as_ptr() as *const std::ffi::c_void,
        bytes,
        MTLResourceOptions::empty(),
    );
    let c_buffer = mtl_device.new_buffer(bytes, MTLResourceOptions::empty());

    let a_td = TensorData::new_with_mtl_buffer(&a_buffer, &[2, 2], DataType::Float32, None);
    let b_td = TensorData::new_with_mtl_buffer(&b_buffer, &[2, 2], DataType::Float32, None);
    let c_td = TensorData::new_with_mtl_buffer(&c_buffer, &[2, 2], DataType::Float32, None);

    let mut feeds = HashMap::new();
    let shaped = ShapedType::new_with_shape_data_type(Some(&[2_isize, 2]), DataType::Float32);
    feeds.insert(&*a, &*shaped);
    feeds.insert(&*b, &*shaped);

    let executable = graph.compile(&device, &feeds, &[&*c], None, None);
    let mtl_command_queue = mtl_device.new_command_queue();
    let mps_command_buffer = CommandBuffer::from_command_queue(&mtl_command_queue);

    executable.encode_to_command_buffer(
        &mps_command_buffer,
        &[&*a_td, &*b_td],
        Some(&[&*c_td]),
        None,
    );

    mps_command_buffer.commit();
    mps_command_buffer
        .root_command_buffer()
        .wait_until_completed();

    let result = unsafe { std::slice::from_raw_parts(c_buffer.contents() as *const f32, 4) };
    println!("Result matrix: {:?}", result); // [19.0, 22.0, 43.0, 50.0]
}
```

## License

Licensed under [MIT license](LICENSE).
