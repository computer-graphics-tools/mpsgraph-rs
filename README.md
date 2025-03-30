# mpsgraph-rs

Rust bindings for Apple's Metal Performance Shaders Graph (MPSGraph) API.

## Workspace Structure

This repository is organized as a Rust workspace with the following crates:

- **mpsgraph**: Core bindings for MPSGraph API
- **mpsgraph-tools**: High-level utilities and ergonomic tensor operations API

## Usage

To use the core MPSGraph bindings:

```toml
[dependencies]
mpsgraph = { path = "path/to/mpsgraph-rs/crates/mpsgraph-rs" }
```

To use the high-level tensor operations API:

```toml
[dependencies]
mpsgraph = { path = "path/to/mpsgraph-rs/crates/mpsgraph-rs" }
mpsgraph-tools = { path = "path/to/mpsgraph-rs/crates/mpsgraph-tools-rs" }
```

## Examples

### Core MPSGraph Examples

Run an example from the core mpsgraph crate:

```bash
cargo run -p mpsgraph --example simple_compile
```

Available examples:
- matmul: Matrix multiplication using MPSGraph
- simple_compile: Simple graph compilation and execution
- type_test: Test of data type conversions
- callback_test: Testing callback functionality

### Tensor Operations Examples

Run an example from the mpsgraph-tools crate:

```bash
cargo run -p mpsgraph-tools --example tensor_ops
```

This example demonstrates:
- Operator overloading for tensor arithmetic
- Functional-style tensor operations
- Activation functions and other neural network operations
- Tensor creation utilities

## Features

### mpsgraph

- **link**: Links against MetalPerformanceShadersGraph.framework (enabled by default)

## Building

```bash
# Build all crates
cargo build

# Build and run examples
cargo run -p mpsgraph --example simple_compile
cargo run -p mpsgraph-tools --example tensor_ops

# Run tests
cargo test -p mpsgraph
cargo test -p mpsgraph-tools
```

## Requirements

- macOS 13.0 or later
- Xcode 14.0 or later
- Metal-supporting GPU

## License

Licensed under MIT license.