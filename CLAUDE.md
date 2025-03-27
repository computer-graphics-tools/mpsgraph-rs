# CLAUDE.md - mpsgraph-rs Development Guide

## Build Commands

- Build: `cargo build`
- Run tests: `cargo test`
- Run single test: `cargo test test_name`
- Run with features: `cargo build --features=link`
- Check compilation: `cargo check`
- Format code: `cargo fmt`
- Lint code: `cargo clippy`

## Code Style Guidelines

- **Formatting**: Follow Rust standard style (rustfmt)
- **Imports**: Group imports by external crates, then standard library
- **Error Handling**: Use `Option<T>` for nullable values
- **Naming**:
  - Use snake_case for variables and functions
  - Use CamelCase for types and structs
- **Safety**: Use `unsafe` blocks only when necessary for FFI
- **Types**: Prefer strong typing with dedicated wrapper types
- **Descriptors**: Implement as structs
- **Memory Management**: Ensure proper resource cleanup with Drop implementations
- **Comments**: Document public APIs with doc comments