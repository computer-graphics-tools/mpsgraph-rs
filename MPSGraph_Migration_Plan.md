# MPSGraph Migration Plan: objc-rs to objc2

## Overview

This document outlines the plan for migrating the mpsgraph-rs project from the deprecated objc-rs crate to the newer, safer objc2 crate and its companion objc2-foundation crate.

## Current Status

The initial work has begun, with core files (core.rs, graph.rs, tensor_data.rs) being migrated to use objc2 instead of objc-rs. However, several issues need to be addressed before the migration is complete. The project currently does not compile due to numerous errors related to the migration.

## Migration Issues

1. **Unresolved Imports**: Most operation files still use the old `objc::runtime::Object` and `objc::msg_send` imports.
2. **RefEncode for Metal Types**: The `MTLCommandQueue` type doesn't implement the `RefEncode` trait needed for objc2's message passing.
3. **UnwindSafe Issues**: `std::panic::catch_unwind` in graph.rs requires `UnwindSafe`, which isn't implemented for objc2 objects.
4. **NSArray Construction**: Several issues with creating NSArray objects using objc2-foundation's API.
5. **Memory Management**: Multiple issues with proper retain/release patterns.

## Implementation Plan

### Phase 1: Update Core Imports in All Files

1. Replace all instances of:
   - `use objc::runtime::Object;` → `use objc2::runtime::Object;` 
   - `use objc::msg_send;` → `use objc2::msg_send;`
   - `use objc::runtime::{Class, YES, NO};` → `use objc2::runtime::{Class, true as YES, false as NO};`

2. Update class resolution:
   - Replace `Class::get("ClassName")` → `Class::get(std::ffi::CStr::from_bytes_with_nul(b"ClassName\0").unwrap())`

### Phase 2: Fix Memory Management

1. Replace manual retain/release calls:
   - `msg_send![obj, retain]` → `objc2::ffi::objc_retain(obj as *mut _)`
   - `msg_send![obj, release]` → `objc2::ffi::objc_release(obj as *mut _)`

2. Update Foundation object creation to use Retained:
   - Use `NSString::from_str()` instead of manual string creation
   - Use `NSArray::from_slice()` with proper reference slice
   - Use `NSDictionary::from_slices()` with keys and values

### Phase 3: Address Metal Type Issues

1. Handle `MTLCommandQueue` and other Metal types properly:
   - Cast Metal objects to void pointers before passing to objc2 message sends
   - Create proper wrappers for Metal types when needed
   - Implement proper conversion functions for Metal types

2. Fix the `RefEncode` error:
   - Create proper wrappers or conversion functions for Metal types
   - Ensure proper type casting for Metal objects

### Phase 4: Fix catch_unwind Issue

1. Restructure code to avoid using catch_unwind with objc2 types:
   - Use alternative error handling mechanisms
   - Move critical sections to separate functions that don't require UnwindSafe
   - Implement proper error handling without relying on catch_unwind

### Phase 5: Fix Foundation Type Handling

1. Correct NSArray creation:
   - Ensure proper slice type is passed to from_slice
   - Create temporary references when needed
   - Properly handle retained/unretained object patterns

2. Update NSDictionary creation:
   - Use from_slices with properly typed key and value slices
   - Handle dictionary access properly

### Phase 6: Systematic Testing

1. Create thorough tests for each component:
   - Basic graph operations
   - Memory management
   - Foundation object creation and handling
   - Metal integration

2. Ensure all examples work correctly:
   - Fix debugging examples
   - Ensure test utilities work properly

## Detailed Migration Rules

### String Handling
```rust
// Old (objc-rs)
let ns_string = NSString::alloc().init_str("hello");
let _: () = msg_send![ns_string, release];

// New (objc2)
let ns_string = NSString::from_str("hello"); // Returns Retained<NSString>
// No manual release needed
```

### Array Handling
```rust
// Old (objc-rs)
let array = NSArray::alloc().init();
let _: () = msg_send![array, release];

// New (objc2)
let array = NSArray::<NSObject>::new(); // Returns Retained<NSArray<NSObject>>
// No manual release needed
```

### Dictionary Handling
```rust
// Old (objc-rs)
let dict = NSDictionary::alloc().init();
let _: () = msg_send![dict, release];

// New (objc2)
let dict = NSDictionary::<NSString, NSObject>::new(); // Returns Retained<NSDictionary>
// No manual release needed
```

### Metal Type Casting
```rust
// Old (objc-rs)
let metal_device: *mut Object = msg_send![class, device];

// New (objc2)
let metal_device: *mut Object = unsafe { 
    msg_send![class!(b"MTLDevice\0"), device] 
};
let metal_device_ptr = metal_device as *mut c_void;
```

### Class Resolution
```rust
// Old (objc-rs)
let cls = Class::get("MPSGraph").unwrap();

// New (objc2)
let class_name = std::ffi::CStr::from_bytes_with_nul(b"MPSGraph\0").unwrap();
let cls = Class::get(class_name).unwrap();
```

## Priority of Work

1. Fix core import issues to get the project to at least parse
2. Address metal type casting issues (RefEncode)
3. Fix memory management issues
4. Address catch_unwind/UnwindSafe issues
5. Correct Foundation object handling
6. Test and refine

## Conclusion

Migrating from objc-rs to objc2 requires a systematic approach to update all Objective-C interoperability code. By following this plan, we can ensure a successful transition to the more modern, safer objc2 crate while preserving all functionality.

The migration will result in:
- Better type safety through objc2's design
- Improved memory management with Retained smart pointers
- Better Foundation framework integration via objc2-foundation
- More maintainable codebase with modern Rust practices