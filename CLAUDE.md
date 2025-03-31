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

## Working with Metal (MTLBuffer, MTLDevice, CommandQueue, CommandBuffer)

### MTLDevice

The `MTLDevice` represents the GPU. Most Metal objects are created from a device:

```rust
// Get the default system device
let device = Device::system_default().expect("No device found");

// Create resources using the device
let buffer = device.new_buffer(size_in_bytes, MTLResourceOptions::StorageModeShared);
let command_queue = device.new_command_queue();
```

### MTLBuffer

Metal buffers store data that can be accessed by the GPU:

```rust
// Create a buffer with shared storage mode (accessible by both CPU and GPU)
let buffer = device.new_buffer(
    size_in_bytes,
    MTLResourceOptions::StorageModeShared
);

// Create a buffer from existing data
let data = vec![1u32, 2, 3, 4];
let buffer = device.new_buffer_with_data(
    data.as_ptr().cast(),
    std::mem::size_of_val(&data) as u64,
    MTLResourceOptions::StorageModeShared
);

// Access buffer contents from CPU (for shared mode buffers)
let contents = unsafe {
    std::slice::from_raw_parts(
        buffer.contents().cast::<u32>(),
        buffer.length() as usize / std::mem::size_of::<u32>()
    )
};

// Modify buffer contents
unsafe {
    let ptr = buffer.contents().cast::<u32>();
    *ptr = 42;
}
```

### MTLCommandQueue

Command queues manage the execution of command buffers:

```rust
// Create a command queue
let command_queue = device.new_command_queue();

// Get a new command buffer from the queue
let command_buffer = command_queue.new_command_buffer();
```

### MTLCommandBuffer

Command buffers contain the commands to be executed by the GPU:

```rust
// Create a command buffer
let command_buffer = command_queue.new_command_buffer();

// Get an encoder from the command buffer
let compute_encoder = command_buffer.new_compute_command_encoder();

// Set up compute work with encoder
compute_encoder.set_compute_pipeline_state(&pipeline_state);
compute_encoder.set_buffer(0, Some(&input_buffer), 0);
compute_encoder.set_buffer(1, Some(&output_buffer), 0);

// Configure thread groups and dispatch
let thread_group_size = MTLSize { width: 16, height: 1, depth: 1 };
let thread_groups = MTLSize { 
    width: (size + 15) / 16, 
    height: 1, 
    depth: 1 
};
compute_encoder.dispatch_thread_groups(thread_groups, thread_group_size);

// Finish encoding
compute_encoder.end_encoding();

// Submit work to GPU
command_buffer.commit();

// Wait for completion (synchronously)
command_buffer.wait_until_completed();

// Alternatively, use completion handler (asynchronously)
command_buffer.add_completed_handler(|buffer| {
    println!("Command buffer completed with status: {:?}", buffer.status());
});
command_buffer.commit();
```

### Putting It All Together

Example workflow for a compute operation:

```rust
// Get device
let device = Device::system_default().expect("No device found");

// Create buffers
let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
let input_buffer = device.new_buffer_with_data(
    input_data.as_ptr().cast(),
    (input_data.len() * std::mem::size_of::<f32>()) as u64,
    MTLResourceOptions::StorageModeShared
);

let output_buffer = device.new_buffer(
    (input_data.len() * std::mem::size_of::<f32>()) as u64,
    MTLResourceOptions::StorageModeShared
);

// Create compute pipeline
let library = device.new_library_with_source(
    "kernel void square(device float *input [[buffer(0)]],
                       device float *output [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
        output[id] = input[id] * input[id];
    }",
    &CompileOptions::new()
).unwrap();

let kernel = library.get_function("square", None).unwrap();
let pipeline_state = device.new_compute_pipeline_state_with_function(&kernel).unwrap();

// Create command queue and buffer
let command_queue = device.new_command_queue();
let command_buffer = command_queue.new_command_buffer();

// Set up and execute compute work
let compute_encoder = command_buffer.new_compute_command_encoder();
compute_encoder.set_compute_pipeline_state(&pipeline_state);
compute_encoder.set_buffer(0, Some(&input_buffer), 0);
compute_encoder.set_buffer(1, Some(&output_buffer), 0);

let thread_group_size = MTLSize { width: 1, height: 1, depth: 1 };
let thread_groups = MTLSize { width: input_data.len() as u64, height: 1, depth: 1 };
compute_encoder.dispatch_thread_groups(thread_groups, thread_group_size);
compute_encoder.end_encoding();

// Submit and wait
command_buffer.commit();
command_buffer.wait_until_completed();

// Read results
let results = unsafe {
    std::slice::from_raw_parts(
        output_buffer.contents().cast::<f32>(),
        input_data.len()
    )
};
println!("Results: {:?}", results);
```

### Best Practices

1. **Resource Management**: Release resources when done (Metal-rs handles this with Drop)
2. **Buffer Options**: Choose appropriate storage mode for your use case:
   - `StorageModeShared`: Accessible by both CPU and GPU, but slower
   - `StorageModeManaged`: CPU and GPU have separate copies, synchronization managed by driver
   - `StorageModePrivate`: GPU-only access, fastest for GPU-only data
3. **Multiple Encoders**: You can create multiple encoders for different work in the same command buffer
4. **Synchronization**: Use events and fences for cross-command buffer synchronization
5. **Autorelease Pool**: Always wrap Metal code in `objc::rc::autoreleasepool` to prevent memory leaks

## Working with objc2

### Overview

[objc2](https://github.com/madsmtm/objc2) is the Rust crate used for Objective-C interoperability in mpsgraph-rs. It provides safe Rust bindings to interact with the Objective-C runtime and Apple's frameworks. The [objc2-foundation](https://docs.rs/objc2-foundation/latest/objc2_foundation/) crate provides Rust bindings for Apple's Foundation framework.

### Basic Principles

- **Wrapper Types**: All Objective-C objects are wrapped in Rust structs with a raw pointer (`*mut AnyObject`)
- **Memory Management**: Either manually manage retain/release cycles with `objc_retain` and `objc_release` or use `objc2-foundation`'s `Retained<T>` and reference-counted types
- **Method Calling**: Use the `msg_send!` macro for calling Objective-C methods
- **Safety**: Many operations are marked as `unsafe` due to the nature of FFI
- **Type Conversion**: The `objc2-foundation` crate provides idiomatic conversions between Rust and Objective-C types

### Common Patterns

1. **Creating Wrapper Types**:

   ```rust
   pub struct MPSGraphTensor(pub(crate) *mut AnyObject);
   
   // Implement Send + Sync 
   unsafe impl Send for MPSGraphTensor {}
   unsafe impl Sync for MPSGraphTensor {}
   ```

2. **Memory Management**:

   ```rust
   // Retain when cloning an object
   impl Clone for MPSGraphTensor {
       fn clone(&self) -> Self {
           unsafe {
               if !self.0.is_null() {
                   let obj = objc2::ffi::objc_retain(self.0 as *mut _) as *mut AnyObject;
                   MPSGraphTensor(obj)
               } else {
                   MPSGraphTensor(ptr::null_mut())
               }
           }
       }
   }
   
   // Release when dropping
   impl Drop for MPSGraphTensor {
       fn drop(&mut self) {
           unsafe {
               if !self.0.is_null() {
                   objc2::ffi::objc_release(self.0 as *mut _);
               }
           }
       }
   }
   ```

3. **Method Calling with `msg_send!`**:

   ```rust
   // Calling Objective-C methods
   unsafe {
       let data_type_val: u64 = msg_send![self.0, dataType];
       std::mem::transmute(data_type_val as u32)
   }
   ```

### Working with NSString

Objective-C strings (NSString) can be accessed and manipulated in several ways:

1. **Converting from NSString to Rust String**:

   ```rust
   unsafe {
       let name: *mut AnyObject = msg_send![self.0, name];
       
       // Handle nil case
       if name.is_null() {
           return String::from("<default>");
       }
       
       // Convert to UTF8
       let utf8: *const i8 = msg_send![name, UTF8String];
       std::ffi::CStr::from_ptr(utf8).to_string_lossy().to_string()
   }
   ```

2. **Using `objc2_foundation`'s NSString**:

   ```rust
   use objc2_foundation::{ns_string, NSString};
   
   // Create a static NSString
   let string = ns_string!("Hello, world!");
   
   // Convert to Rust String
   let rust_string = string.to_string();
   ```

3. **Creating NSString from Rust String**:

   ```rust
   let rust_str = "Hello";
   let ns_string = NSString::from_str(rust_str);
   ```

4. **Memory Characteristics**:

   NSString behaves like `Rc<str>` in Rust - immutable, reference-counted, and thread-safe. The `ns_string!` macro creates compile-time constant strings efficiently.

### Working with NSData

NSData is used for handling binary data in Objective-C:

1. **Creating NSData from Rust slice**:

   ```rust
   use objc2_foundation::NSData;
   
   let bytes = [1, 2, 3, 4];
   let ns_data = NSData::with_bytes(&bytes);
   ```

2. **Using NSData with Metal**:

   ```rust
   // Create NSData with byte slice
   let ns_data = NSData::with_bytes(std::slice::from_raw_parts(
       data.as_ptr() as *const u8,
       data_size
   ));
   
   // Get the raw pointer to NSData for objc2
   let ns_data_ptr: *mut AnyObject = std::mem::transmute::<&NSData, *mut AnyObject>(ns_data.as_ref());
   ```

3. **Accessing bytes from NSData**:

   ```rust
   // Get bytes as Rust slice (unsafe)
   let bytes = unsafe { ns_data.as_bytes_unchecked() };
   
   // Copy bytes to a new Vec
   let bytes_vec = ns_data.to_vec();
   ```

4. **Memory Characteristics**:

   NSData behaves like `Rc<[u8]>` in Rust - immutable, reference-counted binary data. For mutable data, use `NSMutableData` which behaves like `Rc<Cell<Vec<u8>>>`.

5. **Working with NSMutableData**:

   ```rust
   use objc2_foundation::NSMutableData;
   
   // Create a mutable data buffer
   let mut_data = NSMutableData::with_capacity(1024);
   
   // Append bytes
   mut_data.append_bytes(&[1, 2, 3, 4]);
   
   // Get mutable pointer to data (unsafe)
   let ptr = unsafe { mut_data.mutable_bytes() };
   ```

### Working with NSArray

NSArray is used for collections of Objective-C objects:

1. **Creating a Custom Array Wrapper**:

   ```rust
   pub(crate) struct OurNSArray(pub(crate) *mut AnyObject);
   
   impl OurNSArray {
       pub fn from_objects(objects: &[*mut AnyObject]) -> Self {
           unsafe {
               // Convert raw pointers to references
               let refs: Vec<&objc2::runtime::AnyObject> = objects.iter()
                   .map(|&p| &*p.cast::<objc2::runtime::AnyObject>())
                   .collect();
               
               // Create array from references
               let array = NSArray::from_slice(&refs);
               let ptr: *mut AnyObject = std::mem::transmute::<&NSArray<objc2::runtime::AnyObject>, 
                                                             *mut AnyObject>(array.as_ref());
               
               objc2::ffi::objc_retain(ptr as *mut _);
               OurNSArray(ptr)
           }
       }
   }
   
   impl Drop for OurNSArray {
       fn drop(&mut self) {
           unsafe {
               if !self.0.is_null() {
                   objc2::ffi::objc_release(self.0 as *mut _);
               }
           }
       }
   }
   ```

2. **Creating an NSArray from Tensors**:

   ```rust
   // Create array of raw tensor pointers
   let output_tensors_raw: Vec<*mut AnyObject> = output_tensors.iter()
       .map(|t| t.0)
       .collect();
   
   let output_tensors_array = OurNSArray::from_objects(&output_tensors_raw);
   ```

3. **Using Foundation's NSArray Directly**:

   ```rust
   use objc2_foundation::NSArray;
   
   // Create from slice of references
   let objects = vec![&ns_object1, &ns_object2];
   let array = NSArray::from_slice(&objects);
   
   // Iterate over elements
   for obj in array.iter() {
       // Use obj
   }
   
   // Convert to Vec
   let vector = array.to_vec();
   ```

4. **Memory Characteristics**:

   NSArray behaves like `Rc<[Retained<T>]>` in Rust - an immutable, reference-counted collection of objects. For mutable collections, use `NSMutableArray` which behaves like `Rc<Cell<Vec<Retained<T>>>>`.

5. **Using NSMutableArray**:

   ```rust
   use objc2_foundation::NSMutableArray;
   
   // Create a mutable array
   let mut_array = NSMutableArray::new();
   
   // Add objects
   mut_array.add_object(&ns_object1);
   
   // Insert at index
   mut_array.insert_object_at(&ns_object2, 0);
   
   // Remove objects
   mut_array.remove_object_at(1);
   ```

### Working with NSDictionary

NSDictionary is used for key-value mappings in Objective-C:

1. **Creating a Custom Dictionary Wrapper**:

   ```rust
   pub(crate) struct OurNSDictionary(pub(crate) *mut AnyObject);
   
   impl OurNSDictionary {
       pub fn from_keys_and_objects(keys: &[*mut AnyObject], objects: &[*mut AnyObject]) -> Self {
           unsafe {
               // Create references needed for Objective-C
               let key_refs: Vec<&objc2::runtime::AnyObject> = keys.iter()
                   .map(|&ptr| &*ptr.cast::<objc2::runtime::AnyObject>())
                   .collect();
                   
               let obj_refs: Vec<&objc2::runtime::AnyObject> = objects.iter()
                   .map(|&ptr| &*ptr.cast::<objc2::runtime::AnyObject>())
                   .collect();
               
               // Create dictionary using Objective-C method
               let cls = objc2::runtime::AnyClass::get(c"NSDictionary").unwrap();
               let dict_ptr: *mut AnyObject = msg_send![cls, 
                   dictionaryWithObjects: obj_refs.as_ptr(), 
                   forKeys: key_refs.as_ptr(), 
                   count: key_refs.len()
               ];
               
               OurNSDictionary(dict_ptr)
           }
       }
   }
   
   impl Drop for OurNSDictionary {
       fn drop(&mut self) {
           unsafe {
               if !self.0.is_null() {
                   objc2::ffi::objc_release(self.0 as *mut _);
               }
           }
       }
   }
   ```

2. **Creating a Dictionary of Tensors and Tensor Data**:

   ```rust
   // Create feed dictionary from HashMap
   let mut feed_keys = Vec::with_capacity(feeds.len());
   let mut feed_values = Vec::with_capacity(feeds.len());
   
   for (tensor, data) in feeds {
       feed_keys.push(tensor.0);
       feed_values.push(data.0);
   }
   
   let feed_dict = OurNSDictionary::from_keys_and_objects(&feed_keys, &feed_values);
   ```

3. **Using Foundation's NSDictionary Directly**:

   ```rust
   use objc2_foundation::{NSDictionary, NSString};
   
   // Create keys and values
   let keys = &[ns_string1, ns_string2];
   let objects = &[&*obj1, &*obj2];
   
   // Create dictionary
   let dict = NSDictionary::from_slices(keys, objects);
   
   // Look up a value
   let value = dict.objectForKey(key);
   
   // Get dictionary size
   let count = dict.len();
   ```

4. **Iterating Over a Dictionary**:

   ```rust
   // Get enumerator for the dictionary
   let enumerator: *mut AnyObject = msg_send![results, keyEnumerator];
   
   while {
       let key: *mut AnyObject = msg_send![enumerator, nextObject];
       !key.is_null()
   } {
       let key: *mut AnyObject = msg_send![enumerator, currentObject];
       let value: *mut AnyObject = msg_send![results, objectForKey: key];
       
       // Retain objects to manage their memory
       let _: () = msg_send![key, retain];
       let _: () = msg_send![value, retain];
       
       // Use key and value
   }
   ```

5. **Memory Characteristics**:

   NSDictionary behaves like `Rc<HashMap<Retained<K>, Retained<V>>>` in Rust - an immutable, reference-counted mapping where both keys and values are retained. For mutable dictionaries, use `NSMutableDictionary`.

6. **Using NSMutableDictionary**:

   ```rust
   use objc2_foundation::NSMutableDictionary;
   
   // Create a mutable dictionary
   let mut_dict = NSMutableDictionary::new();
   
   // Set values
   mut_dict.set_object_for(&value, &key);
   
   // Remove values
   mut_dict.remove_object_for(&key);
   ```

### Working with NSError

Error handling in Objective-C often involves passing an error pointer by reference:

1. **Custom NSError Wrapper**:

   ```rust
   pub struct OurNSError(pub(crate) *mut AnyObject);
   
   impl OurNSError {
       pub fn new(error_ptr: *mut AnyObject) -> Self {
           unsafe {
               if !error_ptr.is_null() {
                   objc2::ffi::objc_retain(error_ptr as *mut _);
               }
               OurNSError(error_ptr)
           }
       }
       
       pub fn localized_description(&self) -> String {
           unsafe {
               if self.0.is_null() {
                   return String::from("<null error>");
               }
               
               let ns_error: &NSError = &*(self.0 as *const objc2_foundation::NSError);
               ns_error.localizedDescription().to_string()
           }
       }
   }
   
   impl Drop for OurNSError {
       fn drop(&mut self) {
           unsafe {
               if !self.0.is_null() {
                   objc2::ffi::objc_release(self.0 as *mut _);
               }
           }
       }
   }
   ```

2. **Using NSError with Objective-C Methods**:

   ```rust
   // Create a mutable pointer to receive the error
   let mut error_ptr: *mut AnyObject = std::ptr::null_mut();
   
   // Call a method that might produce an error
   let result: *mut AnyObject = unsafe {
       msg_send![obj, doSomethingWithError:&mut error_ptr]
   };
   
   // Check for error
   if !error_ptr.is_null() {
       let error = OurNSError::new(error_ptr);
       println!("Error: {}", error.localized_description());
       return Err(error);
   }
   ```

3. **Using Foundation's NSError Directly**:

   ```rust
   use objc2_foundation::NSError;
   
   // Get error information
   let domain = ns_error.domain().to_string();
   let code = ns_error.code();
   let description = ns_error.localizedDescription().to_string();
   ```

4. **Memory Characteristics**:

   NSError in `objc2-foundation` maps to `Arc<dyn Error + Send + Sync>` in Rust - an error type that works with Rust's standard error handling mechanisms. It implements both `std::error::Error` and can be used with `?` operator.

5. **Creating NSError Objects**:

   ```rust
   use objc2_foundation::{NSError, NSString};
   
   // Create an NSError
   let domain = NSString::from_str("com.example.error");
   let code = 100;
   let user_info = NSDictionary::new();
   let error = NSError::new(&domain, code, &user_info);
   ```

### Working with Objective-C Classes

Objective-C classes can be accessed and used in several ways:

1. **Getting a Class Reference**:

   ```rust
   // Get class by name
   let class_name = c"NSString";  // c-string literal
   if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
       // Use the class
   }
   ```

2. **Allocating and Initializing Objects**:

   ```rust
   // Standard alloc-init pattern
   let obj: *mut AnyObject = unsafe {
       let alloc: *mut AnyObject = msg_send![cls, alloc];
       msg_send![alloc, init]
   };
   ```

3. **Using Class Methods**:

   ```rust
   // Call a class method (static method in Rust terminology)
   let string_ptr: *mut NSString = unsafe {
       msg_send![cls, stringWithUTF8String:c_string.as_ptr()]
   };
   ```

4. **Using the `extern_methods!` Macro**:

   ```rust
   extern_methods!(
       #[unsafe(method(mutableBytes))]
       pub(crate) fn mutable_bytes_raw(&self) -> *mut c_void;
   );
   ```

5. **Using the `ClassType` Trait**:

   ```rust
   use objc2::{ClassType, runtime::AnyClass};
   
   trait MyClassType: ClassType {
       // Define methods on the class
       fn my_class_method(arg1: u32) -> Self;
   }
   
   // Implement for AnyClass
   impl MyClassType for AnyClass {
       fn my_class_method(arg1: u32) -> Self {
           unsafe {
               let ptr: *mut Self = msg_send![Self::class(), myMethodWithArg:arg1];
               Self::from_ptr(ptr).unwrap()
           }
       }
   }
   ```

6. **Using the Declare Macros**:

   The objc2 crate provides macros for declaring Objective-C classes in Rust:

   ```rust
   use objc2::declare::ClassBuilder;
   
   // Create a new class at runtime
   let my_class = ClassBuilder::new("MyClass", NSObject::class())
       .add_method(sel!(doSomething), do_something_impl as extern "C" fn(_, _))
       .build();
   ```

### Best Practices

1. **Null Checking**: Always check if pointers are null before using them
2. **Memory Leaks**: Ensure every retained object is released
3. **Error Handling**: Use `Option<T>` for nullable values and check results
4. **Type Safety**: Use appropriate Rust types to represent Objective-C types
5. **Unsafe Blocks**: Minimize the scope of unsafe blocks
6. **Documentation**: Document unsafe requirements and invariants
7. **Custom Wrappers**: Create dedicated wrapper types for Objective-C objects
8. **Consistent Patterns**: Follow consistent patterns for memory management
9. **Retain/Release Balance**: Always balance retain/release calls
10. **Method Naming**: Use Rust-style naming for wrapper methods (snake_case)
11. **Debugging**: Use Metal's validation layers and enable NSZombies for debugging memory issues
12. **Use objc2-foundation Types**: Prefer using `objc2-foundation` types like `NSString`, `NSData`, etc. when possible
13. **Retained Types**: Use `Retained<T>` when working with collections to ensure proper memory management
14. **Reference Counting**: Understand that most Foundation types use reference counting similar to Rust's `Rc` or `Arc`
15. **Thread Safety**: Consider thread safety when using Foundation types across threads
16. **IMPORTANT: Avoid Custom Foundation Type Wrappers**: Never create custom wrappers like `OurNSString`, `OurNSArray`, or `OurNSDictionary`. Always use the `objc2-foundation` types (`NSString`, `NSArray`, `NSDictionary`) directly instead.

### Additional objc2-foundation Types

1. **NSNumber**: Represents numeric values in Objective-C:

   ```rust
   use objc2_foundation::NSNumber;
   
   // Create from various numeric types
   let n1 = NSNumber::from_i32(42);
   let n2 = NSNumber::from_bool(true);
   let n3 = NSNumber::from_f64(3.14);
   
   // Extract values
   let i: i32 = n1.as_i32();
   let b: bool = n2.as_bool();
   let f: f64 = n3.as_f64();
   ```

2. **NSRange**: Represents a range in Objective-C:

   ```rust
   use std::ops::Range;
   use objc2_foundation::NSRange;
   
   // Create from Rust range
   let rust_range = 2..10;
   let ns_range = NSRange::from(rust_range);
   
   // Convert back to Rust range
   let back_to_rust: Range<usize> = ns_range.into();
   ```

3. **NSValue**: Wraps various value types in Objective-C:

   ```rust
   use objc2_foundation::NSValue;
   
   // Create from NSRange
   let range = NSRange::new(0, 10);
   let value = NSValue::from_range(&range);
   
   // Extract range
   let extracted: NSRange = value.as_range();
   ```
