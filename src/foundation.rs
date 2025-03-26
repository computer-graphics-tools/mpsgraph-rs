use objc::runtime::{Object, BOOL, YES, NO};
use std::ptr;

#[link(name = "Foundation", kind = "framework")]
extern "C" {
    pub static NSString: *const Object;
    pub static NSArray: *const Object;
    pub static NSNumber: *const Object;
}

/// Utility functions for working with NSString objects
pub trait NSStringExt {
    fn from_str(string: &str) -> *mut Object;
    fn to_str(string: *mut Object) -> Option<String>;
}

// Implement as associated functions instead of methods
impl NSStringExt for *const Object {
    #[allow(clippy::wrong_self_convention)]
    fn from_str(string: &str) -> *mut Object {
        unsafe {
            let bytes = string.as_ptr() as *const i8;
            let len = string.len() as u64;
            let encoding = 4u64; // NSUTF8StringEncoding
            let result: *mut Object = msg_send![NSString, alloc];
            let result: *mut Object = msg_send![result,
                initWithBytes:bytes
                length:len
                encoding:encoding
            ];
            result
        }
    }
    
    fn to_str(string: *mut Object) -> Option<String> {
        if string.is_null() {
            return None;
        }
        
        unsafe {
            let utf8_string: *const i8 = msg_send![string, UTF8String];
            if utf8_string.is_null() {
                return None;
            }
            
            let c_str = std::ffi::CStr::from_ptr(utf8_string);
            match c_str.to_str() {
                Ok(s) => Some(s.to_string()),
                Err(_) => None,
            }
        }
    }
}

/// Utility functions for working with NSNumber objects
pub trait NSNumberExt {
    fn from_u64(value: u64) -> *mut Object;
    fn from_i64(value: i64) -> *mut Object;
    fn from_f32(value: f32) -> *mut Object;
    fn from_f64(value: f64) -> *mut Object;
    fn from_bool(value: bool) -> *mut Object;
    fn to_u64(number: *mut Object) -> Option<u64>;
    fn to_i64(number: *mut Object) -> Option<i64>;
    fn to_f32(number: *mut Object) -> Option<f32>;
    fn to_f64(number: *mut Object) -> Option<f64>;
    fn to_bool(number: *mut Object) -> Option<bool>;
}

impl NSNumberExt for *const Object {
    fn from_u64(value: u64) -> *mut Object {
        unsafe {
            let number: *mut Object = msg_send![NSNumber, alloc];
            let number: *mut Object = msg_send![number, initWithUnsignedLongLong:value];
            number
        }
    }
    
    fn from_i64(value: i64) -> *mut Object {
        unsafe {
            let number: *mut Object = msg_send![NSNumber, alloc];
            let number: *mut Object = msg_send![number, initWithLongLong:value];
            number
        }
    }
    
    fn from_f32(value: f32) -> *mut Object {
        unsafe {
            let number: *mut Object = msg_send![NSNumber, alloc];
            let number: *mut Object = msg_send![number, initWithFloat:value];
            number
        }
    }
    
    fn from_f64(value: f64) -> *mut Object {
        unsafe {
            let number: *mut Object = msg_send![NSNumber, alloc];
            let number: *mut Object = msg_send![number, initWithDouble:value];
            number
        }
    }
    
    fn from_bool(value: bool) -> *mut Object {
        unsafe {
            let number: *mut Object = msg_send![NSNumber, alloc];
            let value = if value { YES } else { NO };
            let number: *mut Object = msg_send![number, initWithBool:value];
            number
        }
    }
    
    fn to_u64(number: *mut Object) -> Option<u64> {
        if number.is_null() {
            return None;
        }
        
        unsafe {
            let value: u64 = msg_send![number, unsignedLongLongValue];
            Some(value)
        }
    }
    
    fn to_i64(number: *mut Object) -> Option<i64> {
        if number.is_null() {
            return None;
        }
        
        unsafe {
            let value: i64 = msg_send![number, longLongValue];
            Some(value)
        }
    }
    
    fn to_f32(number: *mut Object) -> Option<f32> {
        if number.is_null() {
            return None;
        }
        
        unsafe {
            let value: f32 = msg_send![number, floatValue];
            Some(value)
        }
    }
    
    fn to_f64(number: *mut Object) -> Option<f64> {
        if number.is_null() {
            return None;
        }
        
        unsafe {
            let value: f64 = msg_send![number, doubleValue];
            Some(value)
        }
    }
    
    fn to_bool(number: *mut Object) -> Option<bool> {
        if number.is_null() {
            return None;
        }
        
        unsafe {
            let value: BOOL = msg_send![number, boolValue];
            Some(value == YES)
        }
    }
}

/// Utility functions for working with NSArray objects
pub trait NSArrayExt {
    fn create_empty() -> *mut Object;
    fn from_vec<T: Copy>(vec: &[T]) -> *mut Object;
    fn from_objects(objects: &[*mut Object]) -> *mut Object;
    fn count(array: *mut Object) -> u64;
    fn object_at_index(array: *mut Object, index: u64) -> *mut Object;
    fn to_vec<T>(array: *mut Object, convert: fn(*mut Object) -> Option<T>) -> Option<Vec<T>>;
}

impl NSArrayExt for *const Object {
    fn create_empty() -> *mut Object {
        unsafe {
            let array: *mut Object = msg_send![NSArray, array];
            let _: () = msg_send![array, retain];
            array
        }
    }
    
    fn from_vec<T: Copy>(vec: &[T]) -> *mut Object {
        unsafe {
            let count = vec.len();
            let array: *mut Object = msg_send![NSArray, alloc];
            let array: *mut Object = msg_send![array,
                initWithObjects:vec.as_ptr()
                count:count
            ];
            array
        }
    }
    
    fn from_objects(objects: &[*mut Object]) -> *mut Object {
        unsafe {
            let count = objects.len();
            let array: *mut Object = msg_send![NSArray, alloc];
            let array: *mut Object = msg_send![array,
                initWithObjects:objects.as_ptr()
                count:count
            ];
            array
        }
    }
    
    fn count(array: *mut Object) -> u64 {
        if array.is_null() {
            return 0;
        }
        
        unsafe {
            msg_send![array, count]
        }
    }
    
    fn object_at_index(array: *mut Object, index: u64) -> *mut Object {
        if array.is_null() {
            return ptr::null_mut();
        }
        
        unsafe {
            msg_send![array, objectAtIndex:index]
        }
    }
    
    fn to_vec<T>(array: *mut Object, convert: fn(*mut Object) -> Option<T>) -> Option<Vec<T>> {
        if array.is_null() {
            return None;
        }
        
        unsafe {
            let count: u64 = msg_send![array, count];
            let mut result = Vec::with_capacity(count as usize);
            
            for i in 0..count {
                let obj: *mut Object = msg_send![array, objectAtIndex:i];
                if let Some(value) = convert(obj) {
                    result.push(value);
                } else {
                    return None;
                }
            }
            
            Some(result)
        }
    }
}