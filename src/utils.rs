use objc2::rc::{Retained, autoreleasepool};
use objc2_foundation::{NSArray, NSInteger, NSNumber, NSUInteger};

pub trait NSNumberValue: Copy {}
pub trait NSNumberValueExt<T: NSNumberValue> {
    fn value(&self) -> T;
    fn new(value: T) -> Retained<NSNumber>;
}

impl NSNumberValue for i8 {}
impl NSNumberValueExt<i8> for NSNumber {
    fn value(&self) -> i8 {
        self.charValue()
    }
    fn new(value: i8) -> Retained<NSNumber> {
        NSNumber::new_i8(value)
    }
}

impl NSNumberValue for u8 {}
impl NSNumberValueExt<u8> for NSNumber {
    fn value(&self) -> u8 {
        self.unsignedCharValue()
    }
    fn new(value: u8) -> Retained<NSNumber> {
        NSNumber::new_u8(value)
    }
}

impl NSNumberValue for i16 {}
impl NSNumberValueExt<i16> for NSNumber {
    fn value(&self) -> i16 {
        self.shortValue()
    }
    fn new(value: i16) -> Retained<NSNumber> {
        NSNumber::new_i16(value)
    }
}

impl NSNumberValue for u16 {}
impl NSNumberValueExt<u16> for NSNumber {
    fn value(&self) -> u16 {
        self.unsignedShortValue()
    }
    fn new(value: u16) -> Retained<NSNumber> {
        NSNumber::new_u16(value)
    }
}

impl NSNumberValue for i32 {}
impl NSNumberValueExt<i32> for NSNumber {
    fn value(&self) -> i32 {
        self.intValue()
    }
    fn new(value: i32) -> Retained<NSNumber> {
        NSNumber::new_i32(value)
    }
}

impl NSNumberValue for u32 {}
impl NSNumberValueExt<u32> for NSNumber {
    fn value(&self) -> u32 {
        self.unsignedIntValue()
    }
    fn new(value: u32) -> Retained<NSNumber> {
        NSNumber::new_u32(value)
    }
}

impl NSNumberValue for i64 {}
impl NSNumberValueExt<i64> for NSNumber {
    fn value(&self) -> i64 {
        self.longValue()
    }
    fn new(value: i64) -> Retained<NSNumber> {
        NSNumber::new_i64(value)
    }
}

impl NSNumberValue for u64 {}
impl NSNumberValueExt<u64> for NSNumber {
    fn value(&self) -> u64 {
        self.unsignedLongValue()
    }
    fn new(value: u64) -> Retained<NSNumber> {
        NSNumber::new_u64(value)
    }
}

impl NSNumberValue for f32 {}
impl NSNumberValueExt<f32> for NSNumber {
    fn value(&self) -> f32 {
        self.floatValue()
    }
    fn new(value: f32) -> Retained<NSNumber> {
        NSNumber::new_f32(value)
    }
}

impl NSNumberValue for f64 {}
impl NSNumberValueExt<f64> for NSNumber {
    fn value(&self) -> f64 {
        self.doubleValue()
    }
    fn new(value: f64) -> Retained<NSNumber> {
        NSNumber::new_f64(value)
    }
}

impl NSNumberValue for bool {}
impl NSNumberValueExt<bool> for NSNumber {
    fn value(&self) -> bool {
        self.boolValue()
    }
    fn new(value: bool) -> Retained<NSNumber> {
        NSNumber::new_bool(value)
    }
}

impl NSNumberValue for NSInteger {}
impl NSNumberValueExt<NSInteger> for NSNumber {
    fn value(&self) -> NSInteger {
        self.integerValue()
    }
    fn new(value: NSInteger) -> Retained<NSNumber> {
        NSNumber::new_isize(value)
    }
}

impl NSNumberValue for NSUInteger {}
impl NSNumberValueExt<NSUInteger> for NSNumber {
    fn value(&self) -> NSUInteger {
        self.unsignedIntegerValue()
    }
    fn new(value: NSUInteger) -> Retained<NSNumber> {
        NSNumber::new_usize(value)
    }
}

pub fn ns_number_array_to_boxed_slice<T: NSNumberValue>(shape: &NSArray<NSNumber>) -> Box<[T]>
where
    NSNumber: NSNumberValueExt<T>,
{
    autoreleasepool(|_| {
        shape
            .iter()
            .map(|num| num.value())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    })
}

pub fn ns_number_array_from_slice<T: NSNumberValue>(slice: &[T]) -> Retained<NSArray<NSNumber>>
where
    NSNumber: NSNumberValueExt<T>,
{
    autoreleasepool(|_| {
        let ns_numbers: Box<[Retained<NSNumber>]> =
            slice.iter().map(|val| NSNumber::new(*val)).collect();
        NSArray::from_retained_slice(&ns_numbers)
    })
}
