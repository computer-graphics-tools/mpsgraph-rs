use super::{DataType, GraphObject, GraphType, Shape};
use crate::{ns_number_array_from_slice, ns_number_array_to_boxed_slice};
use objc2::{
    ClassType, extern_class, extern_conformance, extern_methods, msg_send,
    rc::{Allocated, Retained},
};
use objc2_foundation::{CopyingHelper, NSCopying, NSObject, NSObjectProtocol};

extern_class!(
    /// The shaped type class for types on tensors with a shape and data type.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphshapedtype?language=objc)
    #[unsafe(super(GraphType, GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[name = "MPSGraphShapedType"]
    pub struct ShapedType;
);

extern_conformance!(
    unsafe impl NSCopying for ShapedType {}
);

unsafe impl CopyingHelper for ShapedType {
    type Result = Self;
}

extern_conformance!(
    unsafe impl NSObjectProtocol for ShapedType {}
);

impl ShapedType {
    extern_methods!(
        /// The data type of the shaped type.
        #[unsafe(method(dataType))]
        #[unsafe(method_family = none)]
        pub fn data_type(&self) -> DataType;

        /// Setter for [`dataType`][Self::dataType].
        #[unsafe(method(setDataType:))]
        #[unsafe(method_family = none)]
        pub fn set_data_type(&self, data_type: DataType);

        /// Checks if shapes and element data type are the same as the input shaped type.
        ///
        /// # Arguments
        ///
        /// * `object` - [`ShapedType`] to compare with.
        ///
        /// # Returns
        ///
        /// `true` if equal, `false` otherwise
        #[unsafe(method(isEqualTo:))]
        #[unsafe(method_family = none)]
        pub fn is_equal_to(&self, object: Option<&ShapedType>) -> bool;
    );
}

/// Methods declared on superclass `NSObject`.
impl ShapedType {
    extern_methods!(
        #[unsafe(method(init))]
        #[unsafe(method_family = init)]
        pub fn init(this: Allocated<Self>) -> Retained<Self>;

        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        pub fn new() -> Retained<Self>;
    );
}

impl ShapedType {
    /// The Shape of the shaped type.
    pub fn shape(&self) -> Option<Box<[isize]>> {
        let array: Option<Retained<Shape>> = unsafe { msg_send![self, shape] };
        array.map(|array| ns_number_array_to_boxed_slice(&array))
    }

    /// Setter for [`shape`][Self::shape].
    pub fn set_shape(&self, shape: Option<&[isize]>) {
        let shape = shape.map(|s| ns_number_array_from_slice(s));
        let _: () = unsafe { msg_send![self, setShape: shape.as_deref()] };
    }

    /// Initializes a shaped type.
    ///
    /// # Arguments
    ///
    /// * `shape` - Optional [`[isize]`] of the shaped type.
    /// * `data_type` - [`DataType`] of the shaped type.
    ///
    /// # Returns
    ///
    /// A valid [`ShapedType`] object, or `nil` if allocation failure.
    pub fn new_with_shape_data_type(
        shape: Option<&[isize]>,
        data_type: DataType,
    ) -> Retained<Self> {
        let class = Self::class();
        let allocated: Allocated<Self> = unsafe { msg_send![class, alloc] };
        let shape = shape.map(|s| ns_number_array_from_slice(s));
        unsafe { msg_send![allocated, initWithShape: shape.as_deref(), dataType: data_type] }
    }
}
