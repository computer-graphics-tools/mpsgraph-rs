use crate::{DataType, Graph, GraphObject, Operation, Shape, ns_number_array_to_boxed_slice};
use objc2::{
    extern_class, extern_conformance, extern_methods, msg_send,
    rc::{Allocated, Retained},
    runtime::NSObject,
};
use objc2_foundation::{CopyingHelper, NSCopying, NSObjectProtocol};

extern_class!(
    /// Defines a variable operation within a graph.
    ///
    /// # Details
    ///
    /// Mirrors Apple’s `MPSGraphVariableOp`. See
    /// <https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphvariableop?language=objc>
    #[unsafe(super(Operation, GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct GraphVariableOp;
);

extern_conformance!(
    unsafe impl NSCopying for GraphVariableOp {}
);

unsafe impl CopyingHelper for GraphVariableOp {
    type Result = Self;
}

extern_conformance!(
    unsafe impl NSObjectProtocol for GraphVariableOp {}
);

impl GraphVariableOp {
    extern_methods!(
        /// Returns the data type of the variable.
        ///
        /// # Returns
        ///
        /// The variable's [`DataType`].
        #[unsafe(method(dataType))]
        #[unsafe(method_family = none)]
        pub fn data_type(&self) -> DataType;
    );
}

/// Methods declared on superclass `MPSGraphOperation`.
impl GraphVariableOp {
    extern_methods!(
        /// This initializer is unavailable.
        ///
        /// # Details
        ///
        /// Use graph methods to create and initialize variable operations.
        #[unsafe(method(init))]
        #[unsafe(method_family = init)]
        pub fn init(this: Allocated<Self>) -> Retained<Self>;
    );
}

/// Methods declared on superclass `NSObject`.
impl GraphVariableOp {
    extern_methods!(
        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        pub fn new() -> Retained<Self>;
    );
}

impl GraphVariableOp {
    /// Returns the shape of the variable.
    ///
    /// # Returns
    ///
    /// A boxed slice containing the variable shape.
    pub fn shape(&self) -> Box<[isize]> {
        let array: Retained<Shape> = unsafe { msg_send![self, shape] };
        ns_number_array_to_boxed_slice(&array)
    }
}
