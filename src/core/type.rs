use super::GraphObject;
use objc2::extern_class;
use objc2::extern_conformance;
use objc2::extern_methods;
use objc2::rc::{Allocated, Retained};
use objc2_foundation::CopyingHelper;
use objc2_foundation::{NSCopying, NSObject, NSObjectProtocol};

extern_class!(
    /// The base type class for types on tensors.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphtype?language=objc)
    #[unsafe(super(GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[name = "MPSGraphType"]
    pub struct GraphType;
);

extern_conformance!(
    unsafe impl NSCopying for GraphType {}
);

unsafe impl CopyingHelper for GraphType {
    type Result = Self;
}

extern_conformance!(
    unsafe impl NSObjectProtocol for GraphType {}
);

impl GraphType {
    extern_methods!();
}

/// Methods declared on superclass `NSObject`.
impl GraphType {
    extern_methods!(
        #[unsafe(method(init))]
        #[unsafe(method_family = init)]
        pub fn init(this: Allocated<Self>) -> Retained<Self>;

        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        pub fn new() -> Retained<Self>;
    );
}
