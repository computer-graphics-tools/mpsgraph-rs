use objc2::extern_class;
use objc2::extern_conformance;
use objc2::extern_methods;
use objc2::rc::{Allocated, Retained};
use objc2_foundation::{NSObject, NSObjectProtocol};

extern_class!(
    /// The common base class for all Metal Performance Shaders Graph objects.
    ///
    /// Only the child classes should be used.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphobject?language=objc)
    #[unsafe(super(NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[name = "MPSGraphObject"]
    pub struct GraphObject;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for GraphObject {}
);

impl GraphObject {
    extern_methods!();
}

/// Methods declared on superclass `NSObject`.
impl GraphObject {
    extern_methods!(
        #[unsafe(method(init))]
        #[unsafe(method_family = init)]
        pub fn init(this: Allocated<Self>) -> Retained<Self>;

        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        pub fn new() -> Retained<Self>;
    );
}
