use crate::{Graph, GraphObject, Tensor};
use objc2::rc::Retained;
use objc2::runtime::NSObject;
use objc2::{extern_class, extern_conformance, extern_methods};
use objc2_foundation::CopyingHelper;
use objc2_foundation::{NSArray, NSCopying, NSObjectProtocol, NSString};

extern_class!(
    /// A symbolic representation of a compute operation.
    ///
    /// `NSCopy` will take a refrence, this is so `NSDictionary` can work with the tensor.
    /// All operations are created, owned and destroyed by the graph.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphoperation?language=objc)
    #[unsafe(super(GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[name = "MPSGraphOperation"]
    pub struct Operation;
);

extern_conformance!(
    unsafe impl NSCopying for Operation {}
);

unsafe impl CopyingHelper for Operation {
    type Result = Self;
}

extern_conformance!(
    unsafe impl NSObjectProtocol for Operation {}
);

impl Operation {
    extern_methods!(
        /// The input tensors of the operation.
        #[unsafe(method(inputTensors))]
        #[unsafe(method_family = none)]
        pub fn input_tensors(&self) -> Retained<NSArray<Tensor>>;

        /// The output tensors of the operation.
        #[unsafe(method(outputTensors))]
        #[unsafe(method_family = none)]
        pub fn output_tensors(&self) -> Retained<NSArray<Tensor>>;

        /// The set of operations guaranteed to execute before this operation.
        #[unsafe(method(controlDependencies))]
        #[unsafe(method_family = none)]
        pub fn control_dependencies(&self) -> Retained<NSArray<Operation>>;

        /// The graph on which the operation is defined.
        #[unsafe(method(graph))]
        #[unsafe(method_family = none)]
        pub fn graph(&self) -> Retained<Graph>;

        /// Name of the operation.
        #[unsafe(method(name))]
        #[unsafe(method_family = none)]
        pub fn name(&self) -> Retained<NSString>;
    );
}
