use crate::{DataType, GraphObject, Operation, Shape, ShapedType, ns_number_array_to_boxed_slice};
use objc2::{
    extern_class, extern_conformance, extern_methods, msg_send, rc::Retained, runtime::NSObject,
};
use objc2_foundation::{CopyingHelper, NSCopying, NSObjectProtocol};

extern_class!(
    /// The symbolic representation of a compute data type.
    ///
    /// `NSCopy` will take a refrence, this is so `NSDictionary` can work with the tensor.
    /// All tensors are created, owned and destroyed by the MPSGraph
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphtensor?language=objc)
    #[unsafe(super(GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[name = "MPSGraphTensor"]
    pub struct Tensor;
);

extern_conformance!(
    unsafe impl NSCopying for Tensor {}
);

unsafe impl CopyingHelper for Tensor {
    type Result = Self;
}

extern_conformance!(
    unsafe impl NSObjectProtocol for Tensor {}
);

impl Tensor {
    extern_methods!(
        /// The data type of the tensor.
        #[unsafe(method(dataType))]
        #[unsafe(method_family = none)]
        pub fn data_type(&self) -> DataType;

        /// The operation responsible for creating this tensor.
        #[unsafe(method(operation))]
        #[unsafe(method_family = none)]
        pub fn operation(&self) -> Retained<Operation>;
    );
}

impl Tensor {
    /// The shape of the tensor.
    ///
    /// nil shape represents an unranked tensor.
    /// -1 value for a dimension represents that it will be resolved via shape inference at runtime and it can be anything.
    pub fn shape(&self) -> Option<Box<[isize]>> {
        let shape: Option<Retained<Shape>> = unsafe { msg_send![self, shape] };
        shape.map(|shape| ns_number_array_to_boxed_slice(&shape))
    }

    pub fn shaped_type(&self) -> Retained<ShapedType> {
        let shape = self.shape();
        let data_type = self.data_type();
        ShapedType::new_with_shape_data_type(shape.as_deref(), data_type)
    }
}
