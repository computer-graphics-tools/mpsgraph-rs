use super::DeploymentPlatform;
use objc2::rc::{Allocated, Retained};
use objc2::runtime::NSObject;
use objc2::{extern_class, extern_conformance, extern_methods};
use objc2_foundation::{NSObjectProtocol, NSString};

use crate::GraphObject;

extern_class!(
    /// A class that consists of all the levers  to serialize an executable.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphexecutableserializationdescriptor?language=objc)
    #[unsafe(super(GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[name = "MPSGraphExecutableSerializationDescriptor"]
    pub struct ExecutableSerializationDescriptor;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for ExecutableSerializationDescriptor {}
);

impl ExecutableSerializationDescriptor {
    extern_methods!(
        /// Flag to append to an existing .mpsgraphpackage if found at provided url.
        ///
        /// If false, the exisiting .mpsgraphpackage will be overwritten.
        #[unsafe(method(append))]
        #[unsafe(method_family = none)]
        pub fn append(&self) -> bool;

        /// Setter for [`append`][Self::append].
        #[unsafe(method(setAppend:))]
        #[unsafe(method_family = none)]
        pub fn set_append(&self, append: bool);

        /// The deployment platform used to serialize the executable.
        ///
        /// Defaults to the current platform.
        #[unsafe(method(deploymentPlatform))]
        #[unsafe(method_family = none)]
        pub fn deployment_platform(&self) -> DeploymentPlatform;

        /// Setter for [`deploymentPlatform`][Self::deploymentPlatform].
        #[unsafe(method(setDeploymentPlatform:))]
        #[unsafe(method_family = none)]
        pub fn set_deployment_platform(&self, deployment_platform: DeploymentPlatform);

        /// The minimum deployment target to serialize the executable.
        ///
        /// If not set, the package created will target the latest version of the `deploymentPlatform` set.
        #[unsafe(method(minimumDeploymentTarget))]
        #[unsafe(method_family = none)]
        pub fn minimum_deployment_target(&self) -> Retained<NSString>;

        /// Setter for [`minimumDeploymentTarget`][Self::minimumDeploymentTarget].
        #[unsafe(method(setMinimumDeploymentTarget:))]
        #[unsafe(method_family = none)]
        pub fn set_minimum_deployment_target(&self, minimum_deployment_target: &NSString);
    );
}

/// Methods declared on superclass `NSObject`.
impl ExecutableSerializationDescriptor {
    extern_methods!(
        #[unsafe(method(init))]
        #[unsafe(method_family = init)]
        pub fn init(this: Allocated<Self>) -> Retained<Self>;

        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        pub fn new() -> Retained<Self>;
    );
}
