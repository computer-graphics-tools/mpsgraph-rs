use crate::DeviceType;
use crate::GraphObject;
use metal::Device as MetalDevice;
use metal::foreign_types::ForeignType;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject};
use objc2::{ClassType, extern_class, extern_conformance, extern_methods, msg_send};
use objc2_foundation::NSObjectProtocol;

extern_class!(
    /// A class that describes the compute device.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphdevice?language=objc)
    #[unsafe(super(GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[name = "MPSGraphDevice"]
    pub struct Device;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for Device {}
);

impl Device {
    extern_methods!(
        /// Device of the MPSGraphDevice.
        #[unsafe(method(type))]
        #[unsafe(method_family = none)]
        pub fn r#type(&self) -> DeviceType;
    );
}

impl Device {
    /// Creates a new Device using the system default Metal device
    pub fn new() -> Retained<Self> {
        let device = MetalDevice::system_default()
            .expect("No Metal device found")
            .to_owned();
        Self::with_device(&device)
    }

    /// Creates a new Device from a Metal device
    pub fn with_device(device: &MetalDevice) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let device_ptr = device.as_ptr() as *mut AnyObject;
            msg_send![class, deviceWithMTLDevice: device_ptr]
        }
    }
}
