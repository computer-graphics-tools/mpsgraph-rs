use crate::{
    DataType, Device, GraphObject, Shape, ns_number_array_from_slice,
    ns_number_array_to_boxed_slice,
};
use metal::{Buffer, foreign_types::ForeignType};
use objc2::{
    ClassType, extern_class, extern_conformance, extern_methods, msg_send,
    rc::{Allocated, Retained, autoreleasepool},
    runtime::{AnyObject, NSObject},
};
use objc2_foundation::{NSData, NSObjectProtocol};
use std::mem::size_of_val;
use std::slice::from_raw_parts;

extern_class!(
    /// The representation of a compute data type.
    ///
    /// Pass data to a graph using a tensor data, a reference will be taken to your data and used just in time when the graph is run.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphtensordata?language=objc)
    #[unsafe(super(GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[name = "MPSGraphTensorData"]
    pub struct TensorData;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for TensorData {}
);

impl TensorData {
    extern_methods!(
        /// The data type of the tensor data.
        #[unsafe(method(dataType))]
        #[unsafe(method_family = none)]
        pub fn data_type(&self) -> DataType;

        /// The device of the tensor data.
        #[unsafe(method(device))]
        #[unsafe(method_family = none)]
        pub fn device(&self) -> Retained<Device>;
    );
}

impl TensorData {
    /// Initializes the tensor data with an `NSData` on a device.
    ///
    /// - Parameters:
    /// - device: Device on which the TensorData exists
    /// - data: NSData from which to copy the contents
    /// - shape: shape of the output tensor
    /// - dataType: dataType of the placeholder tensor
    /// - Returns: A valid TensorData, or nil if allocation failure.
    pub fn new_with_ns_data(
        device: &Device,
        data: &NSData,
        shape: &[usize],
        data_type: DataType,
    ) -> Retained<Self> {
        let class = Self::class();
        let allocated: Allocated<Self> = unsafe { msg_send![class, alloc] };
        unsafe {
            msg_send![
                allocated,
                initWithDevice: device,
                data: data,
                shape: &*ns_number_array_from_slice(shape),
                dataType: data_type
            ]
        }
    }

    /// Initializes the tensor data with a slice on a device.
    ///
    /// - Parameters:
    /// - device: Device on which the TensorData exists
    /// - data: Slice from which to copy the contents
    /// - shape: shape of the output tensor
    /// - dataType: dataType of the placeholder tensor
    /// - Returns: A valid TensorData, or nil if allocation failure.
    pub fn new_with_data<T: Copy>(
        device: &Device,
        data: &[T],
        shape: &[usize],
        data_type: DataType,
    ) -> Retained<Self> {
        autoreleasepool(|_| unsafe {
            let data_size = size_of_val(data);
            let ns_data = NSData::with_bytes(from_raw_parts(data.as_ptr() as *const u8, data_size));
            Self::new_with_ns_data(device, &ns_data, shape, data_type)
        })
    }

    /// Initializes the tensor data with a Metal buffer specifying rowBytes (stride between rows)
    ///
    /// - Parameters:
    /// - buffer: Metal buffer from which to copy the contents
    /// - shape: shape of the output tensor
    /// - dataType: dataType of the placeholder tensor
    /// - rowBytes: rowBytes for the fastest moving dimension, must be larger than or equal to sizeOf(dataType)shape[rank - 1] and must be a multiple of sizeOf(dataType)
    /// - Returns: A valid TensorData, or nil if allocation failure.
    pub fn new_with_mtl_buffer(
        buffer: &Buffer,
        shape: &[usize],
        data_type: DataType,
        row_bytes: Option<u64>,
    ) -> Retained<Self> {
        let class = Self::class();
        let buffer_ptr = buffer.as_ptr() as *mut AnyObject;
        let allocated: Allocated<Self> = unsafe { msg_send![class, alloc] };
        let shape = ns_number_array_from_slice(shape);
        match row_bytes {
            Some(row_bytes) => unsafe {
                msg_send![
                    allocated,
                    initWithMTLBuffer: buffer_ptr,
                    shape: &*shape,
                    dataType: data_type,
                    rowBytes: row_bytes
                ]
            },
            None => unsafe {
                msg_send![
                    allocated,
                    initWithMTLBuffer: buffer_ptr,
                    shape: &*shape,
                    dataType: data_type
                ]
            },
        }
    }

    /// The shape of the tensor data.
    pub fn shape(&self) -> Box<[usize]> {
        let array: Retained<Shape> = unsafe { msg_send![self, shape] };
        ns_number_array_to_boxed_slice(&array)
    }
}
