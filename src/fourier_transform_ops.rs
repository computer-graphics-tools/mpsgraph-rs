use objc2::runtime::AnyObject;
use objc2::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, AsRawObject};

/// Scaling mode for FFT operations
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum MPSGraphFFTScalingMode {
    /// No scaling
    None = 0,
    /// Scale by 1/n
    UniFactor = 1,
    /// Scale by 1/sqrt(n)
    SqrtUniFactor = 2,
}

/// Descriptor for FFT operations
pub struct MPSGraphFFTDescriptor(pub(crate) *mut AnyObject);

impl Default for MPSGraphFFTDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl MPSGraphFFTDescriptor {
    /// Creates a new FFT descriptor with default settings
    pub fn new() -> Self {
        unsafe {
            let class_name = c"MPSGraphFFTDescriptor";
            if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                let descriptor: *mut AnyObject = msg_send![cls, descriptor];
                let descriptor = objc2::ffi::objc_retain(descriptor as *mut _) as *mut AnyObject;
                MPSGraphFFTDescriptor(descriptor)
            } else {
                panic!("Class MPSGraphFFTDescriptor not found")
            }
        }
    }
    
    /// Sets the FFT length
    pub fn set_length(&self, length: usize) {
        unsafe {
            let _: () = msg_send![self.0, setLength: length,];
        }
    }
    
    /// Sets the input batch dimension
    pub fn set_batch_dimension(&self, dimension: usize) {
        unsafe {
            let _: () = msg_send![self.0, setBatchDimension: dimension,];
        }
    }
    
    /// Sets the input transform dimension
    pub fn set_transform_dimension(&self, dimension: usize) {
        unsafe {
            let _: () = msg_send![self.0, setTransformDimension: dimension,];
        }
    }
    
    /// Sets the scaling mode
    pub fn set_scaling_mode(&self, mode: MPSGraphFFTScalingMode) {
        unsafe {
            let _: () = msg_send![self.0, setScalingMode: mode as u64];
        }
    }
    
    /// Gets the FFT length
    pub fn length(&self) -> usize {
        unsafe {
            msg_send![self.0, length]
        }
    }
    
    /// Gets the input batch dimension
    pub fn batch_dimension(&self) -> usize {
        unsafe {
            msg_send![self.0, batchDimension]
        }
    }
    
    /// Gets the input transform dimension
    pub fn transform_dimension(&self) -> usize {
        unsafe {
            msg_send![self.0, transformDimension]
        }
    }
    
    /// Gets the scaling mode
    pub fn scaling_mode(&self) -> MPSGraphFFTScalingMode {
        unsafe {
            let mode: u64 = msg_send![self.0, scalingMode];
            match mode {
                0 => MPSGraphFFTScalingMode::None,
                1 => MPSGraphFFTScalingMode::UniFactor,
                2 => MPSGraphFFTScalingMode::SqrtUniFactor,
                _ => MPSGraphFFTScalingMode::None,
            }
        }
    }
}

impl Drop for MPSGraphFFTDescriptor {
    fn drop(&mut self) {
        unsafe {
            objc2::ffi::objc_release(self.0 as *mut _);
        }
    }
}

impl Clone for MPSGraphFFTDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let desc: *mut AnyObject = msg_send![self.0, copy];
            MPSGraphFFTDescriptor(desc)
        }
    }
}

/// Fourier transform operations for MPSGraph
impl MPSGraph {
    /// Creates a forward FFT operation using complex-valued input.
    ///
    /// # Arguments
    ///
    /// * `real` - Tensor with the real part of the input
    /// * `imaginary` - Tensor with the imaginary part of the input
    /// * `descriptor` - FFT descriptor specifying the transform parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tuple of MPSGraphTensor objects (real_output, imaginary_output).
    pub fn forward_fft(
        &self,
        real: &MPSGraphTensor,
        imaginary: &MPSGraphTensor,
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, forwardFFTWithRealTensor: real.0,
                imaginaryTensor: imaginary.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            // This returns an NSArray with two tensors: real and imaginary parts
            // Extract both tensors from the array
            let count: usize = msg_send![result, count];
            assert_eq!(count, 2, "Expected 2 result tensors from forward FFT");
            
            let real_output: *mut AnyObject = msg_send![result, objectAtIndex: 0];
            let imag_output: *mut AnyObject = msg_send![result, objectAtIndex: 1];
            
            let real_output = objc2::ffi::objc_retain(real_output as *mut _) as *mut AnyObject;
            let imag_output = objc2::ffi::objc_retain(imag_output as *mut _) as *mut AnyObject;
            
            (MPSGraphTensor(real_output), MPSGraphTensor(imag_output))
        }
    }
    
    /// Creates an inverse FFT operation using complex-valued input.
    ///
    /// # Arguments
    ///
    /// * `real` - Tensor with the real part of the input
    /// * `imaginary` - Tensor with the imaginary part of the input
    /// * `descriptor` - FFT descriptor specifying the transform parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tuple of MPSGraphTensor objects (real_output, imaginary_output).
    pub fn inverse_fft(
        &self,
        real: &MPSGraphTensor,
        imaginary: &MPSGraphTensor,
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, inverseFFTWithRealTensor: real.0,
                imaginaryTensor: imaginary.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            // This returns an NSArray with two tensors: real and imaginary parts
            // Extract both tensors from the array
            let count: usize = msg_send![result, count];
            assert_eq!(count, 2, "Expected 2 result tensors from inverse FFT");
            
            let real_output: *mut AnyObject = msg_send![result, objectAtIndex: 0];
            let imag_output: *mut AnyObject = msg_send![result, objectAtIndex: 1];
            
            let real_output = objc2::ffi::objc_retain(real_output as *mut _) as *mut AnyObject;
            let imag_output = objc2::ffi::objc_retain(imag_output as *mut _) as *mut AnyObject;
            
            (MPSGraphTensor(real_output), MPSGraphTensor(imag_output))
        }
    }
    
    /// Creates a forward FFT operation using real-valued input.
    ///
    /// # Arguments
    ///
    /// * `real` - Tensor with the real input values
    /// * `descriptor` - FFT descriptor specifying the transform parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tuple of MPSGraphTensor objects (real_output, imaginary_output).
    pub fn forward_real_fft(
        &self,
        real: &MPSGraphTensor,
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, forwardRealFFTWithRealTensor: real.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            // This returns an NSArray with two tensors: real and imaginary parts
            // Extract both tensors from the array
            let count: usize = msg_send![result, count];
            assert_eq!(count, 2, "Expected 2 result tensors from forward real FFT");
            
            let real_output: *mut AnyObject = msg_send![result, objectAtIndex: 0];
            let imag_output: *mut AnyObject = msg_send![result, objectAtIndex: 1];
            
            let real_output = objc2::ffi::objc_retain(real_output as *mut _) as *mut AnyObject;
            let imag_output = objc2::ffi::objc_retain(imag_output as *mut _) as *mut AnyObject;
            
            (MPSGraphTensor(real_output), MPSGraphTensor(imag_output))
        }
    }
    
    /// Creates an inverse FFT operation that produces real-valued output.
    ///
    /// # Arguments
    ///
    /// * `real` - Tensor with the real part of the input
    /// * `imaginary` - Tensor with the imaginary part of the input
    /// * `descriptor` - FFT descriptor specifying the transform parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// The real-valued output as an MPSGraphTensor object.
    pub fn inverse_real_fft(
        &self,
        real: &MPSGraphTensor,
        imaginary: &MPSGraphTensor,
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, inverseRealFFTWithRealTensor: real.0,
                imaginaryTensor: imaginary.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
}