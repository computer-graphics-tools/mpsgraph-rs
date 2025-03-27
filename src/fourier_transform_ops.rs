use objc2::runtime::AnyObject;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSNumber};
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, AsRawObject};

/// Scaling mode for FFT operations
///
/// Available since macOS 14.0, iOS 17.0, tvOS 17.0
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum MPSGraphFFTScalingMode {
    /// No scaling
    None = 0,
    /// Scale by reciprocal of total FFT size
    Size = 1,
    /// Scale by reciprocal square root of total FFT size
    Unitary = 2,
}

/// Descriptor for FFT operations
///
/// Available since macOS 14.0, iOS 17.0, tvOS 17.0
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
    
    /// Sets whether to use inverse FFT (positive phase factor)
    pub fn set_inverse(&self, inverse: bool) {
        unsafe {
            let _: () = msg_send![self.0, setInverse: inverse];
        }
    }

    /// Gets whether inverse FFT is used
    pub fn inverse(&self) -> bool {
        unsafe {
            msg_send![self.0, inverse]
        }
    }
    
    /// Sets the scaling mode
    pub fn set_scaling_mode(&self, mode: MPSGraphFFTScalingMode) {
        unsafe {
            let _: () = msg_send![self.0, setScalingMode: mode as u64];
        }
    }
    
    /// Gets the scaling mode
    pub fn scaling_mode(&self) -> MPSGraphFFTScalingMode {
        unsafe {
            let mode: u64 = msg_send![self.0, scalingMode];
            match mode {
                0 => MPSGraphFFTScalingMode::None,
                1 => MPSGraphFFTScalingMode::Size,
                2 => MPSGraphFFTScalingMode::Unitary,
                _ => MPSGraphFFTScalingMode::None,
            }
        }
    }
    
    /// Sets whether to round to odd Hermitean output dimensions
    pub fn set_round_to_odd_hermitean(&self, round: bool) {
        unsafe {
            let _: () = msg_send![self.0, setRoundToOddHermitean: round];
        }
    }

    /// Gets whether to round to odd Hermitean output dimensions
    pub fn round_to_odd_hermitean(&self) -> bool {
        unsafe {
            msg_send![self.0, roundToOddHermitean]
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
    /// Creates a fast Fourier transform operation
    ///
    /// # Arguments
    ///
    /// * `tensor` - Tensor to transform (complex tensor)
    /// * `axes` - Axes along which to apply the transform
    /// * `descriptor` - FFT descriptor specifying parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    /// 
    /// A transformed tensor
    ///
    /// # Availability
    ///
    /// Available since macOS 14.0, iOS 17.0, tvOS 17.0
    pub fn fast_fourier_transform(
        &self,
        tensor: &MPSGraphTensor,
        axes: &[u64],
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        // Convert axes to NSArray
        let axes_numbers: Vec<Retained<NSNumber>> = axes.iter()
            .map(|&x| NSNumber::new_u64(x))
            .collect();
        
        let refs: Vec<&NSNumber> = axes_numbers.iter().map(|n| n.as_ref()).collect();
        let ns_array = NSArray::from_slice(&refs);
        let axes_array = ns_array.as_raw_object();
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, fastFourierTransformWithTensor: tensor.0,
                axes: axes_array,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a fast Fourier transform operation with axes specified by a tensor
    ///
    /// # Arguments
    ///
    /// * `tensor` - Tensor to transform (complex tensor)
    /// * `axes_tensor` - Tensor containing the axes along which to apply the transform
    /// * `descriptor` - FFT descriptor specifying parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    /// 
    /// A transformed tensor
    ///
    /// # Availability
    ///
    /// Available since macOS 14.0, iOS 17.0, tvOS 17.0
    pub fn fast_fourier_transform_with_tensor_axes(
        &self,
        tensor: &MPSGraphTensor,
        axes_tensor: &MPSGraphTensor,
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, fastFourierTransformWithTensor: tensor.0,
                axesTensor: axes_tensor.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a real-to-Hermitean fast Fourier transform operation
    ///
    /// # Arguments
    ///
    /// * `tensor` - Real tensor to transform
    /// * `axes` - Axes along which to apply the transform
    /// * `descriptor` - FFT descriptor specifying parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    /// 
    /// A complex tensor in Hermitean format
    ///
    /// # Availability
    ///
    /// Available since macOS 14.0, iOS 17.0, tvOS 17.0
    pub fn real_to_hermitean_fft(
        &self,
        tensor: &MPSGraphTensor,
        axes: &[u64],
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        // Convert axes to NSArray
        let axes_numbers: Vec<Retained<NSNumber>> = axes.iter()
            .map(|&x| NSNumber::new_u64(x))
            .collect();
        
        let refs: Vec<&NSNumber> = axes_numbers.iter().map(|n| n.as_ref()).collect();
        let ns_array = NSArray::from_slice(&refs);
        let axes_array = ns_array.as_raw_object();
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, realToHermiteanFFTWithTensor: tensor.0,
                axes: axes_array,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a real-to-Hermitean fast Fourier transform operation with axes specified by a tensor
    ///
    /// # Arguments
    ///
    /// * `tensor` - Real tensor to transform
    /// * `axes_tensor` - Tensor containing the axes along which to apply the transform
    /// * `descriptor` - FFT descriptor specifying parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    /// 
    /// A complex tensor in Hermitean format
    ///
    /// # Availability
    ///
    /// Available since macOS 14.0, iOS 17.0, tvOS 17.0
    pub fn real_to_hermitean_fft_with_tensor_axes(
        &self,
        tensor: &MPSGraphTensor,
        axes_tensor: &MPSGraphTensor,
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, realToHermiteanFFTWithTensor: tensor.0,
                axesTensor: axes_tensor.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Hermitean-to-real fast Fourier transform operation
    ///
    /// # Arguments
    ///
    /// * `tensor` - Complex tensor in Hermitean format to transform
    /// * `axes` - Axes along which to apply the transform
    /// * `descriptor` - FFT descriptor specifying parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    /// 
    /// A real tensor
    ///
    /// # Availability
    ///
    /// Available since macOS 14.0, iOS 17.0, tvOS 17.0
    pub fn hermitean_to_real_fft(
        &self,
        tensor: &MPSGraphTensor,
        axes: &[u64],
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        // Convert axes to NSArray
        let axes_numbers: Vec<Retained<NSNumber>> = axes.iter()
            .map(|&x| NSNumber::new_u64(x))
            .collect();
        
        let refs: Vec<&NSNumber> = axes_numbers.iter().map(|n| n.as_ref()).collect();
        let ns_array = NSArray::from_slice(&refs);
        let axes_array = ns_array.as_raw_object();
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, HermiteanToRealFFTWithTensor: tensor.0,
                axes: axes_array,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a Hermitean-to-real fast Fourier transform operation with axes specified by a tensor
    ///
    /// # Arguments
    ///
    /// * `tensor` - Complex tensor in Hermitean format to transform
    /// * `axes_tensor` - Tensor containing the axes along which to apply the transform
    /// * `descriptor` - FFT descriptor specifying parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    /// 
    /// A real tensor
    ///
    /// # Availability
    ///
    /// Available since macOS 14.0, iOS 17.0, tvOS 17.0
    pub fn hermitean_to_real_fft_with_tensor_axes(
        &self,
        tensor: &MPSGraphTensor,
        axes_tensor: &MPSGraphTensor,
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, HermiteanToRealFFTWithTensor: tensor.0,
                axesTensor: axes_tensor.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
            MPSGraphTensor(tensor)
        }
    }
    
    // Keep legacy methods for backward compatibility
    
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
    ///
    /// # Deprecated
    ///
    /// This method uses the older API. Consider using `fast_fourier_transform` instead.
    #[deprecated(since = "0.1.0", note = "Use `fast_fourier_transform` instead")]
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
    ///
    /// # Deprecated
    ///
    /// This method uses the older API. Consider using `fast_fourier_transform` instead
    /// with `descriptor.set_inverse(true)`.
    #[deprecated(since = "0.1.0", note = "Use `fast_fourier_transform` with `descriptor.set_inverse(true)` instead")]
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
    ///
    /// # Deprecated
    ///
    /// This method uses the older API. Consider using `real_to_hermitean_fft` instead.
    #[deprecated(since = "0.1.0", note = "Use `real_to_hermitean_fft` instead")]
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
    ///
    /// # Deprecated
    ///
    /// This method uses the older API. Consider using `hermitean_to_real_fft` instead.
    #[deprecated(since = "0.1.0", note = "Use `hermitean_to_real_fft` instead")]
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