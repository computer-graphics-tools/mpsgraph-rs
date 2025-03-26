use objc::runtime::{Object, Class, YES, NO};
use objc::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::NSString;

/// The scaling modes for Fourier transform operations.
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum MPSGraphFFTScalingMode {
    /// Computes the FFT result with no scaling.
    None = 0,
    /// Scales the FFT result with reciprocal of the total FFT size over all transformed dimensions.
    Size = 1,
    /// Scales the FFT result with reciprocal square root of the total FFT size over all transformed dimensions, resulting in signal strength conserving transformation.
    Unitary = 2,
}

/// The descriptor for Fast Fourier Transform operations
pub struct MPSGraphFFTDescriptor(pub(crate) *mut Object);

impl MPSGraphFFTDescriptor {
    /// Creates a new FFT descriptor with default parameter values
    pub fn new() -> Self {
        unsafe {
            let cls = Class::get("MPSGraphFFTDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![cls, descriptor];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphFFTDescriptor(descriptor)
        }
    }
    
    /// Set whether the FFT operation is inverse
    ///
    /// When set to `true` graph uses the positive phase factor: `exp(+i 2Pi mu nu / n)`, when computing the (inverse) Fourier transform.
    /// Otherwise MPSGraph uses the negative phase factor: `exp(-i 2Pi mu nu / n)`, when computing the Fourier transform.
    /// Default value: `false`.
    pub fn set_inverse(&self, inverse: bool) {
        unsafe {
            let _: () = msg_send![self.0, setInverse:if inverse { YES } else { NO }];
        }
    }
    
    /// Set the scaling mode for the FFT operation
    ///
    /// The scaling mode is independent from the phase factor.
    /// Default value: `MPSGraphFFTScalingMode::None`.
    pub fn set_scaling_mode(&self, mode: MPSGraphFFTScalingMode) {
        unsafe {
            let _: () = msg_send![self.0, setScalingMode:mode as u64];
        }
    }
    
    /// Set whether to round the output tensor size for a Hermitean-to-real Fourier transform
    ///
    /// If set to `true` then MPSGraph rounds the last output dimension of the result tensor in
    /// Hermitean-to-real FFT to an odd value.
    /// Has no effect in the other Fourier transform operations.
    /// Default value: `false`.
    pub fn set_round_to_odd_hermitean(&self, round: bool) {
        unsafe {
            let _: () = msg_send![self.0, setRoundToOddHermitean:if round { YES } else { NO }];
        }
    }
}

impl Drop for MPSGraphFFTDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

impl Clone for MPSGraphFFTDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let desc: *mut Object = msg_send![self.0, copy];
            MPSGraphFFTDescriptor(desc)
        }
    }
}

/// Helper function to create an NSArray of NSNumbers from a slice of i64s
fn create_ns_array_from_i64s(integers: &[i64]) -> *mut Object {
    unsafe {
        let ns_number_class = Class::get("NSNumber").unwrap();
        let ns_array_class = Class::get("NSArray").unwrap();
        
        let mut ns_numbers: Vec<*mut Object> = Vec::with_capacity(integers.len());
        for &i in integers {
            let ns_number: *mut Object = msg_send![ns_number_class, numberWithLongLong:i];
            ns_numbers.push(ns_number);
        }
        
        let count = ns_numbers.len();
        let ns_array: *mut Object = msg_send![
            ns_array_class,
            arrayWithObjects:ns_numbers.as_ptr()
            count:count
        ];
        ns_array
    }
}

/// Fourier Transform operations for MPSGraph
impl MPSGraph {
    /// Creates a fast Fourier transform operation and returns the result tensor.
    ///
    /// This operation computes the fast Fourier transform of the input tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - A complex or real-valued input tensor.
    /// * `axes` - A slice of numbers that specifies over which axes MPSGraph performs the Fourier transform.
    ///            All axes must be contained within the last four dimensions of the input tensor.
    /// * `descriptor` - A descriptor that defines the parameters of the Fourier transform operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A complex-valued MPSGraphTensor of the same shape as the input tensor.
    pub fn fast_fourier_transform(
        &self,
        tensor: &MPSGraphTensor,
        axes: &[i64],
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        let axes_array = create_ns_array_from_i64s(axes);
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                fastFourierTransformWithTensor:tensor.0
                axes:axes_array
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a fast Fourier transform operation using an axes tensor and returns the result tensor.
    ///
    /// This operation computes the fast Fourier transform of the input tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - A complex or real-valued input tensor.
    /// * `axes_tensor` - A tensor of rank one containing the axes over which MPSGraph performs the transformation.
    /// * `descriptor` - A descriptor that defines the parameters of the Fourier transform operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A complex-valued MPSGraphTensor of the same shape as the input tensor.
    pub fn fast_fourier_transform_with_axes_tensor(
        &self,
        tensor: &MPSGraphTensor,
        axes_tensor: &MPSGraphTensor,
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                fastFourierTransformWithTensor:tensor.0
                axesTensor:axes_tensor.0
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a Real-to-Hermitean fast Fourier transform operation and returns the result tensor.
    ///
    /// This operation computes the fast Fourier transform of a real-valued input tensor.
    /// The result tensor has size `(n/2)+1` in the last dimension defined by `axes`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - A real-valued input tensor. Must have datatype `Float32` or `Float16`.
    /// * `axes` - A slice of numbers that specifies over which axes MPSGraph performs the Fourier transform.
    ///            All axes must be contained within the last four dimensions of the input tensor.
    /// * `descriptor` - A descriptor that defines the parameters of the Fourier transform operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A complex-valued MPSGraphTensor with reduced size in the last transformed dimension.
    pub fn real_to_hermitean_fft(
        &self,
        tensor: &MPSGraphTensor,
        axes: &[i64],
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        let axes_array = create_ns_array_from_i64s(axes);
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                realToHermiteanFFTWithTensor:tensor.0
                axes:axes_array
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a Real-to-Hermitean fast Fourier transform operation using an axes tensor and returns the result tensor.
    ///
    /// This operation computes the fast Fourier transform of a real-valued input tensor.
    /// The result tensor has size `(n/2)+1` in the last dimension defined by `axes`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - A real-valued input tensor. Must have datatype `Float32` or `Float16`.
    /// * `axes_tensor` - A tensor of rank one containing the axes over which MPSGraph performs the transformation.
    /// * `descriptor` - A descriptor that defines the parameters of the Fourier transform operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A complex-valued MPSGraphTensor with reduced size in the last transformed dimension.
    pub fn real_to_hermitean_fft_with_axes_tensor(
        &self,
        tensor: &MPSGraphTensor,
        axes_tensor: &MPSGraphTensor,
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                realToHermiteanFFTWithTensor:tensor.0
                axesTensor:axes_tensor.0
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a Hermitean-to-real fast Fourier transform operation and returns the result tensor.
    ///
    /// This operation computes the fast Fourier transform of a complex-valued input tensor with hermitean symmetry.
    /// The result tensor has size `(inSize-1)*2 + x` in the last dimension defined by `axes`,
    /// where `inSize = shape(input)[axis] ( = (n/2)+1 )` is the size of the input tensor in the last transformed dimension
    /// and `x = 1` when `round_to_odd_hermitean` = `true` and `x = 0` otherwise.
    ///
    /// # Arguments
    ///
    /// * `tensor` - A complex-valued input tensor. Must have datatype `ComplexFloat32` or `ComplexFloat16`.
    /// * `axes` - A slice of numbers that specifies over which axes MPSGraph performs the Fourier transform.
    ///            All axes must be contained within the last four dimensions of the input tensor.
    /// * `descriptor` - A descriptor that defines the parameters of the Fourier transform operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A real-valued MPSGraphTensor with full size.
    pub fn hermitean_to_real_fft(
        &self,
        tensor: &MPSGraphTensor,
        axes: &[i64],
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        let axes_array = create_ns_array_from_i64s(axes);
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                HermiteanToRealFFTWithTensor:tensor.0
                axes:axes_array
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
    
    /// Creates a Hermitean-to-real fast Fourier transform operation using an axes tensor and returns the result tensor.
    ///
    /// This operation computes the fast Fourier transform of a complex-valued input tensor with hermitean symmetry.
    /// The result tensor has size `(inSize-1)*2 + x` in the last dimension defined by `axes`,
    /// where `inSize = shape(input)[axis] ( = (n/2)+1 )` is the size of the input tensor in the last transformed dimension
    /// and `x = 1` when `round_to_odd_hermitean` = `true` and `x = 0` otherwise.
    ///
    /// # Arguments
    ///
    /// * `tensor` - A complex-valued input tensor. Must have datatype `ComplexFloat32` or `ComplexFloat16`.
    /// * `axes_tensor` - A tensor of rank one containing the axes over which MPSGraph performs the transformation.
    /// * `descriptor` - A descriptor that defines the parameters of the Fourier transform operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A real-valued MPSGraphTensor with full size.
    pub fn hermitean_to_real_fft_with_axes_tensor(
        &self,
        tensor: &MPSGraphTensor,
        axes_tensor: &MPSGraphTensor,
        descriptor: &MPSGraphFFTDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut Object = msg_send![
                self.0,
                HermiteanToRealFFTWithTensor:tensor.0
                axesTensor:axes_tensor.0
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let result: *mut Object = msg_send![result, retain];
            MPSGraphTensor(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MPSDataType, tests::should_skip_test};
    use crate::core::MPSShape;
    use std::collections::HashMap;
    
    #[test]
    fn test_fft() {
        if should_skip_test("test_fft") {
            return;
        }
        
        let graph = MPSGraph::new();
        
        // Create a simple 1D signal
        let tensor_shape = MPSShape::from_slice(&[8]);
        let real_part = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let imag_part = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        
        // Combine real and imaginary parts
        let mut complex_data = Vec::with_capacity(real_part.len() * 2);
        for i in 0..real_part.len() {
            complex_data.push(real_part[i]);
            complex_data.push(imag_part[i]);
        }
        
        let tensor = graph.placeholder(&tensor_shape, MPSDataType::Complex32, None);
        
        // Create FFT descriptor with default settings
        let descriptor = MPSGraphFFTDescriptor::new();
        
        // Perform FFT on dimension 0
        let fft_result = graph.fast_fourier_transform(&tensor, &[0], &descriptor, Some("fft"));
        
        // Run the graph
        let mut feeds = HashMap::new();
        feeds.insert(&tensor, crate::MPSGraphTensorData::new(&complex_data, &[8], MPSDataType::Complex32));
        
        let results = graph.run(feeds, &[&fft_result]);
        
        // Get the result data
        let result_data = results[&fft_result].to_vec::<f32>();
        assert_eq!(result_data.len(), 16); // 8 complex numbers (16 floats)
    }
    
    #[test]
    fn test_real_to_hermitean_fft() {
        if should_skip_test("test_real_to_hermitean_fft") {
            return;
        }
        
        let graph = MPSGraph::new();
        
        // Create a simple 1D signal
        let tensor_shape = MPSShape::from_slice(&[8]);
        let real_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        let tensor = graph.placeholder(&tensor_shape, MPSDataType::Float32, None);
        
        // Create FFT descriptor with default settings
        let descriptor = MPSGraphFFTDescriptor::new();
        
        // Perform Real-to-Hermitean FFT on dimension 0
        let fft_result = graph.real_to_hermitean_fft(&tensor, &[0], &descriptor, Some("real_to_hermitean_fft"));
        
        // Run the graph
        let mut feeds = HashMap::new();
        feeds.insert(&tensor, crate::MPSGraphTensorData::new(&real_data, &[8], MPSDataType::Float32));
        
        let results = graph.run(feeds, &[&fft_result]);
        
        // Get the result data
        let result_data = results[&fft_result].to_vec::<f32>();
        assert_eq!(result_data.len(), 10); // 5 complex numbers (10 floats), output size is floor(N/2) + 1
    }
}