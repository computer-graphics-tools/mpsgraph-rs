use objc2::runtime::AnyObject;
use objc2::msg_send;
use crate::tensor::MPSGraphTensor;
use crate::graph::MPSGraph;
use crate::core::{NSString, AsRawObject};

/// Activation functions for RNN operations
#[repr(u64)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MPSGraphRNNActivation {
    /// No activation
    None = 0,
    /// ReLU activation
    ReLU = 1,
    /// TanH activation
    TanH = 2, 
    /// Sigmoid activation
    Sigmoid = 3,
    /// Hard Sigmoid activation
    HardSigmoid = 4,
}

/// Descriptor for single-gate RNN operations
pub struct MPSGraphSingleGateRNNDescriptor(pub(crate) *mut AnyObject);

impl Default for MPSGraphSingleGateRNNDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl MPSGraphSingleGateRNNDescriptor {
    /// Creates a new single-gate RNN descriptor with default settings
    pub fn new() -> Self {
        unsafe {
            let class_name = c"MPSGraphSingleGateRNNDescriptor";
            if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                let descriptor: *mut AnyObject = msg_send![cls, descriptor];
                let descriptor = objc2::ffi::objc_retain(descriptor as *mut _) as *mut AnyObject;
                MPSGraphSingleGateRNNDescriptor(descriptor)
            } else {
                panic!("Class MPSGraphSingleGateRNNDescriptor not found")
            }
        }
    }
    
    /// Sets the input gate activation function
    pub fn set_input_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setInputGateActivation: activation as u64];
        }
    }
    
    /// Sets the bias vector for the RNN
    pub fn set_use_bias_vectors(&self, use_bias: bool) {
        unsafe {
            let _: () = msg_send![self.0, setUseBiasVectors: use_bias];
        }
    }
    
    /// Sets whether to reverse the input sequence
    pub fn set_reverse_input_sequence(&self, reverse: bool) {
        unsafe {
            let _: () = msg_send![self.0, setReverseInputSequence: reverse];
        }
    }
    
    /// Sets the training mode
    pub fn set_training(&self, training: bool) {
        unsafe {
            let _: () = msg_send![self.0, setTraining: training];
        }
    }
}

impl Drop for MPSGraphSingleGateRNNDescriptor {
    fn drop(&mut self) {
        unsafe {
            objc2::ffi::objc_release(self.0 as *mut _);
        }
    }
}

impl Clone for MPSGraphSingleGateRNNDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let desc: *mut AnyObject = msg_send![self.0, copy];
            MPSGraphSingleGateRNNDescriptor(desc)
        }
    }
}

/// Descriptor for LSTM operations
pub struct MPSGraphLSTMDescriptor(pub(crate) *mut AnyObject);

impl Default for MPSGraphLSTMDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl MPSGraphLSTMDescriptor {
    /// Creates a new LSTM descriptor with default settings
    pub fn new() -> Self {
        unsafe {
            let class_name = c"MPSGraphLSTMDescriptor";
            if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                let descriptor: *mut AnyObject = msg_send![cls, descriptor];
                let descriptor = objc2::ffi::objc_retain(descriptor as *mut _) as *mut AnyObject;
                MPSGraphLSTMDescriptor(descriptor)
            } else {
                panic!("Class MPSGraphLSTMDescriptor not found")
            }
        }
    }
    
    /// Sets the input gate activation function
    pub fn set_input_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setInputGateActivation: activation as u64];
        }
    }
    
    /// Sets the forget gate activation function
    pub fn set_forget_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setForgetGateActivation: activation as u64];
        }
    }
    
    /// Sets the cell gate activation function
    pub fn set_cell_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setCellGateActivation: activation as u64];
        }
    }
    
    /// Sets the output gate activation function
    pub fn set_output_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setOutputGateActivation: activation as u64];
        }
    }
    
    /// Sets the bias vector for the LSTM
    pub fn set_use_bias_vectors(&self, use_bias: bool) {
        unsafe {
            let _: () = msg_send![self.0, setUseBiasVectors: use_bias];
        }
    }
    
    /// Sets whether to reverse the input sequence
    pub fn set_reverse_input_sequence(&self, reverse: bool) {
        unsafe {
            let _: () = msg_send![self.0, setReverseInputSequence: reverse];
        }
    }
    
    /// Sets the training mode
    pub fn set_training(&self, training: bool) {
        unsafe {
            let _: () = msg_send![self.0, setTraining: training];
        }
    }
    
    /// Sets whether to use layer norm
    pub fn set_use_layer_norm(&self, use_layer_norm: bool) {
        unsafe {
            let _: () = msg_send![self.0, setUseLayerNorm: use_layer_norm];
        }
    }
}

impl Drop for MPSGraphLSTMDescriptor {
    fn drop(&mut self) {
        unsafe {
            objc2::ffi::objc_release(self.0 as *mut _);
        }
    }
}

impl Clone for MPSGraphLSTMDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let desc: *mut AnyObject = msg_send![self.0, copy];
            MPSGraphLSTMDescriptor(desc)
        }
    }
}

/// Descriptor for GRU operations
pub struct MPSGraphGRUDescriptor(pub(crate) *mut AnyObject);

impl Default for MPSGraphGRUDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl MPSGraphGRUDescriptor {
    /// Creates a new GRU descriptor with default settings
    pub fn new() -> Self {
        unsafe {
            let class_name = c"MPSGraphGRUDescriptor";
            if let Some(cls) = objc2::runtime::AnyClass::get(class_name) {
                let descriptor: *mut AnyObject = msg_send![cls, descriptor];
                let descriptor = objc2::ffi::objc_retain(descriptor as *mut _) as *mut AnyObject;
                MPSGraphGRUDescriptor(descriptor)
            } else {
                panic!("Class MPSGraphGRUDescriptor not found")
            }
        }
    }
    
    /// Sets the reset gate activation function
    pub fn set_reset_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setResetGateActivation: activation as u64];
        }
    }
    
    /// Sets the update gate activation function
    pub fn set_update_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setUpdateGateActivation: activation as u64];
        }
    }
    
    /// Sets the hidden gate activation function
    pub fn set_hidden_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setHiddenGateActivation: activation as u64];
        }
    }
    
    /// Sets the bias vector for the GRU
    pub fn set_use_bias_vectors(&self, use_bias: bool) {
        unsafe {
            let _: () = msg_send![self.0, setUseBiasVectors: use_bias];
        }
    }
    
    /// Sets whether to reverse the input sequence
    pub fn set_reverse_input_sequence(&self, reverse: bool) {
        unsafe {
            let _: () = msg_send![self.0, setReverseInputSequence: reverse];
        }
    }
    
    /// Sets the training mode
    pub fn set_training(&self, training: bool) {
        unsafe {
            let _: () = msg_send![self.0, setTraining: training];
        }
    }
}

impl Drop for MPSGraphGRUDescriptor {
    fn drop(&mut self) {
        unsafe {
            objc2::ffi::objc_release(self.0 as *mut _);
        }
    }
}

impl Clone for MPSGraphGRUDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let desc: *mut AnyObject = msg_send![self.0, copy];
            MPSGraphGRUDescriptor(desc)
        }
    }
}

/// RNN operation for MPSGraph
impl MPSGraph {
    /// Creates a single-gate RNN operation.
    ///
    /// # Arguments
    ///
    /// * `input` - Input sequence tensor of shape [T,N,C] or [N,T,C]
    /// * `initial_state` - Initial hidden state tensor of shape [N,H]
    /// * `weights` - Kernel tensor of shape [C+H,H]
    /// * `recurrent_weights` - Recurrent kernel tensor of shape [H,H]
    /// * `biases` - Bias tensor of shape [H], may be NULL if descriptor.useBiasVectors is false
    /// * `descriptor` - RNN descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// Tuple containing (output tensor of shape [T,N,H] or [N,T,H], output state tensor of shape [N,H])
    pub fn single_gate_rnn(
        &self,
        input:  &MPSGraphTensor,
        initial_state:  &MPSGraphTensor,
        weights:  &MPSGraphTensor,
        recurrent_weights:  &MPSGraphTensor,
        biases:  Option<&MPSGraphTensor>,
        descriptor:  &MPSGraphSingleGateRNNDescriptor,
        name:  Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        let biases_obj = match biases {
            Some(b) => b.0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, singleGateRNNWithSourceTensor: input.0
                recurrentSourceTensor: initial_state.0
                weightsTensor: weights.0
                recurrentWeightsTensor: recurrent_weights.0
                biasesTensor: biases_obj
                descriptor: descriptor.0
                name: name_obj
            ];
            
            // This returns an NSArray with two tensors: output and output_state
            // Extract both tensors from the array
            let count: usize = msg_send![result, count];
            assert_eq!(count, 2, "Expected 2 result tensors from singleGateRNN");
            
            let output_tensor: *mut AnyObject = msg_send![result, objectAtIndex: 0];
            let output_state_tensor: *mut AnyObject = msg_send![result, objectAtIndex: 1];
            
            let output_tensor = objc2::ffi::objc_retain(output_tensor as *mut _) as *mut AnyObject;
            let output_state_tensor = objc2::ffi::objc_retain(output_state_tensor as *mut _) as *mut AnyObject;
            
            (MPSGraphTensor(output_tensor), MPSGraphTensor(output_state_tensor))
        }
    }
    
    /// Creates an LSTM operation.
    ///
    /// # Arguments
    ///
    /// * `input` - Input sequence tensor of shape [T,N,C] or [N,T,C]
    /// * `initial_hidden_state` - Initial hidden state tensor of shape [N,H]
    /// * `initial_cell_state` - Initial cell state tensor of shape [N,H]
    /// * `weights` - Kernel tensor of shape [C+H,4*H]
    /// * `recurrent_weights` - Recurrent kernel tensor of shape [H,4*H]
    /// * `biases` - Bias tensor of shape [4*H], may be NULL if descriptor.useBiasVectors is false
    /// * `descriptor` - LSTM descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// Tuple containing (output tensor of shape [T,N,H] or [N,T,H], output hidden state tensor of shape [N,H], output cell state tensor of shape [N,H])
    pub fn lstm(
        &self,
        input:  &MPSGraphTensor,
        initial_hidden_state:  &MPSGraphTensor,
        initial_cell_state:  &MPSGraphTensor,
        weights:  &MPSGraphTensor,
        recurrent_weights:  &MPSGraphTensor,
        biases:  Option<&MPSGraphTensor>,
        descriptor:  &MPSGraphLSTMDescriptor,
        name:  Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor, MPSGraphTensor) {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        let biases_obj = match biases {
            Some(b) => b.0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, LSTMWithSourceTensor: input.0
                recurrentSourceTensor: initial_hidden_state.0
                cellSourceTensor: initial_cell_state.0
                weightsTensor: weights.0
                recurrentWeightsTensor: recurrent_weights.0
                biasesTensor: biases_obj
                descriptor: descriptor.0
                name: name_obj
            ];
            
            // This returns an NSArray with three tensors: output, output_hidden_state, and output_cell_state
            // Extract all three tensors from the array
            let count: usize = msg_send![result, count];
            assert_eq!(count, 3, "Expected 3 result tensors from LSTM");
            
            let output_tensor: *mut AnyObject = msg_send![result, objectAtIndex: 0];
            let output_hidden_state_tensor: *mut AnyObject = msg_send![result, objectAtIndex: 1];
            let output_cell_state_tensor: *mut AnyObject = msg_send![result, objectAtIndex: 2];
            
            let output_tensor = objc2::ffi::objc_retain(output_tensor as *mut _) as *mut AnyObject;
            let output_hidden_state_tensor = objc2::ffi::objc_retain(output_hidden_state_tensor as *mut _) as *mut AnyObject;
            let output_cell_state_tensor = objc2::ffi::objc_retain(output_cell_state_tensor as *mut _) as *mut AnyObject;
            
            (
                MPSGraphTensor(output_tensor),
                MPSGraphTensor(output_hidden_state_tensor),
                MPSGraphTensor(output_cell_state_tensor)
            )
        }
    }
    
    /// Creates a GRU operation.
    ///
    /// # Arguments
    ///
    /// * `input` - Input sequence tensor of shape [T,N,C] or [N,T,C]
    /// * `initial_state` - Initial hidden state tensor of shape [N,H]
    /// * `weights` - Kernel tensor of shape [C+H,3*H]
    /// * `recurrent_weights` - Recurrent kernel tensor of shape [H,3*H]
    /// * `biases` - Bias tensor of shape [3*H], may be NULL if descriptor.useBiasVectors is false
    /// * `descriptor` - GRU descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// Tuple containing (output tensor of shape [T,N,H] or [N,T,H], output state tensor of shape [N,H])
    pub fn gru(
        &self,
        input:  &MPSGraphTensor,
        initial_state:  &MPSGraphTensor,
        weights:  &MPSGraphTensor,
        recurrent_weights:  &MPSGraphTensor,
        biases:  Option<&MPSGraphTensor>,
        descriptor:  &MPSGraphGRUDescriptor,
        name:  Option<&str>,
    ) -> (MPSGraphTensor, MPSGraphTensor) {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        let biases_obj = match biases {
            Some(b) => b.0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, GRUWithSourceTensor: input.0
                recurrentSourceTensor: initial_state.0
                weightsTensor: weights.0
                recurrentWeightsTensor: recurrent_weights.0
                biasesTensor: biases_obj
                descriptor: descriptor.0
                name: name_obj
            ];
            
            // This returns an NSArray with two tensors: output and output_state
            // Extract both tensors from the array
            let count: usize = msg_send![result, count];
            assert_eq!(count, 2, "Expected 2 result tensors from GRU");
            
            let output_tensor: *mut AnyObject = msg_send![result, objectAtIndex: 0];
            let output_state_tensor: *mut AnyObject = msg_send![result, objectAtIndex: 1];
            
            let output_tensor = objc2::ffi::objc_retain(output_tensor as *mut _) as *mut AnyObject;
            let output_state_tensor = objc2::ffi::objc_retain(output_state_tensor as *mut _) as *mut AnyObject;
            
            (MPSGraphTensor(output_tensor), MPSGraphTensor(output_state_tensor))
        }
    }
}