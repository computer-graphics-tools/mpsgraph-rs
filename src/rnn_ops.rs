use objc::runtime::Object;
use objc::runtime::{YES, NO};
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::NSString;

/// The activation modes for RNN operations
#[repr(u64)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MPSGraphRNNActivation {
    /// Pass through activation (identity)
    None = 0,
    /// ReLU activation
    Relu = 1,
    /// Tanh activation
    Tanh = 2,
    /// Sigmoid activation
    Sigmoid = 3,
    /// Hard sigmoid activation
    HardSigmoid = 4,
}

/// The descriptor class for SingleGateRNN operations
pub struct MPSGraphSingleGateRNNDescriptor(pub(crate) *mut Object);

impl MPSGraphSingleGateRNNDescriptor {
    /// Creates a new SingleGateRNN descriptor with default values
    pub fn new() -> Self {
        unsafe {
            let cls = objc::runtime::Class::get("MPSGraphSingleGateRNNDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![cls, descriptor];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphSingleGateRNNDescriptor(descriptor)
        }
    }
    
    /// Sets whether to reverse the time direction of the input sequence
    pub fn set_reverse(&self, reverse: bool) {
        unsafe {
            let _: () = msg_send![self.0, setReverse: if reverse { YES } else { NO }];
        }
    }
    
    /// Sets whether this is a bidirectional RNN
    pub fn set_bidirectional(&self, bidirectional: bool) {
        unsafe {
            let _: () = msg_send![self.0, setBidirectional: if bidirectional { YES } else { NO }];
        }
    }
    
    /// Sets whether this RNN should support training
    pub fn set_training(&self, training: bool) {
        unsafe {
            let _: () = msg_send![self.0, setTraining: if training { YES } else { NO }];
        }
    }
    
    /// Sets the activation function for this RNN
    pub fn set_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setActivation: activation as u64];
        }
    }
}

impl Drop for MPSGraphSingleGateRNNDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

/// The descriptor class for LSTM operations
pub struct MPSGraphLSTMDescriptor(pub(crate) *mut Object);

impl MPSGraphLSTMDescriptor {
    /// Creates a new LSTM descriptor with default values
    pub fn new() -> Self {
        unsafe {
            let cls = objc::runtime::Class::get("MPSGraphLSTMDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![cls, descriptor];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphLSTMDescriptor(descriptor)
        }
    }
    
    /// Sets whether to reverse the time direction of the input sequence
    pub fn set_reverse(&self, reverse: bool) {
        unsafe {
            let _: () = msg_send![self.0, setReverse: if reverse { YES } else { NO }];
        }
    }
    
    /// Sets whether this is a bidirectional LSTM
    pub fn set_bidirectional(&self, bidirectional: bool) {
        unsafe {
            let _: () = msg_send![self.0, setBidirectional: if bidirectional { YES } else { NO }];
        }
    }
    
    /// Sets whether this LSTM should produce the cell state as output
    pub fn set_produce_cell(&self, produce_cell: bool) {
        unsafe {
            let _: () = msg_send![self.0, setProduceCell: if produce_cell { YES } else { NO }];
        }
    }
    
    /// Sets whether this LSTM should support training
    pub fn set_training(&self, training: bool) {
        unsafe {
            let _: () = msg_send![self.0, setTraining: if training { YES } else { NO }];
        }
    }
    
    /// Sets whether to use the gate ordering [i, z, f, o] instead of default [i, f, z, o]
    pub fn set_forget_gate_last(&self, forget_gate_last: bool) {
        unsafe {
            let _: () = msg_send![self.0, setForgetGateLast: if forget_gate_last { YES } else { NO }];
        }
    }
    
    /// Sets the activation function for the input gate
    pub fn set_input_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setInputGateActivation: activation as u64];
        }
    }
    
    /// Sets the activation function for the forget gate
    pub fn set_forget_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setForgetGateActivation: activation as u64];
        }
    }
    
    /// Sets the activation function for the cell gate
    pub fn set_cell_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setCellGateActivation: activation as u64];
        }
    }
    
    /// Sets the activation function for the output gate
    pub fn set_output_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setOutputGateActivation: activation as u64];
        }
    }
    
    /// Sets the activation function for the current cell value
    pub fn set_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setActivation: activation as u64];
        }
    }
}

impl Drop for MPSGraphLSTMDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

/// The descriptor class for GRU operations
pub struct MPSGraphGRUDescriptor(pub(crate) *mut Object);

impl MPSGraphGRUDescriptor {
    /// Creates a new GRU descriptor with default values
    pub fn new() -> Self {
        unsafe {
            let cls = objc::runtime::Class::get("MPSGraphGRUDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![cls, descriptor];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphGRUDescriptor(descriptor)
        }
    }
    
    /// Sets whether to reverse the time direction of the input sequence
    pub fn set_reverse(&self, reverse: bool) {
        unsafe {
            let _: () = msg_send![self.0, setReverse: if reverse { YES } else { NO }];
        }
    }
    
    /// Sets whether this is a bidirectional GRU
    pub fn set_bidirectional(&self, bidirectional: bool) {
        unsafe {
            let _: () = msg_send![self.0, setBidirectional: if bidirectional { YES } else { NO }];
        }
    }
    
    /// Sets whether this GRU should support training
    pub fn set_training(&self, training: bool) {
        unsafe {
            let _: () = msg_send![self.0, setTraining: if training { YES } else { NO }];
        }
    }
    
    /// Sets whether to use the gate ordering [r, z, o] instead of default [z, r, o]
    pub fn set_reset_gate_first(&self, reset_gate_first: bool) {
        unsafe {
            let _: () = msg_send![self.0, setResetGateFirst: if reset_gate_first { YES } else { NO }];
        }
    }
    
    /// Sets whether to compute the intermediate value as c[t] = (b + (h[t-1] m) R^T) r[t]
    /// instead of c[t] = (h[t-1] r[t] m) R^T
    pub fn set_reset_after(&self, reset_after: bool) {
        unsafe {
            let _: () = msg_send![self.0, setResetAfter: if reset_after { YES } else { NO }];
        }
    }
    
    /// Sets whether to compute the final value as h[t] = z[t] h[t-1] + (1-z[t]) o[t]
    /// instead of h[t] = (1-z[t]) h[t-1] + z[t] o[t]
    pub fn set_flip_z(&self, flip_z: bool) {
        unsafe {
            let _: () = msg_send![self.0, setFlipZ: if flip_z { YES } else { NO }];
        }
    }
    
    /// Sets the activation function for the update gate
    pub fn set_update_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setUpdateGateActivation: activation as u64];
        }
    }
    
    /// Sets the activation function for the reset gate
    pub fn set_reset_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setResetGateActivation: activation as u64];
        }
    }
    
    /// Sets the activation function for the output gate
    pub fn set_output_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setOutputGateActivation: activation as u64];
        }
    }
}

impl Drop for MPSGraphGRUDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

/// RNN operations for MPSGraph
impl MPSGraph {
    /// Creates a single-gate RNN operation.
    ///
    /// This operation returns tensors `h` and optionally `z` that are defined recursively as follows:
    /// ```
    /// for t = 0 to T-1
    ///   z[t] = x[t] W^T + (h[t-1]m) R^T + b
    ///   h[t] = activation(z[t])
    /// ```
    /// where `W` is optional `input_weight`, `R` is `recurrent_weight`, `b` is `bias`, 
    /// `m` is optional `mask`, `x[t]` is `source`, `h[t]` is the first output, 
    /// `z[t]` is the second output (optional) and `h[-1]` is `init_state`.
    ///
    /// # Parameters
    ///
    /// * `source` - Tensor containing the source data with layout [T,N,I]
    /// * `recurrent_weight` - Tensor containing the recurrent weights
    /// * `input_weight` - Optional tensor containing the input weights matrix
    /// * `bias` - Optional tensor containing the bias
    /// * `init_state` - Optional tensor containing the initial state
    /// * `mask` - Optional tensor containing the mask
    /// * `descriptor` - Descriptor that defines parameters for the RNN operation
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector of tensors containing the outputs of the RNN
    pub fn single_gate_rnn(
        &self,
        source: &MPSGraphTensor,
        recurrent_weight: &MPSGraphTensor,
        input_weight: Option<&MPSGraphTensor>,
        bias: Option<&MPSGraphTensor>,
        init_state: Option<&MPSGraphTensor>,
        mask: Option<&MPSGraphTensor>,
        descriptor: &MPSGraphSingleGateRNNDescriptor,
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let input_weight_obj = match input_weight {
                Some(w) => w.0,
                None => std::ptr::null_mut(),
            };
            
            let bias_obj = match bias {
                Some(b) => b.0,
                None => std::ptr::null_mut(),
            };
            
            let init_state_obj = match init_state {
                Some(s) => s.0,
                None => std::ptr::null_mut(),
            };
            
            let mask_obj = match mask {
                Some(m) => m.0,
                None => std::ptr::null_mut(),
            };
            
            let result_array: *mut Object = msg_send![
                self.0, 
                singleGateRNNWithSourceTensor:source.0
                recurrentWeight:recurrent_weight.0
                inputWeight:input_weight_obj
                bias:bias_obj
                initState:init_state_obj
                mask:mask_obj
                descriptor:descriptor.0
                name:name_obj
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut Object = msg_send![result_array, objectAtIndex:i];
                let tensor: *mut Object = msg_send![tensor, retain];
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
    
    /// Creates a simpler version of single-gate RNN operation without input_weight, bias, and mask
    pub fn single_gate_rnn_simple(
        &self,
        source: &MPSGraphTensor,
        recurrent_weight: &MPSGraphTensor,
        init_state: Option<&MPSGraphTensor>,
        descriptor: &MPSGraphSingleGateRNNDescriptor,
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let init_state_obj = match init_state {
                Some(s) => s.0,
                None => std::ptr::null_mut(),
            };
            
            let result_array: *mut Object = msg_send![
                self.0, 
                singleGateRNNWithSourceTensor:source.0
                recurrentWeight:recurrent_weight.0
                initState:init_state_obj
                descriptor:descriptor.0
                name:name_obj
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut Object = msg_send![result_array, objectAtIndex:i];
                let tensor: *mut Object = msg_send![tensor, retain];
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
    
    /// Creates an LSTM operation.
    ///
    /// This operation returns tensors `h` and optionally `c` and optionally `z` that are defined recursively:
    /// ```
    /// for t = 0 to T-1
    ///   z[t] = [i, f, z, o][t] = f( (h[t-1] m) R^T + x'[t] + p c[t-1] )
    ///   x'[t] = x[t] W^T + b
    ///   c[t] = f[t]c[t-1] + i[t]z[t]
    ///   h[t] = o[t]g(c[t])
    /// ```
    ///
    /// # Parameters
    ///
    /// * `source` - Tensor containing the source data with layout [T,N,I]
    /// * `recurrent_weight` - Tensor containing the recurrent weights
    /// * `input_weight` - Optional tensor containing the input weights matrix
    /// * `bias` - Optional tensor containing the bias
    /// * `init_state` - Optional tensor containing the initial hidden state
    /// * `init_cell` - Optional tensor containing the initial cell state
    /// * `mask` - Optional tensor containing the mask
    /// * `peephole` - Optional tensor containing the peephole weights
    /// * `descriptor` - Descriptor that defines parameters for the LSTM operation
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector of tensors containing the outputs of the LSTM, which may include the hidden state,
    /// cell state, and training state depending on the descriptor configuration
    pub fn lstm(
        &self,
        source: &MPSGraphTensor,
        recurrent_weight: &MPSGraphTensor,
        input_weight: Option<&MPSGraphTensor>,
        bias: Option<&MPSGraphTensor>,
        init_state: Option<&MPSGraphTensor>,
        init_cell: Option<&MPSGraphTensor>,
        mask: Option<&MPSGraphTensor>,
        peephole: Option<&MPSGraphTensor>,
        descriptor: &MPSGraphLSTMDescriptor,
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let input_weight_obj = match input_weight {
                Some(w) => w.0,
                None => std::ptr::null_mut(),
            };
            
            let bias_obj = match bias {
                Some(b) => b.0,
                None => std::ptr::null_mut(),
            };
            
            let init_state_obj = match init_state {
                Some(s) => s.0,
                None => std::ptr::null_mut(),
            };
            
            let init_cell_obj = match init_cell {
                Some(c) => c.0,
                None => std::ptr::null_mut(),
            };
            
            let mask_obj = match mask {
                Some(m) => m.0,
                None => std::ptr::null_mut(),
            };
            
            let peephole_obj = match peephole {
                Some(p) => p.0,
                None => std::ptr::null_mut(),
            };
            
            let result_array: *mut Object = msg_send![
                self.0, 
                LSTMWithSourceTensor:source.0
                recurrentWeight:recurrent_weight.0
                inputWeight:input_weight_obj
                bias:bias_obj
                initState:init_state_obj
                initCell:init_cell_obj
                mask:mask_obj
                peephole:peephole_obj
                descriptor:descriptor.0
                name:name_obj
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut Object = msg_send![result_array, objectAtIndex:i];
                let tensor: *mut Object = msg_send![tensor, retain];
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
    
    /// Creates a simpler version of LSTM operation without mask and peephole
    pub fn lstm_simple(
        &self,
        source: &MPSGraphTensor,
        recurrent_weight: &MPSGraphTensor,
        input_weight: Option<&MPSGraphTensor>,
        bias: Option<&MPSGraphTensor>,
        init_state: Option<&MPSGraphTensor>,
        init_cell: Option<&MPSGraphTensor>,
        descriptor: &MPSGraphLSTMDescriptor,
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let input_weight_obj = match input_weight {
                Some(w) => w.0,
                None => std::ptr::null_mut(),
            };
            
            let bias_obj = match bias {
                Some(b) => b.0,
                None => std::ptr::null_mut(),
            };
            
            let init_state_obj = match init_state {
                Some(s) => s.0,
                None => std::ptr::null_mut(),
            };
            
            let init_cell_obj = match init_cell {
                Some(c) => c.0,
                None => std::ptr::null_mut(),
            };
            
            let result_array: *mut Object = msg_send![
                self.0, 
                LSTMWithSourceTensor:source.0
                recurrentWeight:recurrent_weight.0
                inputWeight:input_weight_obj
                bias:bias_obj
                initState:init_state_obj
                initCell:init_cell_obj
                descriptor:descriptor.0
                name:name_obj
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut Object = msg_send![result_array, objectAtIndex:i];
                let tensor: *mut Object = msg_send![tensor, retain];
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
    
    /// Creates a minimal version of LSTM operation with only source, recurrent_weight, init_state and init_cell
    pub fn lstm_minimal(
        &self,
        source: &MPSGraphTensor,
        recurrent_weight: &MPSGraphTensor,
        init_state: Option<&MPSGraphTensor>,
        init_cell: Option<&MPSGraphTensor>,
        descriptor: &MPSGraphLSTMDescriptor,
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let init_state_obj = match init_state {
                Some(s) => s.0,
                None => std::ptr::null_mut(),
            };
            
            let init_cell_obj = match init_cell {
                Some(c) => c.0,
                None => std::ptr::null_mut(),
            };
            
            let result_array: *mut Object = msg_send![
                self.0, 
                LSTMWithSourceTensor:source.0
                recurrentWeight:recurrent_weight.0
                initState:init_state_obj
                initCell:init_cell_obj
                descriptor:descriptor.0
                name:name_obj
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut Object = msg_send![result_array, objectAtIndex:i];
                let tensor: *mut Object = msg_send![tensor, retain];
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
    
    /// Creates a GRU operation.
    ///
    /// # Parameters
    ///
    /// * `source` - Tensor containing the source data with layout [T,N,I]
    /// * `recurrent_weight` - Tensor containing the recurrent weights
    /// * `input_weight` - Optional tensor containing the input weights matrix
    /// * `bias` - Optional tensor containing the bias
    /// * `init_state` - Optional tensor containing the initial hidden state
    /// * `mask` - Optional tensor containing the mask
    /// * `secondary_bias` - Optional tensor containing the secondary bias (used with reset_after=true)
    /// * `descriptor` - Descriptor that defines parameters for the GRU operation
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector of tensors containing the outputs of the GRU
    pub fn gru(
        &self,
        source: &MPSGraphTensor,
        recurrent_weight: &MPSGraphTensor,
        input_weight: Option<&MPSGraphTensor>,
        bias: Option<&MPSGraphTensor>,
        init_state: Option<&MPSGraphTensor>,
        mask: Option<&MPSGraphTensor>,
        secondary_bias: Option<&MPSGraphTensor>,
        descriptor: &MPSGraphGRUDescriptor,
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let input_weight_obj = match input_weight {
                Some(w) => w.0,
                None => std::ptr::null_mut(),
            };
            
            let bias_obj = match bias {
                Some(b) => b.0,
                None => std::ptr::null_mut(),
            };
            
            let init_state_obj = match init_state {
                Some(s) => s.0,
                None => std::ptr::null_mut(),
            };
            
            let mask_obj = match mask {
                Some(m) => m.0,
                None => std::ptr::null_mut(),
            };
            
            let secondary_bias_obj = match secondary_bias {
                Some(b) => b.0,
                None => std::ptr::null_mut(),
            };
            
            let result_array: *mut Object = msg_send![
                self.0, 
                GRUWithSourceTensor:source.0
                recurrentWeight:recurrent_weight.0
                inputWeight:input_weight_obj
                bias:bias_obj
                initState:init_state_obj
                mask:mask_obj
                secondaryBias:secondary_bias_obj
                descriptor:descriptor.0
                name:name_obj
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut Object = msg_send![result_array, objectAtIndex:i];
                let tensor: *mut Object = msg_send![tensor, retain];
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
    
    /// Creates a simpler version of GRU operation without mask and secondary_bias
    pub fn gru_simple(
        &self,
        source: &MPSGraphTensor,
        recurrent_weight: &MPSGraphTensor,
        input_weight: Option<&MPSGraphTensor>,
        bias: Option<&MPSGraphTensor>,
        init_state: Option<&MPSGraphTensor>,
        descriptor: &MPSGraphGRUDescriptor,
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let input_weight_obj = match input_weight {
                Some(w) => w.0,
                None => std::ptr::null_mut(),
            };
            
            let bias_obj = match bias {
                Some(b) => b.0,
                None => std::ptr::null_mut(),
            };
            
            let init_state_obj = match init_state {
                Some(s) => s.0,
                None => std::ptr::null_mut(),
            };
            
            let result_array: *mut Object = msg_send![
                self.0, 
                GRUWithSourceTensor:source.0
                recurrentWeight:recurrent_weight.0
                inputWeight:input_weight_obj
                bias:bias_obj
                initState:init_state_obj
                descriptor:descriptor.0
                name:name_obj
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut Object = msg_send![result_array, objectAtIndex:i];
                let tensor: *mut Object = msg_send![tensor, retain];
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
    
    /// Creates a minimal version of GRU operation with only source, recurrent_weight, input_weight, and bias
    pub fn gru_minimal(
        &self,
        source: &MPSGraphTensor,
        recurrent_weight: &MPSGraphTensor,
        input_weight: Option<&MPSGraphTensor>,
        bias: Option<&MPSGraphTensor>,
        descriptor: &MPSGraphGRUDescriptor,
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).0,
                None => std::ptr::null_mut(),
            };
            
            let input_weight_obj = match input_weight {
                Some(w) => w.0,
                None => std::ptr::null_mut(),
            };
            
            let bias_obj = match bias {
                Some(b) => b.0,
                None => std::ptr::null_mut(),
            };
            
            let result_array: *mut Object = msg_send![
                self.0, 
                GRUWithSourceTensor:source.0
                recurrentWeight:recurrent_weight.0
                inputWeight:input_weight_obj
                bias:bias_obj
                descriptor:descriptor.0
                name:name_obj
            ];
            
            // Get the count of result tensors
            let count: usize = msg_send![result_array, count];
            
            // Convert NSArray to Vec<MPSGraphTensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor: *mut Object = msg_send![result_array, objectAtIndex:i];
                let tensor: *mut Object = msg_send![tensor, retain];
                result.push(MPSGraphTensor(tensor));
            }
            
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MPSDataType;
    use crate::device::MPSGraphDevice;
    use std::collections::HashMap;
    
    #[test]
    fn test_rnn_descriptor() {
        // This test will be skipped in environments without Metal
        if crate::tests::should_skip_test("test_rnn_descriptor") {
            return;
        }
        
        // Create a SingleGateRNN descriptor and set parameters
        let rnn_desc = MPSGraphSingleGateRNNDescriptor::new();
        rnn_desc.set_activation(MPSGraphRNNActivation::Tanh);
        rnn_desc.set_bidirectional(true);
        rnn_desc.set_training(true);
        
        // Create an LSTM descriptor and set parameters
        let lstm_desc = MPSGraphLSTMDescriptor::new();
        lstm_desc.set_bidirectional(true);
        lstm_desc.set_produce_cell(true);
        lstm_desc.set_forget_gate_activation(MPSGraphRNNActivation::Sigmoid);
        
        // Create a GRU descriptor and set parameters
        let gru_desc = MPSGraphGRUDescriptor::new();
        gru_desc.set_reset_after(true);
        gru_desc.set_update_gate_activation(MPSGraphRNNActivation::Sigmoid);
        
        // No assertions needed, just making sure we can create descriptors and set parameters
    }
    
    #[test]
    fn test_lstm() {
        // This test will be skipped in environments without Metal
        if crate::tests::should_skip_test("test_lstm") {
            return;
        }
        
        let graph = MPSGraph::new();
        let device = MPSGraphDevice::new();
        
        // Create a simple LSTM
        // Sequence length: 2, batch size: 1, input size: 4, hidden size: 3
        
        // Create the input tensor with shape [2, 1, 4]
        let input = graph.placeholder_with_shape(&[2, 1, 4], MPSDataType::Float32, Some("input"));
        
        // Create recurrent weights with shape [4*3, 3] (4 gates * hidden_size, hidden_size)
        let recurrent_weight = graph.placeholder_with_shape(&[12, 3], MPSDataType::Float32, Some("recurrent_weight"));
        
        // Create input weights with shape [4*3, 4] (4 gates * hidden_size, input_size)
        let input_weight = graph.placeholder_with_shape(&[12, 4], MPSDataType::Float32, Some("input_weight"));
        
        // Create bias with shape [4*3] (4 gates * hidden_size)
        let bias = graph.placeholder_with_shape(&[12], MPSDataType::Float32, Some("bias"));
        
        // Create the LSTM descriptor
        let lstm_desc = MPSGraphLSTMDescriptor::new();
        lstm_desc.set_produce_cell(true); // Output the cell state
        
        // Create the LSTM operation
        let lstm_outputs = graph.lstm_simple(
            &input,
            &recurrent_weight,
            Some(&input_weight),
            Some(&bias),
            None, // No initial state, defaults to zeros
            None, // No initial cell, defaults to zeros
            &lstm_desc,
            Some("lstm")
        );
        
        // Verify that we got the expected number of outputs
        // With produce_cell=true, we should get two outputs: hidden state and cell state
        assert_eq!(lstm_outputs.len(), 2, "LSTM should produce 2 outputs with produce_cell=true");
        
        // Check the output shapes
        let hidden_state = &lstm_outputs[0];
        let cell_state = &lstm_outputs[1];
        
        // Dimensions should be [sequence_length, batch_size, hidden_size]
        assert_eq!(hidden_state.dimensions(), vec![2, 1, 3], "Hidden state should have shape [2, 1, 3]");
        assert_eq!(cell_state.dimensions(), vec![2, 1, 3], "Cell state should have shape [2, 1, 3]");
        
        // Fill the input tensors with data
        let input_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; // 2x1x4
        let input_tensor_data = crate::tensor_data::MPSGraphTensorData::new(
            &input_data, 
            &[2, 1, 4], 
            MPSDataType::Float32
        );
        
        // Initialize recurrent weights to small random values
        let mut recurrent_data = Vec::with_capacity(12*3);
        for i in 0..36 {
            recurrent_data.push(0.01 * (i as f32));
        }
        let recurrent_tensor_data = crate::tensor_data::MPSGraphTensorData::new(
            &recurrent_data, 
            &[12, 3], 
            MPSDataType::Float32
        );
        
        // Initialize input weights to small random values
        let mut input_weight_data = Vec::with_capacity(12*4);
        for i in 0..48 {
            input_weight_data.push(0.01 * (i as f32));
        }
        let input_weight_tensor_data = crate::tensor_data::MPSGraphTensorData::new(
            &input_weight_data, 
            &[12, 4], 
            MPSDataType::Float32
        );
        
        // Initialize bias to zeros
        let bias_data = vec![0.0f32; 12];
        let bias_tensor_data = crate::tensor_data::MPSGraphTensorData::new(
            &bias_data, 
            &[12], 
            MPSDataType::Float32
        );
        
        // Run the LSTM
        let feeds = HashMap::from([
            (input, input_tensor_data),
            (recurrent_weight, recurrent_tensor_data),
            (input_weight, input_weight_tensor_data),
            (bias, bias_tensor_data),
        ]);
        
        let results = graph.run_with_feeds(
            &feeds, 
            &[hidden_state.clone(), cell_state.clone()], 
            &device, 
            None
        );
        
        // Get the results
        let hidden_output = results.get(hidden_state).unwrap().to_vec::<f32>();
        let cell_output = results.get(cell_state).unwrap().to_vec::<f32>();
        
        // Verify output shapes
        assert_eq!(hidden_output.len(), 6, "Hidden output should have 6 elements (2x1x3)");
        assert_eq!(cell_output.len(), 6, "Cell output should have 6 elements (2x1x3)");
        
        // We can't easily verify the exact values without reimplementing the LSTM,
        // but we can verify that the outputs are non-zero since we provided non-zero inputs
        let hidden_sum: f32 = hidden_output.iter().sum();
        let cell_sum: f32 = cell_output.iter().sum();
        
        assert!(hidden_sum != 0.0, "Hidden output should be non-zero");
        assert!(cell_sum != 0.0, "Cell output should be non-zero");
    }
}