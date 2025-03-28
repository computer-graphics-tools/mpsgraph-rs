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
    
    /// A parameter that defines time direction of the input sequence.
    ///
    /// If set to `true` then the input sequence is passed in reverse time order to the layer.
    /// Note: Ignored when `bidirectional = true`.
    /// Default value: `false`.
    pub fn set_reverse(&self, reverse: bool) {
        unsafe {
            let _: () = msg_send![self.0, setReverse: reverse];
        }
    }
    
    /// A parameter that defines a bidirectional RNN layer.
    ///
    /// If set to `true` then the input sequence is traversed in both directions and the two results
    /// are concatenated together on the channel-axis.
    /// Default value: `false`.
    pub fn set_bidirectional(&self, bidirectional: bool) {
        unsafe {
            let _: () = msg_send![self.0, setBidirectional: bidirectional];
        }
    }
    
    /// A parameter that makes the RNN layer support training.
    ///
    /// If set to `true` then the layer will produce training state tensor as a secondary output.
    /// Default value: `false`.
    pub fn set_training(&self, training: bool) {
        unsafe {
            let _: () = msg_send![self.0, setTraining: training];
        }
    }
    
    /// A parameter that defines the activation function to use with the RNN operation.
    /// 
    /// Default value: `MPSGraphRNNActivation::ReLU`.
    pub fn set_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setActivation: activation as u64];
        }
    }
    
    /// Returns the reverse setting.
    pub fn reverse(&self) -> bool {
        unsafe {
            msg_send![self.0, reverse]
        }
    }
    
    /// Returns the bidirectional setting.
    pub fn bidirectional(&self) -> bool {
        unsafe {
            msg_send![self.0, bidirectional]
        }
    }
    
    /// Returns the training setting.
    pub fn training(&self) -> bool {
        unsafe {
            msg_send![self.0, training]
        }
    }
    
    /// Returns the activation function setting.
    pub fn activation(&self) -> MPSGraphRNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self.0, activation];
            std::mem::transmute(activation_val)
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
    
    /// A parameter that defines time direction of the input sequence.
    ///
    /// If set to `true` then the input sequence is passed in reverse time order to the layer.
    /// Note: Ignored when `bidirectional = true`.
    /// Default value: `false`.
    pub fn set_reverse(&self, reverse: bool) {
        unsafe {
            let _: () = msg_send![self.0, setReverse: reverse];
        }
    }
    
    /// A parameter that defines a bidirectional LSTM layer.
    ///
    /// If set to `true` then the input sequence is traversed in both directions and the two results
    /// are concatenated together on the channel-axis.
    /// Default value: `false`.
    pub fn set_bidirectional(&self, bidirectional: bool) {
        unsafe {
            let _: () = msg_send![self.0, setBidirectional: bidirectional];
        }
    }
    
    /// A parameter that controls whether or not to return the output cell from the LSTM layer.
    ///
    /// If set to `true` then this layer will produce the internal cell of the LSTM unit as secondary output.
    /// Default value: `false`.
    pub fn set_produce_cell(&self, produce_cell: bool) {
        unsafe {
            let _: () = msg_send![self.0, setProduceCell: produce_cell];
        }
    }
    
    /// A parameter that enables the LSTM layer to support training.
    ///
    /// If set to `true` then the layer will produce training state tensor as a secondary output.
    /// Default value: `false`.
    pub fn set_training(&self, training: bool) {
        unsafe {
            let _: () = msg_send![self.0, setTraining: training];
        }
    }
    
    /// A parameter that controls the internal order of the LSTM gates.
    ///
    /// If set to `true` then the layer will use the gate-ordering `[ i, z, f, o ]` instead of default `[ i, f, z, o ]`.
    /// Default value: `false`
    pub fn set_forget_gate_last(&self, forget_gate_last: bool) {
        unsafe {
            let _: () = msg_send![self.0, setForgetGateLast: forget_gate_last];
        }
    }
    
    /// A parameter that defines the activation function used with the input gate of the LSTM operation.
    /// 
    /// Default value: `MPSGraphRNNActivation::Sigmoid`.
    pub fn set_input_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setInputGateActivation: activation as u64];
        }
    }
    
    /// A parameter that defines the activation function used with the forget gate of the LSTM operation.
    /// 
    /// Default value: `MPSGraphRNNActivation::Sigmoid`.
    pub fn set_forget_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setForgetGateActivation: activation as u64];
        }
    }
    
    /// A parameter that defines the activation function used with the cell gate of the LSTM operation.
    /// 
    /// Default value: `MPSGraphRNNActivation::TanH`.
    pub fn set_cell_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setCellGateActivation: activation as u64];
        }
    }
    
    /// A parameter that defines the activation function used with the output gate of the LSTM operation.
    /// 
    /// Default value: `MPSGraphRNNActivation::Sigmoid`.
    pub fn set_output_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setOutputGateActivation: activation as u64];
        }
    }
    
    /// A parameter that defines the activation function used with the current cell value of the LSTM operation.
    /// 
    /// Default value: `MPSGraphRNNActivation::TanH`.
    pub fn set_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setActivation: activation as u64];
        }
    }
    
    // Getter methods
    
    /// Returns the reverse setting.
    pub fn reverse(&self) -> bool {
        unsafe {
            msg_send![self.0, reverse]
        }
    }
    
    /// Returns the bidirectional setting.
    pub fn bidirectional(&self) -> bool {
        unsafe {
            msg_send![self.0, bidirectional]
        }
    }
    
    /// Returns the produce cell setting.
    pub fn produce_cell(&self) -> bool {
        unsafe {
            msg_send![self.0, produceCell]
        }
    }
    
    /// Returns the training setting.
    pub fn training(&self) -> bool {
        unsafe {
            msg_send![self.0, training]
        }
    }
    
    /// Returns the forget gate last setting.
    pub fn forget_gate_last(&self) -> bool {
        unsafe {
            msg_send![self.0, forgetGateLast]
        }
    }
    
    /// Returns the input gate activation setting.
    pub fn input_gate_activation(&self) -> MPSGraphRNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self.0, inputGateActivation];
            std::mem::transmute(activation_val)
        }
    }
    
    /// Returns the forget gate activation setting.
    pub fn forget_gate_activation(&self) -> MPSGraphRNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self.0, forgetGateActivation];
            std::mem::transmute(activation_val)
        }
    }
    
    /// Returns the cell gate activation setting.
    pub fn cell_gate_activation(&self) -> MPSGraphRNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self.0, cellGateActivation];
            std::mem::transmute(activation_val)
        }
    }
    
    /// Returns the output gate activation setting.
    pub fn output_gate_activation(&self) -> MPSGraphRNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self.0, outputGateActivation];
            std::mem::transmute(activation_val)
        }
    }
    
    /// Returns the activation setting.
    pub fn activation(&self) -> MPSGraphRNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self.0, activation];
            std::mem::transmute(activation_val)
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
    
    /// A parameter that defines the time direction of the input sequence.
    ///
    /// If set to `true` then the input sequence is passed in reverse time order to the layer.
    /// Note: Ignored when `bidirectional = true`.
    /// Default value: `false`.
    pub fn set_reverse(&self, reverse: bool) {
        unsafe {
            let _: () = msg_send![self.0, setReverse: reverse];
        }
    }
    
    /// A parameter that defines a bidirectional GRU layer.
    ///
    /// If set to `true` then the input sequence is traversed in both directions and the two results
    /// are concatenated together on the channel-axis.
    /// Default value: `false`.
    pub fn set_bidirectional(&self, bidirectional: bool) {
        unsafe {
            let _: () = msg_send![self.0, setBidirectional: bidirectional];
        }
    }
    
    /// A parameter that enables the GRU layer to support training.
    ///
    /// If set to `true` then the layer will produce training state tensor as a secondary output.
    /// Default value: `false`.
    pub fn set_training(&self, training: bool) {
        unsafe {
            let _: () = msg_send![self.0, setTraining: training];
        }
    }
    
    /// A parameter that controls the internal order of the GRU gates.
    ///
    /// If set to `true` then the layer will use the gate-ordering `[ r, z, o ]` instead of default `[ z, r, o ]`.
    /// Default value: `false`.
    pub fn set_reset_gate_first(&self, reset_gate_first: bool) {
        unsafe {
            let _: () = msg_send![self.0, setResetGateFirst: reset_gate_first];
        }
    }
    
    /// A parameter that chooses between two variants for the reset gate computation.
    ///
    /// If set to `true` then the layer will compute the intermediate value as `c[t] = ( b + (h[t-1] m ) R^T) r[t]`.
    /// Otherwise it's computed as `c[t] = (h[t-1] r[t] m) R^T`.
    /// Default value: `false`.
    pub fn set_reset_after(&self, reset_after: bool) {
        unsafe {
            let _: () = msg_send![self.0, setResetAfter: reset_after];
        }
    }
    
    /// A parameter that chooses between two variants for the final output computation.
    ///
    /// If set to `true` then the layer will compute the final value as `h[t] = z[t] h[t-1] + (1-z[t]) o[t]`.
    /// Otherwise it's computed as `h[t] = (1-z[t]) h[t-1] + z[t] o[t]`.
    /// Default value: `false`.
    pub fn set_flip_z(&self, flip_z: bool) {
        unsafe {
            let _: () = msg_send![self.0, setFlipZ: flip_z];
        }
    }
    
    /// A parameter that defines the activation function to use with the update-gate of the GRU operation.
    ///
    /// Default value: `MPSGraphRNNActivation::Sigmoid`.
    pub fn set_update_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setUpdateGateActivation: activation as u64];
        }
    }
    
    /// A parameter that defines the activation function to use with the reset-gate of the GRU operation.
    ///
    /// Default value: `MPSGraphRNNActivation::Sigmoid`.
    pub fn set_reset_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setResetGateActivation: activation as u64];
        }
    }
    
    /// A parameter that defines the activation function to use with the output-gate of the GRU operation.
    ///
    /// Default value: `MPSGraphRNNActivation::TanH`.
    pub fn set_output_gate_activation(&self, activation: MPSGraphRNNActivation) {
        unsafe {
            let _: () = msg_send![self.0, setOutputGateActivation: activation as u64];
        }
    }
    
    // Getter methods
    
    /// Returns the reverse setting.
    pub fn reverse(&self) -> bool {
        unsafe {
            msg_send![self.0, reverse]
        }
    }
    
    /// Returns the bidirectional setting.
    pub fn bidirectional(&self) -> bool {
        unsafe {
            msg_send![self.0, bidirectional]
        }
    }
    
    /// Returns the training setting.
    pub fn training(&self) -> bool {
        unsafe {
            msg_send![self.0, training]
        }
    }
    
    /// Returns the reset gate first setting.
    pub fn reset_gate_first(&self) -> bool {
        unsafe {
            msg_send![self.0, resetGateFirst]
        }
    }
    
    /// Returns the reset after setting.
    pub fn reset_after(&self) -> bool {
        unsafe {
            msg_send![self.0, resetAfter]
        }
    }
    
    /// Returns the flip Z setting.
    pub fn flip_z(&self) -> bool {
        unsafe {
            msg_send![self.0, flipZ]
        }
    }
    
    /// Returns the update gate activation setting.
    pub fn update_gate_activation(&self) -> MPSGraphRNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self.0, updateGateActivation];
            std::mem::transmute(activation_val)
        }
    }
    
    /// Returns the reset gate activation setting.
    pub fn reset_gate_activation(&self) -> MPSGraphRNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self.0, resetGateActivation];
            std::mem::transmute(activation_val)
        }
    }
    
    /// Returns the output gate activation setting.
    pub fn output_gate_activation(&self) -> MPSGraphRNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self.0, outputGateActivation];
            std::mem::transmute(activation_val)
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
    /// Creates a single-gate RNN operation with mask support.
    ///
    /// This operation returns tensors `h` and optionally `z` that are defined recursively as follows:
    /// ```
    /// for t = 0 to T-1
    ///   z[t] = x[t] W^T + (h[t-1]m) R^T + b
    ///   h[t] = activation( z[t] ), where
    /// ```
    /// `W` is input_weight, `R` is recurrent_weight, `b` is bias, `m` is mask,
    /// `x[t]` is input, `h[t]` is the first output, `z[t]` is the second output (optional) and `h[-1]` is init_state.
    ///
    /// # Arguments
    ///
    /// * `input` - A tensor that contains the source data `x[t]` with the data layout [T,N,I].
    ///             In case `input_weight = None` and `bidirectional = false` then the layout is [T,N,H] and
    ///             for `input_weight = None` and `bidirectional = true` the layout is [T,N,2H].
    /// * `recurrent_weight` - A tensor containing the recurrent weights `R`. For `bidirectional` the layout is [2,H,H] and otherwise it is [H,H].
    /// * `input_weight` - A tensor containing the input weights matrix `W` - optional, if missing the operation assumes a diagonal unit-matrix.
    ///                  For `bidirectional` the layout is [2H,I] and otherwise it is [H,I].
    /// * `bias` - A tensor containing the bias `b` - optional, if missing the operation assumes zeroes. For `bidirectional` the layout is [2H] and otherwise it is [H].
    /// * `init_state` - The initial internal state of the RNN `h[-1]` - optional, if missing the operation assumes zeroes. For `bidirectional` the layout is [N,2H] and otherwise it is [N,H].
    /// * `mask` - A tensor containing the mask `m` - optional, if missing the operation assumes ones. This is useful for dropout support.
    /// * `descriptor` - A descriptor that defines the parameters for the RNN operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A vector of MPSGraphTensor objects of size 1 or 2, depending on value of `descriptor.training`.
    /// The layout of both outputs is [T,N,H] or [T,N,2H] for bidirectional.
    pub fn single_gate_rnn_with_mask(
        &self,
        input: &MPSGraphTensor,
        recurrent_weight: &MPSGraphTensor,
        input_weight: Option<&MPSGraphTensor>,
        bias: Option<&MPSGraphTensor>,
        init_state: Option<&MPSGraphTensor>,
        mask: Option<&MPSGraphTensor>,
        descriptor: &MPSGraphSingleGateRNNDescriptor,
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
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
        
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, singleGateRNNWithSourceTensor: input.0,
                recurrentWeight: recurrent_weight.0,
                inputWeight: input_weight_obj,
                bias: bias_obj,
                initState: init_state_obj,
                mask: mask_obj,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            // Count the number of result tensors (can be 1 or 2 depending on training flag)
            let count: usize = msg_send![result, count];
            let mut tensors = Vec::with_capacity(count);
            
            // Extract all tensors from the array
            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![result, objectAtIndex: i];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
                tensors.push(MPSGraphTensor(tensor));
            }
            
            objc2::ffi::objc_release(result as *mut _);
            tensors
        }
    }
    
    /// Creates a single-gate RNN operation without mask support.
    ///
    /// This operation returns tensors `h` and optionally `z` that are defined recursively as follows:
    /// ```
    /// for t = 0 to T-1
    ///   z[t] = x[t] W^T + (h[t-1]) R^T + b
    ///   h[t] = activation( z[t] ), where
    /// ```
    /// `W` is input_weight, `R` is recurrent_weight, `b` is bias,
    /// `x[t]` is input, `h[t]` is the first output, `z[t]` is the second output (optional) and `h[-1]` is init_state.
    ///
    /// # Arguments
    ///
    /// * `input` - A tensor that contains the source data `x[t]` with the data layout [T,N,I].
    ///             In case `input_weight = None` and `bidirectional = false` then the layout is [T,N,H] and
    ///             for `input_weight = None` and `bidirectional = true` the layout is [T,N,2H].
    /// * `recurrent_weight` - A tensor containing the recurrent weights `R`. For `bidirectional` the layout is [2,H,H] and otherwise it is [H,H].
    /// * `input_weight` - A tensor containing the input weights matrix `W` - optional, if missing the operation assumes a diagonal unit-matrix.
    ///                  For `bidirectional` the layout is [2H,I] and otherwise it is [H,I].
    /// * `bias` - A tensor containing the bias `b` - optional, if missing the operation assumes zeroes. For `bidirectional` the layout is [2H] and otherwise it is [H].
    /// * `init_state` - The initial internal state of the RNN `h[-1]` - optional, if missing the operation assumes zeroes. For `bidirectional` the layout is [N,2H] and otherwise it is [N,H].
    /// * `descriptor` - A descriptor that defines the parameters for the RNN operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A vector of MPSGraphTensor objects of size 1 or 2, depending on value of `descriptor.training`.
    /// The layout of both outputs is [T,N,H] or [T,N,2H] for bidirectional.
    pub fn single_gate_rnn(
        &self,
        input: &MPSGraphTensor,
        recurrent_weight: &MPSGraphTensor,
        input_weight: Option<&MPSGraphTensor>,
        bias: Option<&MPSGraphTensor>,
        init_state: Option<&MPSGraphTensor>,
        descriptor: &MPSGraphSingleGateRNNDescriptor,
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
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
        
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, singleGateRNNWithSourceTensor: input.0,
                recurrentWeight: recurrent_weight.0,
                inputWeight: input_weight_obj,
                bias: bias_obj,
                initState: init_state_obj,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            // Count the number of result tensors (can be 1 or 2 depending on training flag)
            let count: usize = msg_send![result, count];
            let mut tensors = Vec::with_capacity(count);
            
            // Extract all tensors from the array
            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![result, objectAtIndex: i];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
                tensors.push(MPSGraphTensor(tensor));
            }
            
            objc2::ffi::objc_release(result as *mut _);
            tensors
        }
    }
    
    /// Creates a single-gate RNN operation with minimal parameters.
    ///
    /// This operation returns tensors `h` and optionally `z` that are defined recursively as follows:
    /// ```
    /// for t = 0 to T-1
    ///   z[t] = x[t] R^T
    ///   h[t] = activation( z[t] ), where
    /// ```
    /// `R` is recurrent_weight, `x[t]` is input, `h[t]` is the first output, 
    /// `z[t]` is the second output (optional) and `h[-1]` is init_state.
    ///
    /// # Arguments
    ///
    /// * `input` - A tensor that contains the source data `x[t]` with the data layout [T,N,H] or [T,N,2H] for bidirectional.
    /// * `recurrent_weight` - A tensor containing the recurrent weights `R`. For `bidirectional` the layout is [2,H,H] and otherwise it is [H,H].
    /// * `init_state` - The initial internal state of the RNN `h[-1]` - optional, if missing the operation assumes zeroes. For `bidirectional` the layout is [N,2H] and otherwise it is [N,H].
    /// * `descriptor` - A descriptor that defines the parameters for the RNN operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A vector of MPSGraphTensor objects of size 1 or 2, depending on value of `descriptor.training`.
    /// The layout of both outputs is [T,N,H] or [T,N,2H] for bidirectional.
    pub fn single_gate_rnn_minimal(
        &self,
        input: &MPSGraphTensor,
        recurrent_weight: &MPSGraphTensor,
        init_state: Option<&MPSGraphTensor>,
        descriptor: &MPSGraphSingleGateRNNDescriptor,
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        let init_state_obj = match init_state {
            Some(s) => s.0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, singleGateRNNWithSourceTensor: input.0,
                recurrentWeight: recurrent_weight.0,
                initState: init_state_obj,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            // Count the number of result tensors (can be 1 or 2 depending on training flag)
            let count: usize = msg_send![result, count];
            let mut tensors = Vec::with_capacity(count);
            
            // Extract all tensors from the array
            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![result, objectAtIndex: i];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
                tensors.push(MPSGraphTensor(tensor));
            }
            
            objc2::ffi::objc_release(result as *mut _);
            tensors
        }
    }
    
    /// Creates a single-gate RNN gradient operation with all parameters.
    ///
    /// # Arguments
    ///
    /// * `input` - A tensor that contains the source data `x[t]` with the data layout [T,N,I].
    /// * `recurrent_weight` - A tensor containing the recurrent weights `R`. For `bidirectional` the layout is [2,H,H] and otherwise it is [H,H].
    /// * `source_gradient` - The input gradient, that is the gradient of a tensor with respect to the first output of the forward pass.
    /// * `z_state` - The second output of the forward pass with `descriptor.training = true`.
    /// * `state_gradient` - The input gradient coming from the future timestep - optional, if missing the operation assumes zeroes.
    /// * `input_weight` - A tensor containing the input weights matrix `W` - optional.
    /// * `bias` - A tensor containing the bias `b` - optional.
    /// * `init_state` - The initial internal state of the RNN `h[-1]` - optional.
    /// * `mask` - A tensor containing the mask `m` - optional.
    /// * `descriptor` - A descriptor that defines the parameters for the RNN operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A vector of MPSGraphTensor objects containing gradients for each input tensor, except for `source_gradient` and `mask`.
    /// In case an input is `None`, no gradient will be returned for it.
    /// The order of the gradients will be: for `input`, for `recurrent_weight`, for `input_weight`, for `bias` and finally for `init_state`.
    pub fn single_gate_rnn_gradients(
        &self,
        input: &MPSGraphTensor,
        recurrent_weight: &MPSGraphTensor,
        source_gradient: &MPSGraphTensor,
        z_state: &MPSGraphTensor,
        state_gradient: Option<&MPSGraphTensor>,
        input_weight: Option<&MPSGraphTensor>,
        bias: Option<&MPSGraphTensor>,
        init_state: Option<&MPSGraphTensor>,
        mask: Option<&MPSGraphTensor>,
        descriptor: &MPSGraphSingleGateRNNDescriptor,
        name: Option<&str>,
    ) -> Vec<MPSGraphTensor> {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };
        
        let state_gradient_obj = match state_gradient {
            Some(sg) => sg.0,
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
        
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, singleGateRNNGradientsWithSourceTensor: input.0,
                recurrentWeight: recurrent_weight.0,
                sourceGradient: source_gradient.0,
                zState: z_state.0,
                stateGradient: state_gradient_obj,
                inputWeight: input_weight_obj,
                bias: bias_obj,
                initState: init_state_obj,
                mask: mask_obj,
                descriptor: descriptor.0,
                name: name_obj,
            ];
            
            // Count the number of result tensors (depends on which inputs were provided)
            let count: usize = msg_send![result, count];
            let mut tensors = Vec::with_capacity(count);
            
            // Extract all tensors from the array
            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![result, objectAtIndex: i];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _) as *mut AnyObject;
                tensors.push(MPSGraphTensor(tensor));
            }
            
            objc2::ffi::objc_release(result as *mut _);
            tensors
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
                self.0, LSTMWithSourceTensor: input.0,
                recurrentSourceTensor: initial_hidden_state.0,
                cellSourceTensor: initial_cell_state.0,
                weightsTensor: weights.0,
                recurrentWeightsTensor: recurrent_weights.0,
                biasesTensor: biases_obj,
                descriptor: descriptor.0,
                name: name_obj,
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
                self.0, GRUWithSourceTensor: input.0,
                recurrentSourceTensor: initial_state.0,
                weightsTensor: weights.0,
                recurrentWeightsTensor: recurrent_weights.0,
                biasesTensor: biases_obj,
                descriptor: descriptor.0,
                name: name_obj,
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