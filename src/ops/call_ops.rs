use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSString};

use crate::{Graph, ShapedType, Tensor};

impl Graph {
    /// Invokes a compiled graph callable identified by `symbol_name`.
    ///
    /// # Arguments
    ///
    /// * `symbol_name` – Identifier used to look up the executable inside the
    ///   compilation descriptor’s `callables` map.
    /// * `input_tensors` – Inputs passed to the callee.
    /// * `output_types` – Expected result tensor types.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// Boxed slice of output [`Tensor`] objects returned by the callee.
    pub fn call_symbol_name(
        &self,
        symbol_name: &str,
        input_tensors: &[&Tensor],
        output_types: &[&ShapedType],
        name: Option<&str>,
    ) -> Box<[Retained<Tensor>]> {
        unsafe {
            let symbol_name_ns = NSString::from_str(symbol_name);
            let input_tensors_array = NSArray::from_slice(input_tensors);
            let output_types_array = NSArray::from_slice(output_types);

            let result_nsarray: Retained<NSArray<Tensor>> = msg_send![
                self,
                callSymbolName: &*symbol_name_ns,
                inputTensors: &*input_tensors_array,
                outputTypes: &*output_types_array,
                name: name.map(NSString::from_str).as_deref(),
            ];

            result_nsarray.to_vec().into_boxed_slice()
        }
    }
}
