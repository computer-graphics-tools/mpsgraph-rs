mod control_flow_dependency_block;
mod for_loop_body_block;
mod if_then_else_block;
mod while_after_block;
mod while_before_block;

pub use control_flow_dependency_block::ControlFlowDependencyBlock;
pub use for_loop_body_block::ForLoopBodyBlock;
pub use if_then_else_block::IfThenElseBlock;
pub use while_after_block::WhileAfterBlock;
pub use while_before_block::WhileBeforeBlock;

use crate::{Graph, Operation, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSString};
use std::{ops::Deref, ptr::NonNull};

/// MPSGraphControlFlowOps.
impl Graph {
    /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
    ///
    /// This call blocks until execution has completed.
    ///
    /// # Arguments
    ///
    /// - operations: Operations marked as control dependency for all ops created inside the dependent block
    /// - dependent_block: closure which is provided by caller to create dependent ops
    /// - name: name of scope
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor array with results returned from dependent_block forwarded
    fn control_dependency(
        &self,
        operations: &[&Operation],
        dependent_block: ControlFlowDependencyBlock,
        name: Option<&str>,
    ) -> Box<[Retained<Tensor>]> {
        let operations_array = NSArray::from_slice(operations);
        let result: Retained<NSArray<Tensor>> = unsafe {
            msg_send![
                self,
                controlDependencyWithOperations: &*operations_array,
                dependentBlock: &*dependent_block,
                name: name.map(NSString::from_str).as_deref(),
            ]
        };
        result.to_vec().into_boxed_slice()
    }

    /// Adds an if-then-else operation to the graph.
    ///
    /// # Arguments
    ///
    /// * `predicate_tensor` - [`Tensor`] must have a single scalar value, used to decide between then/else branches
    /// * `then_block` - If predicate is true operations in this block are executed
    /// * `else_block` - If predicate is false operations in this block are executed
    /// * `name` - Name of the operation
    ///
    /// # Returns
    ///
    /// If no error, the tensors returned by the user. If not empty, the user must define both `then_block` and `else_block`;
    /// both should have the same number of arguments and each corresponding argument should have the same element types.
    pub fn if_then_else(
        &self,
        predicate_tensor: &Tensor,
        then_block: IfThenElseBlock,
        else_block: IfThenElseBlock,
        name: Option<&str>,
    ) -> Box<[Retained<Tensor>]> {
        let result: Retained<NSArray<Tensor>> = unsafe {
            msg_send![
                self,
                ifWithPredicateTensor: predicate_tensor,
                thenBlock: &*then_block,
                elseBlock: &*else_block,
                name: name.map(NSString::from_str).as_deref(),
            ]
        };
        result.to_vec().into_boxed_slice()
    }

    /// Adds a while loop operation.
    ///
    /// # Arguments
    ///
    /// * `initial_inputs` - Input tensors to the `before_block`. For the first iteration, these are the same as the `initial_inputs` passed to the while loop.
    /// * `before_block` - This block is run first and then calls the `after_block` with the results, or returns the results from the loop.
    /// * `after_block` - Executed after the condition evaluation.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] slice with results returned from the condition block, depending on the predicate tensor.
    pub fn while_loop(
        &self,
        initial_inputs: &[&Tensor],
        before_block: WhileBeforeBlock,
        after_block: WhileAfterBlock,
        name: Option<&str>,
    ) -> Box<[Retained<Tensor>]> {
        let initial_intputs_array = NSArray::from_slice(initial_inputs);
        let result: Retained<NSArray<Tensor>> = unsafe {
            msg_send![
                self,
                whileWithInitialInputs: &*initial_intputs_array,
                before: &*before_block,
                after: &*after_block,
                name: name.map(NSString::from_str).as_deref(),
            ]
        };
        result.to_vec().into_boxed_slice()
    }

    /// Adds a for loop operation. The lower and upper bounds specify a half-open range: the range includes the lower bound but does not include the upper bound.
    ///
    /// # Arguments
    ///
    /// * `lower_bound` - Lower bound value of the loop. This is a scalar tensor and is the index the loop will start with.
    /// * `upper_bound` - Upper bound value of the loop. This is a scalar tensor.
    /// * `step` - Step value of the loop. This is a scalar tensor and must be positive.
    /// * `initial_body_arguments` - Initial set of iteration arguments passed to the `body` block of the for loop.
    /// * `body` - This block will execute the body of the for loop.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] slice with the same count and corresponding element types as `initial_body_arguments` and the return types of the for loop.
    pub fn for_loop(
        &self,
        lower_bound: &Tensor,
        upper_bound: &Tensor,
        step: &Tensor,
        initial_body_arguments: &[&Tensor],
        body: ForLoopBodyBlock,
        name: Option<&str>,
    ) -> Box<[Retained<Tensor>]> {
        let initial_body_arguments_array = NSArray::from_slice(initial_body_arguments);
        let result: Retained<NSArray<Tensor>> = unsafe {
            msg_send![
                self,
                forLoopWithLowerBound: lower_bound,
                upperBound: upper_bound,
                step: step,
                initialBodyArguments: &*initial_body_arguments_array,
                body: &*body,
                name: name.map(NSString::from_str).as_deref(),
            ]
        };
        result.to_vec().into_boxed_slice()
    }

    /// Adds a for loop operation with a specific number of iterations.
    ///
    /// # Arguments
    ///
    /// * `number_of_iterations` - [`Tensor`] with the number of iterations the loop will execute.
    /// * `initial_body_arguments` - Initial set of iteration arguments passed to the `body` block of the for loop.
    /// * `body` - The `body` block. This executes the body of the for loop; the index will go from 0 to `number_of_iterations` − 1.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] slice with the same count and corresponding element types as `initial_body_arguments` and the return types of the for loop.
    pub fn for_loop_with_number_of_iterations(
        &self,
        number_of_iterations: &Tensor,
        initial_body_arguments: &[&Tensor],
        body: ForLoopBodyBlock,
        name: Option<&str>,
    ) -> Box<[Retained<Tensor>]> {
        let initial_body_arguments_array = NSArray::from_slice(initial_body_arguments);
        let result: Retained<NSArray<Tensor>> = unsafe {
            msg_send![
                self,
                forLoopWithNumberOfIterations: number_of_iterations,
                initialBodyArguments: &*initial_body_arguments_array,
                body: &*body,
                name: name.map(NSString::from_str).as_deref(),
            ]
        };
        result.to_vec().into_boxed_slice()
    }
}
