use crate::{Graph, Tensor};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Copies the input tensor values into the output, behaving as an identity operation.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object which is a copy of the input.
    pub fn identity(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                identityWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the natural exponent to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn exponent(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                exponentWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies an exponent with base 2 to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn exponent_base2(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                exponentBase2WithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies an exponent with base 10 to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn exponent_base10(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                exponentBase10WithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Computes the natural logarithm to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn logarithm(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                logarithmWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Computes the logarithm with base 2 to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn logarithm_base2(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                logarithmBase2WithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Computes the logarithm with base 10 to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn logarithm_base10(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                logarithmBase10WithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the square operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn square(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                squareWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the square root operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn square_root(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                squareRootWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the reciprocal square root operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn reciprocal_square_root(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                reciprocalSquareRootWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the reciprocal operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn reciprocal(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                reciprocalWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the absolute values of the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn absolute(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                absoluteWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the absolute square of the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn absolute_square(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                absoluteSquareWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies negative to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn negative(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                negativeWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the sign of the input tensor elements.
    ///
    /// This operation returns 1.0 if the correspnding input element is greater than 0,
    /// -1.0 if it is lesser than 0, -0.0 if it is equal to -0.0, and
    /// +0.0 if it is equal to +0.0.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn sign(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                signWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the sign bit of the input tensor elements.
    ///
    /// This operation returns `true` if the sign bit is set for the correspnding floating-point input element,
    /// otherwise it returns `false`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn signbit(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                signbitWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the ceiling operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn ceil(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                ceilWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the floor operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn floor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                floorWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Rounds the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn round(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                roundWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Rounds the input tensor elements by rounding to nearest even.
    ///
    /// - Parameters:
    ///   - tensor: The input tensor.
    ///   - name: An optional string which serves as an identifier for the operation.
    /// - Returns: A valid `Tensor` object containing the elementwise result of the applied operation.
    pub fn rint(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                rintWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the sine operation to the input tensor elements.
    ///
    /// - Parameters:
    ///   - tensor: The input tensor.
    ///   - name: An optional string which serves as an identifier for the operation.
    /// - Returns: A valid `Tensor` object containing the elementwise result of the applied operation.
    pub fn sin(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                sinWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the cosine operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn cos(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                cosWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the tangent operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn tan(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                tanWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the hyperbolic sine operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn sinh(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                sinhWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the hyperbolic cosine operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn cosh(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                coshWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the hyperbolic tangent operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn tanh(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                tanhWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the inverse sine operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn asin(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                asinWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the inverse cosine operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn acos(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                acosWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the inverse tangent operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn atan(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                atanWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the inverse hyperbolic sine operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn asinh(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                asinhWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the inverse hyperbolic cosine operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn acosh(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                acoshWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the inverse hyperbolic tangent operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn atanh(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                atanhWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the logical NOT operation to the input tensor elements.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn not(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                notWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Checks if the input tensor elements are infinite or not.
    ///
    /// If the input tensor element is infinite, the operation returns `true`, else it returns `false`.
    ///
    /// # Arguments
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn is_infinite(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                isInfiniteWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Checks if the input tensor elements are finite or not.
    ///
    /// If the input tensor element is finite, the operation returns `true`, else it returns `false`.
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn is_finite(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                isFiniteWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Checks if the input tensor elements are `NaN` or not.
    ///
    /// If the input tensor element is `NaN`, the operation returns `true`, else it returns `false`.
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn is_nan(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                isNaNWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the error function to the input tensor elements.
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn erf(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                erfWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the truncate operation to the input tensor elements.
    ///
    /// This operation applies the floor operation to positive inputs and ceiling operation to negative inputs.
    ///
    /// - Parameters:
    ///   - tensor: The input tensor.
    ///   - name: An optional string which serves as an identifier for the operation.
    /// - Returns: A valid `Tensor` object containing the elementwise result of the applied operation.
    pub fn truncate(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                truncateWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Applies the bitwise NOT operation to the input tensor element.
    ///
    ///  This operation only accepts integer tensors.
    ///
    /// * `tensor` – The input tensor, which must be of integer type.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn bitwise_not(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                bitwiseNOTWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the population count of the input tensor elements.
    ///
    ///  This operation only accepts integer tensors, and returns the number of bits set in the input element.
    ///
    /// * `tensor` – The input tensor, which must be of integer type.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn bitwise_population_count(
        &self,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                bitwisePopulationCountWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the complex conjugate of the input tensor elements.
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn conjugate(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                conjugateWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    // endregion: UnaryArithmeticOps

    // region: BinaryArithmeticOps

    /// Adds two input tensors.
    ///
    /// This operation creates an add operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = primaryTensor + secondaryTensor
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn addition(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                additionWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Subtracts the second input tensor from the first.
    ///
    /// This operation creates a subtract operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = primaryTensor - secondaryTensor
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn subtraction(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                subtractionWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Multiplies two input tensors.
    ///
    /// This operation creates a multiply operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = primaryTensor * secondaryTensor
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn multiplication(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                multiplicationWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Divides the first input tensor by the second.
    ///
    /// This operation creates a divide operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = primaryTensor / secondaryTensor
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn division(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                divisionWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the remainder obtained by dividing the first input tensor by the second.
    ///
    /// This operation creates a modulo operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = primaryTensor % secondaryTensor
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn modulo(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                moduloWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise result of raising the first tensor to the power of the second tensor.
    ///
    /// This operation creates a power operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = pow(primaryTensor, secondaryTensor)
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn power(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                powerWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise minimum of the input tensors.
    ///
    /// This operation creates a minimum operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = min(primaryTensor, secondaryTensor)
    /// ```
    ///
    /// - Parameters:
    ///   - primary: The LHS tensor of the binary Op.
    ///   - secondary: The RHS tensor of the binary Op.
    ///   - name: An optional string which serves as an identifier for the operation.
    /// - Returns: A valid `Tensor` object containing the elementwise result of the applied operation.
    pub fn minimum(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                minimumWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise maximum of the input tensors.
    ///
    /// This operation creates a maximum operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = max(primaryTensor, secondaryTensor)
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn maximum(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                maximumWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise minimum of the input tensors, while propagating `NaN` values.
    ///
    /// This operation creates a minimum with `NaN` propagation operation and returns the result tensor.
    /// If any of the elementwise operands is `NaN`, the result is `NaN`. It supports broadcasting as well.
    /// ```md
    /// resultTensor = isNaN(primaryTensor) || isNan(secondaryTensor) ? NaN : min(primaryTensor, secondaryTensor)
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn minimum_with_nan_propagation(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                minimumWithNaNPropagationWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise maximum of the input tensors, while propagating `NaN` values.
    ///
    /// This operation creates a maximum with `NaN` propagation operation and returns the result tensor.
    /// If any of the elementwise operands is `NaN`, the result is `NaN`. It supports broadcasting as well.
    /// ```md
    /// resultTensor = isNaN(primaryTensor) || isNan(secondaryTensor) ? NaN : max(primaryTensor, secondaryTensor)
    /// ```
    ///
    /// - Parameters:
    ///   - primary: The LHS tensor of the binary Op.
    ///   - secondary: The RHS tensor of the binary Op.
    ///   - name: An optional string which serves as an identifier for the operation.
    /// - Returns: A valid `Tensor` object containing the elementwise result of the applied operation.
    pub fn maximum_with_nan_propagation(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                maximumWithNaNPropagationWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise equality check of the input tensors.
    ///
    /// This operation creates a equal operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = primaryTensor == secondaryTensor
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn equal(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                equalWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise inequality check of the input tensors.
    ///
    /// This operation creates a not equal operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = primaryTensor != secondaryTensor
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn not_equal(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                notEqualWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Checks in an elementwise manner if the first input tensor is less than the second.
    ///
    /// This operation creates a `lessThan` operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = primaryTensor < secondaryTensor
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn less_than(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                lessThanWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Checks in an elementwise manner if the first input tensor is less than or equal to the second.
    ///
    /// This operation creates a `lessThanOrEqualTo` operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = primaryTensor <= secondaryTensor
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn less_than_or_equal_to(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                lessThanOrEqualToWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Checks in an elementwise manner if the first input tensor is greater than the second.
    ///
    /// This operation creates a `greaterThan` operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = primaryTensor > secondaryTensor
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn greater_than(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                greaterThanWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Checks in an elementwise manner if the first input tensor is greater than or equal to the second.
    ///
    /// This operation creates a `greaterThanOrEqual` operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = primaryTensor < secondaryTensor
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn greater_than_or_equal_to(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                greaterThanOrEqualToWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise logical AND of the input tensors.
    ///
    /// This operation creates a logical AND operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = primaryTensor && secondaryTensor
    /// ```
    ///
    /// - Parameters:
    ///   - primary: The LHS tensor of the binary Op.
    ///   - secondary: The RHS tensor of the binary Op.
    ///   - name: An optional string which serves as an identifier for the operation.
    /// - Returns: A valid `MPSGraphTensor` object containing the elementwise result of the applied operation.
    pub fn logical_and(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                logicalANDWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise logical OR of the input tensors.
    ///
    /// This operation creates a logical OR operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = primaryTensor || secondaryTensor
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn logical_or(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                logicalORWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise logical NAND of the input tensors.
    ///
    /// This operation creates a logical NAND operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = !(primaryTensor && secondaryTensor)
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn logical_nand(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                logicalNANDWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise logical NOR of the input tensors.
    ///
    /// This operation creates a logical NOR operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = !(primaryTensor || secondaryTensor)
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn logical_nor(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                logicalNORWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise logical XOR of the input tensors.
    ///
    /// This operation creates a logical XOR operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = XOR(primaryTensor, secondaryTensor)
    /// ```
    ///
    /// - Parameters:
    ///   - primary: The LHS tensor of the binary Op.
    ///   - secondary: The RHS tensor of the binary Op.
    ///   - name: An optional string which serves as an identifier for the operation.
    /// - Returns: A valid `Tensor` object containing the elementwise result of the applied operation.
    pub fn logical_xor(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                logicalXORWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise logical XNOR of the input tensors.
    ///
    /// This operation creates a logical XNOR operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = XNOR(primaryTensor, secondaryTensor)
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn logical_xnor(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                logicalXNORWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise two-argument arctangent of the input tensors.
    ///
    /// This operation creates a `atan2` operation and returns the result tensor. It supports broadcasting as well.
    /// Graph computes arc tangent of primaryTensor over secondaryTensor.
    /// ```md
    /// resultTensor = atan2(primaryTensor, secondaryTensor)
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn atan2(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                atan2WithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise bitwise AND of binary representations of two integer tensors.
    ///
    /// * `primary` – The primary input tensor, must be of integer type.
    /// * `secondary` – The secondary input tensor, must be of integer type.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn bitwise_and(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                bitwiseANDWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise bitwise OR of binary representations of two integer tensors.
    ///
    /// * `primary` – The primary input tensor, must be of integer type.
    /// * `secondary` – The secondary input tensor, must be of integer type.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn bitwise_or(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                bitwiseORWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise bitwise XOR of binary representations of two integer tensors.
    ///
    /// * `primary` – The primary input tensor, must be of integer type.
    /// * `secondary` – The secondary input tensor, must be of integer type.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn bitwise_xor(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                bitwiseXORWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise left-shifted binary representations of the primary integer by the secondary tensor amount.
    ///
    /// * `primary` – The primary input tensor, must be of integer type.
    /// * `secondary` – The secondary input tensor, must be of integer type.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn bitwise_left_shift(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                bitwiseLeftShiftWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the elementwise right-shifted binary representations of the primary integer by the secondary tensor amount.
    ///
    /// * `primary` – The primary input tensor, must be of integer type.
    /// * `secondary` – The secondary input tensor, must be of integer type.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn bitwise_right_shift(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                bitwiseRightShiftWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Selects values from either the true or false predicate tensor, depending on the values in the first input.
    ///
    /// This operation creates a select operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = select(predicateTensor, truePredicateTensor, falseSelectTensor)
    /// ```
    ///
    /// * `predicate` – The predicate tensor.
    /// * `true_predicate` – The tensor to select values from if predicate is true.
    /// * `false_select` – The tensor to select values from if predicate is false.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn select(
        &self,
        predicate: &Tensor,
        true_tensor: &Tensor,
        false_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                selectWithPredicateTensor: predicate,
                truePredicateTensor: true_tensor,
                falsePredicateTensor: false_tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Clamps the values in the first tensor between the corresponding values in the minimum and maximum value tensor.
    ///
    /// This operation creates a clamp operation and returns the result tensor. It supports broadcasting as well.
    /// ```md
    /// resultTensor = clamp(tensor, minValueTensor, maxValueTensor)
    /// ```
    ///
    /// * `tensor` – The tensor to be clamped.
    /// * `min_tensor` – The tensor with min values to clamp to.
    /// * `max_tensor` – The tensor with max values to clamp to.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn clamp(
        &self,
        tensor: &Tensor,
        min_tensor: &Tensor,
        max_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                clampWithTensor: tensor,
                minValueTensor: min_tensor,
                maxValueTensor: max_tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Divides the first input tensor by the second, with the result being 0 if the denominator is 0.
    ///
    /// ```md
    /// resultTensor = select(secondaryTensor, primaryTensor / secondaryTensor, 0)
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn division_no_nan(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                divisionNoNaNWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the remainder of floor divison between the primary and secondary tensor.
    ///
    /// Creates a floorModulo operation and returns the result tensor, it supports broadcasting as well, returns 0 if divisor is 0.
    /// ```md
    /// resultTensor = primaryTensor - (floor(primaryTensor / secondaryTensor) * secondaryTensor)
    /// ```
    ///
    /// * `primary` – The LHS tensor of the binary Op.
    /// * `secondary` – The RHS tensor of the binary Op.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn floor_modulo(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                floorModuloWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns the real part of a tensor.
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn real_part_of_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                realPartOfTensor: tensor,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Returns the imaginary part of a tensor.
    ///
    /// * `tensor` – The input tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn imaginary_part_of_tensor(
        &self,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                imaginaryPartOfTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Returns a complex tensor from the two input tensors.
    ///
    /// * `real` – The real part of the complex tensor.
    /// * `imaginary` – The imaginary part of the complex tensor.
    /// * `name` – Optional debug label.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object containing the elementwise result of the applied operation.
    pub fn complex_tensor(
        &self,
        real: &Tensor,
        imaginary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                complexTensorWithRealTensor: real,
                imaginaryTensor: imaginary,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }
}
