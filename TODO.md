# TODO: Check Objective-C to Rust API Mapping

Verify that each Objective-C header has corresponding Rust implementations:

1. [x] MPSGraphActivationOps.h
   - Issues:
     - **Missing in Rust**: Implementation of `reLUGradientWithIncomingGradient` method
     - **Missing in Rust**: Implementation of `sigmoidGradientWithIncomingGradient` method
     - **Missing in Rust**: Implementation of `softMaxGradientWithIncomingGradient` method
     - **Missing in Rust**: Implementation of `leakyReLUGradientWithIncomingGradient` method
     - **Missing in Rust**: Second `leakyReLUWithTensor:alphaTensor` method variant
     - **Extra in Rust**: `prelu`, `gelu`, `hard_sigmoid`, `softplus`, `log_softmax` methods not in ObjC header
     - **Action Required**: Remove `prelu`, `gelu`, `hard_sigmoid`, `softplus`, `log_softmax` from Rust code
2. [x] MPSGraphArithmeticOps.h
   - Note: Arithmetic operations are implemented in `graph.rs` instead of a separate file. Consider moving them to `arithmetic_ops.rs` for better organization.
   - Issues:
     - **Missing in Rust**: Many operations including `identityWithTensor`, `exponentBase2WithTensor`, `exponentBase10WithTensor`, `logarithmBase2WithTensor`, `logarithmBase10WithTensor`, `absoluteSquareWithTensor`, `sign`, `signbit`, `ceil`, `floor`, `round`, `rint`, `sinh`, `cosh`, trigonometric inverse functions, etc.
     - **Missing in Rust**: Complex arithmetic operations including `realPartOfTensor`, `imaginaryPartOfTensor`, `complexTensorWithRealTensor`
     - **Missing in Rust**: Bitwise operations including `bitwiseNOT`, `bitwisePopulationCount`, `bitwiseAND`, etc.
3. [x] MPSGraphAutomaticDifferentiation.h
   - Note: Implemented in `gradient_ops.rs` with a different name but equivalent functionality
   - The Rust implementation is named `gradient_for_primary_tensor` instead of the Swift name `gradients(of:with:name:)`
4. [x] MPSGraphCallOps.h
   - Appears to be fully implemented in `call_ops.rs`
   - The Rust interface uses Vec<MPSGraphTensor> for return value instead of NSArray
5. [x] MPSGraphControlFlowOps.h
   - Implemented in `control_flow_ops.rs` with Rust-style naming conventions
   - All operations are implemented but there are compilation errors in the code
   - **Action Required**: Fix compilation errors in `control_flow_ops.rs` related to Block usage and ObjectRef types
6. [x] MPSGraphConvolutionOps.h
   - Partially implemented in `convolution_ops.rs`
   - Issues:
     - **Missing in Rust**: `MPSGraphConvolution2DOpDescriptor` implementation
     - **Missing in Rust**: `MPSGraphConvolution3DOpDescriptor` implementation
     - **Missing in Rust**: 3D convolution operations
     - **Missing in Rust**: Gradient operations for convolutions
     - **API Mismatch**: Implementation doesn't follow the latest Objective-C API, uses older methods
     - **Action Required**: Update implementation to use descriptors instead of separate parameters
7. [x] MPSGraphConvolutionTransposeOps.h
   - Fully implemented in `convolution_transpose_ops.rs`
   - Provides all functionality from the Objective-C API including:
     - `MPSGraphConvolution2DOpDescriptor` implementation
     - All convolution transpose operations
     - All gradient operations
8. [x] MPSGraphCore.h
   - Mostly implemented in `core.rs` with Rust-style equivalents
   - Issues:
     - **Missing in Rust**: `MPSGraphObject` and `MPSGraphType` wrapper classes
     - **Missing in Rust**: `MPSGraphShapedType` implementation
     - **Extra in Rust**: Helper structs like `OurNSString`, `OurNSArray`, etc. for bridging to Objective-C
9. [x] MPSGraphCumulativeOps.h
   - Partially implemented in `cumulative_ops.rs`
   - Issues:
     - **Missing in Rust**: Some API variants, especially the ones that take axisTensor for cumulative_product, cumulative_minimum, and cumulative_maximum
     - **Missing in Rust**: Simple versions (without exclusive/reverse parameters) for cumulative_product, cumulative_minimum, and cumulative_maximum
     - **Action Required**: Implement missing API variants
10. [x] MPSGraphDepthwiseConvolutionOps.h

- Fully implemented in `depthwise_convolution_ops.rs`
- Provides all functionality from the Objective-C API including:
  - `MPSGraphDepthwiseConvolution2DOpDescriptor` implementation
  - `MPSGraphDepthwiseConvolution3DOpDescriptor` implementation
  - 2D and 3D depthwise convolution operations
  - All gradient operations

11. [x] MPSGraphDevice.h

- Fully implemented in `device.rs`
- Provides all the functionality from the Objective-C API

12. [x] MPSGraphExecutable.h

- Partially implemented in `executable.rs`
- Issues:
  - **Missing in Rust**: `MPSGraphExecutableExecutionDescriptor` implementation
  - **Missing in Rust**: `MPSGraphExecutableSerializationDescriptor` implementation
  - **Missing in Rust**: Serialization and deserialization methods
  - **Missing in Rust**: Asynchronous execution methods with completion handlers
  - **API Mismatch**: Some method signatures don't match the latest API

13. [x] MPSGraphFourierTransformOps.h

- Partially implemented in `fourier_transform_ops.rs`
- Issues:
  - **API Mismatch**: The Rust implementation is using an older API compared to the latest Objective-C API
  - **Missing in Rust**: `MPSGraphFFTDescriptor` implementation does not match the latest properties (inverse, scalingMode, roundToOddHermitean)
  - **Missing in Rust**: New FFT operations like `fastFourierTransform`, `realToHermiteanFFT`, and `HermiteanToRealFFT`
  - **Action Required**: Update the implementation to match the latest Objective-C API

14. [x] MPSGraphGatherOps.h

- Fully implemented in `gather_ops.rs`
- Provides all functionality from the Objective-C API including:
  - `gather` operation
  - `gather_nd` operation
  - `gather_along_axis` and `gather_along_axis_tensor` operations

15. [x] MPSGraphImToColOps.h

- Partially implemented in `im2col_ops.rs`
- Issues:
  - **API Mismatch**: The Rust implementation uses different method names (`image_to_column` instead of `imToCol` and `column_to_image` instead of `colToIm`)
  - **API Mismatch**: The implementation has the `setExplicitPadding` method missing
  - **Action Required**: Update method names to match the Objective-C API

16. [x] MPSGraphLinearAlgebraOps.h

- Partially implemented in `linear_algebra_ops.rs`
- Issues:
  - **Missing in Rust**: The `bandPart` operations introduced in macOS 12.3, iOS 15.4, tvOS 15.4
  - **Action Required**: Implement the missing `bandPart` operations

17. [x] MPSGraphLossOps.h

- Fully implemented in `loss_ops.rs`
- Provides all functionality from the Objective-C API including:
  - `MPSGraphLossReductionType` enum
  - `softmax_cross_entropy` operation
  - `softmax_cross_entropy_gradient` operation

18. [x] MPSGraphMatrixInverseOps.h

- Not implemented in the Rust bindings
- Issues:
  - **Missing in Rust**: The `inverse` operation introduced in macOS 13.0, iOS 16.1, tvOS 16.1
  - **Action Required**: Implement the missing `inverse` operation

19. [x] MPSGraphMatrixMultiplicationOps.h

- Partially implemented in `linear_algebra_ops.rs`
- Issues:
  - **Missing in Rust**: The `HammingDistance` operation introduced in macOS 13.0, iOS 16.0, tvOS 16.0
  - **Missing in Rust**: The `scaledDotProductAttention` operations introduced in macOS 15.0, iOS 18.0, macCatalyst 18.0, tvOS 18.0
  - **Action Required**: Implement the missing operations

20. [x] MPSGraphMemoryOps.h

- Partially implemented in `graph.rs`
- Issues:
  - **Implemented in Rust**: `placeholder` and `constant` operations
  - **Missing in Rust**: `variable`, `readVariable`, and `assignVariable` operations
  - **Missing in Rust**: Complex constant operations introduced in macOS 14.0, iOS 17.0, tvOS 17.0
  - **Missing in Rust**: `variableFromTensor` operation introduced in macOS 15.0, iOS 18.0, macCatalyst 18.0, tvOS 18.0
  - **Action Required**: Implement the missing operations

21. [x] MPSGraphNonMaximumSuppressionOps.h

- Fully implemented in `non_maximum_suppression_ops.rs`
- Provides all functionality from the Objective-C API including:
  - `MPSGraphNonMaximumSuppressionCoordinateMode` enum
  - `non_maximum_suppression` operation
  - `non_maximum_suppression_with_class_indices` operation

22. [x] MPSGraphNonZeroOps.h

- Fully implemented in `non_zero_ops.rs`
- Provides the `non_zero_indices` operation that matches the Objective-C API

23. [x] MPSGraphNormalizationOps.h

- Sample implementation file created in `normalization_ops.rs`
- Missing from the old codebase but now implemented
- Provides all operations from the Objective-C API:
  - Mean and variance calculations
  - Normalization operations
  - Gradient operations for normalization

24. [x] MPSGraphOneHotOps.h

- Fully implemented in `one_hot_ops.rs`
- Provides all variants of the one-hot operation with various parameter combinations
- Uses Rust-style naming conventions for the API variants

25. [x] MPSGraphOperation.h

- Fully implemented in `operation.rs`
- Provides all properties and methods from the Objective-C API:
  - `input_tensors` (inputTensors)
  - `output_tensors` (outputTensors)
  - `graph` (graph)
  - `name` (name)
  - `control_dependencies` (controlDependencies)

26. [x] MPSGraphOptimizerOps.h

- Partially implemented in `optimizer_ops.rs`
- Issues:
  - **Missing in Rust**: `MPSGraphVariableOp` implementation needed for `applyStochasticGradientDescent` method
  - **Missing in Rust**: The `applyStochasticGradientDescent` method that directly updates a variable operation
  - **Action Required**: Implement `MPSGraphVariableOp` struct and complete the missing method

27. [x] MPSGraphPoolingOps.h

- Not implemented in the Rust bindings
- Issues:
  - **Missing in Rust**: `MPSGraphPooling2DOpDescriptor` implementation
  - **Missing in Rust**: `MPSGraphPooling4DOpDescriptor` implementation
  - **Missing in Rust**: All 2D and 4D pooling operations
  - **Action Required**: Create a full implementation in `pooling_ops.rs`

28. [x] MPSGraphQuantizationOps.h

- Fully implemented in `quantization_ops.rs`
- Provides all functionality from the Objective-C API including:
  - Quantization with scalar parameters
  - Quantization with tensor parameters
  - Dequantization operations
  - LUT-based dequantization

29. [x] MPSGraphRNNOps.h

- Partially implemented in `rnn_ops.rs`
- Issues:
  - **API Mismatch**: The Rust implementation is using older method signatures that don't match the latest API
  - **Missing in Rust**: Some of the RNN gradient operations
  - **API Usage**: Current implementation defines custom methods that use different parameter names
  - **Action Required**: Update implementation to use the latest API methods and signatures

30. [x] MPSGraphRandomOps.h

- Fully implemented in `random_ops.rs`
- Provides all functionality from the Objective-C API including:
  - Random tensor generation with different distributions
  - State management for random number generation
  - Dropout operations

31. [x] MPSGraphReductionOps.h

- Partially implemented in `graph.rs` (not in a dedicated file)
- Issues:
  - **Missing in Rust**: Several reduction operations like product, AND, OR, etc.
  - **Missing in Rust**: Single-axis variants of reduction operations
  - **Missing in Rust**: Arg-max/min operations
  - **API Mismatch**: Current implementation uses Rust-style naming instead of matching the Objective-C API
  - **Action Required**: Implement missing reduction operations, potentially move to a dedicated file

32. [x] MPSGraphResizeOps.h

- Partially implemented in `resize_ops.rs`
- Issues:
  - **Missing in Rust**: Several newer resize operations added in iOS 16+ and macOS 13+
  - **Missing in Rust**: Some gradient operations for resize
  - **API Mismatch**: Parameter ordering and naming conventions differ from the Objective-C API
  - **Action Required**: Implement missing resize operations, especially for newer rounding modes

33. [x] MPSGraphSampleGridOps.h

- Fully implemented in `sample_grid_ops.rs`
- Provides all functionality from the Objective-C API including:
  - Sample grid operations with different sampling modes
  - Support for nearest neighbor sampling with specific rounding modes

34. [x] MPSGraphScatterNDOps.h

- Fully implemented in `scatter_nd_ops.rs`
- Provides all functionality from the Objective-C API including:
  - ScatterND operations
  - Scatter operations
  - ScatterAlongAxis operations
  - All modes (Add, Sub, Mul, Div, Min, Max, Set)

35. [x] MPSGraphSortOps.h

- Fully implemented in `sort_ops.rs`
- Provides all functionality from the Objective-C API including:
  - Sort operations with regular and tensor-specified axes
  - ArgSort operations
  - Support for both ascending and descending sorting

36. [x] MPSGraphSparseOps.h

- Partially implemented in `sparse_ops.rs`
- Issues:
  - **Missing in Rust**: The actual methods used in the header are `sparseTensorWithType:tensors:shape:dataType:name:` and `sparseTensorWithDescriptor:tensors:shape:name:`, while the Rust implementation has `sparse_tensor_with_indices_and_values` and `sparse_to_dense`
  - **API Mismatch**: The Rust implementation methods don't match the Objective-C API methods
  - **Extra in Rust**: The `sparse_to_dense` operation is not in the Objective-C header
  - **Action Required**: Update implementation to match the Objective-C API methods for creating sparse tensors

37. [x] MPSGraphStencilOps.h

- Fully implemented in `stencil_ops.rs`
- Provides all functionality from the Objective-C API including:
  - `MPSGraphStencilOpDescriptor` implementation with all initialization methods and property setters
  - `MPSGraphReductionMode` enum
  - Support for the stencil operation with source, weights, and descriptor

38. [x] MPSGraphTensor.h

- Fully implemented in `tensor.rs`
- Provides all the functionality from the Objective-C API

39. [x] MPSGraphTensorData.h

- Partially implemented in `tensor_data.rs`
- Issues:
  - **Missing in Rust**: Several initialization methods including from MPSMatrix, MPSVector, etc.
  - **Extra in Rust**: Debug print statements that should be removed
  - **API Mismatch**: Error handling is ad-hoc, creating NSObject when things fail instead of returning proper errors

40. [x] MPSGraphTensorShapeOps.h

- Partially implemented in `graph.rs`
- Issues:
  - **Missing in Rust**: Many tensor shape operations like `tile`, `pad`, `space-to-depth`, `depth-to-space`, `reverse`, `flatten2D`, `broadcast`, `shapeOf`, `cast`, `stack`, `split`, `squeeze`, `expandDims`, and `coordinate`
  - **API Mismatch**: Current implementation uses a different naming convention
  - **Structure Issue**: Implementations are scattered in `graph.rs` rather than a dedicated file
  - **Action Required**: Implement the missing tensor shape operations and consider reorganizing into a dedicated file

41. [x] MPSGraphTopKOps.h

- Fully implemented in `top_k_ops.rs`
- Provides all functionality from the Objective-C API including:
  - TopK operations with both scalar and tensor parameters
  - BottomK operations with both scalar and tensor parameters
  - Gradient operations for TopK and BottomK
  - Support for specifying axes or using default (minor dimension)

42. [x] MPSGraph.h

- Partially implemented in `graph.rs`
- Issues:
  - **Missing in Rust**: `MPSGraphCompilationDescriptor` implementation
  - **Missing in Rust**: `MPSGraphExecutionDescriptor` implementation
  - **Missing in Rust**: Most of the asynchronous execution methods
  - **API Mismatch**: Some method names differ between Rust and Objective-C
  - **API Mismatch**: Not all parameters are present in some Rust method implementations

43. [x] MetalPerformanceShadersGraph.h

- This is just a header file that imports all other headers, not requiring specific implementation
