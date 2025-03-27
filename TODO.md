# TODO

## Missing Implementations

- Implement MPSGraphAutomaticDifferentiation.h (currently partially implemented in gradient_ops.rs)
- Split matrix_ops.rs into separate files:
  - matrix_inverse_ops.rs for MPSGraphMatrixInverseOps.h
  - matrix_multiplication_ops.rs for MPSGraphMatrixMultiplicationOps.h

## Structure Review

- Review if scatter_nd_ops.rs properly implements MPSGraphScatterNDOps.h
- Evaluate if graph.rs properly covers all MPSGraph.h functionality
- Consider if any utility code in data_types.rs and utils.rs should be reorganized
