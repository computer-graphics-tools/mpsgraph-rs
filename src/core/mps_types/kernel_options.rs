use objc2::Encoding;
use objc2::{Encode, RefEncode};
use objc2_foundation::NSUInteger;

/// [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpskerneloptions?language=objc)
// NS_OPTIONS
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelOptions(pub NSUInteger);
bitflags::bitflags! {
    impl KernelOptions: NSUInteger {
        /// Use default options
        const None = 0;
        /// Most MPS functions will sanity check their arguments. This has a small but
        /// non-zero CPU cost. Setting the MPSKernelOptionsSkipAPIValidation will skip these checks.
        /// MPSKernelOptionsSkipAPIValidation does not skip checks for memory allocation failure.
        /// Caution:  turning on MPSKernelOptionsSkipAPIValidation can result in undefined behavior
        /// if the requested operation can not be completed for some reason. Most error states
        /// will be passed through to Metal which may do nothing or abort the program if Metal
        /// API validation is turned on.
        const SkipAPIValidation = 1<<0;
        /// When possible, MPSKernels use a higher precision data representation internally than
        /// the destination storage format to avoid excessive accumulation of computational
        /// rounding error in the result. MPSKernelOptionsAllowReducedPrecision advises the
        /// MPSKernel that the destination storage format already has too much precision for
        /// what is ultimately required downstream, and the MPSKernel may use reduced precision
        /// internally when it feels that a less precise result would yield better performance.
        /// The expected performance win is often small, perhaps 0-20%. When enabled, the
        /// precision of the result may vary by hardware and operating system.
        const AllowReducedPrecision = 1<<1;
        /// Some MPSKernels may automatically split up the work internally into multiple tiles.
        /// This improves performance on larger textures and reduces the amount of memory needed by
        /// MPS for temporary storage. However, if you are using your own tiling scheme to achieve
        /// similar results, your tile sizes and MPS's choice of tile sizes may interfere with
        /// one another causing MPS to subdivide your tiles for its own use inefficiently. Pass
        /// MPSKernelOptionsDisableInternalTiling to force MPS to process your data tile as a
        /// single chunk.
        const DisableInternalTiling = 1<<2;
        /// Enabling this bit will cause various -encode... methods to call MTLCommandEncoder
        /// push/popDebugGroup.  The debug string will be drawn from MPSKernel.label, if any
        /// or the name of the class otherwise.
        const InsertDebugGroups = 1<<3;
        /// Some parts of MPS can provide debug commentary and tuning advice when run.
        /// Setting this bit to 1 will cause the commentary to be emitted to stderr. Otherwise,
        /// the code is silent.  This is especially useful for debugging MPSNNGraph. This option
        /// is on by default when the MPS_LOG_INFO environment variable is defined.  For
        /// even more detailed output on a MPS object, you can use the po command in llvm
        /// with MPS objects:
        ///
        /// ```text
        ///     llvm>  po  <MPS object pointer>
        /// ```
        const Verbose = 1<<4;
    }
}

unsafe impl Encode for KernelOptions {
    const ENCODING: Encoding = NSUInteger::ENCODING;
}

unsafe impl RefEncode for KernelOptions {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
