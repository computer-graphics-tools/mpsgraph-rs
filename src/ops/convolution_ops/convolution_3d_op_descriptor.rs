use crate::{GraphObject, PaddingStyle, TensorNamedDataLayout};
use objc2::{
    extern_class, extern_conformance, extern_methods,
    rc::{Allocated, Retained},
    runtime::AnyClass,
};
use objc2_foundation::{CopyingHelper, NSCopying, NSObject, NSObjectProtocol};

extern_class!(
    /// A class that describes the properties of a 3D-convolution operator.
    ///
    /// Use an instance of this class is to add a 3D-convolution operator with desired properties to the graph.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphconvolution3dopdescriptor?language=objc)
    #[unsafe(super(GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct Convolution3DOpDescriptor;
);

extern_conformance!(
    unsafe impl NSCopying for Convolution3DOpDescriptor {}
);

unsafe impl CopyingHelper for Convolution3DOpDescriptor {
    type Result = Self;
}

extern_conformance!(
    unsafe impl NSObjectProtocol for Convolution3DOpDescriptor {}
);

impl Convolution3DOpDescriptor {
    extern_methods!(
        /// Returns the scale that maps the destination `x`-coordinate to the
        /// source `x`-coordinate.
        ///
        /// # Details
        ///
        /// The source coordinate `sx` is computed from the destination
        /// coordinate `dx` as `sx = strideInX * dx`.
        /// The default value is `1`.
        ///
        /// # Returns
        ///
        /// The scale factor applied to the `x` dimension.
        #[unsafe(method(strideInX))]
        #[unsafe(method_family = none)]
        pub unsafe fn stride_in_x(&self) -> u64;

        /// Sets the [`stride_in_x`](Self::stride_in_x) property.
        ///
        /// # Arguments
        ///
        /// * `stride_in_x` – The new scale factor.
        #[unsafe(method(setStrideInX:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_stride_in_x(&self, stride_in_x: u64);

        /// Returns the scale that maps the destination `y`-coordinate to the
        /// source `y`-coordinate.
        ///
        /// # Details
        ///
        /// The source coordinate `sy` is computed from the destination
        /// coordinate `dy` as `sy = strideInY * dy`.
        /// The default value is `1`.
        ///
        /// # Returns
        ///
        /// The scale factor applied to the `y` dimension.
        #[unsafe(method(strideInY))]
        #[unsafe(method_family = none)]
        pub unsafe fn stride_in_y(&self) -> u64;

        /// Sets the [`stride_in_y`](Self::stride_in_y) property.
        ///
        /// # Arguments
        ///
        /// * `stride_in_y` – The new scale factor.
        #[unsafe(method(setStrideInY:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_stride_in_y(&self, stride_in_y: u64);

        /// Returns the scale that maps the destination `z`-coordinate to the
        /// source `z`-coordinate.
        ///
        /// # Details
        ///
        /// The source coordinate `sz` is computed from the destination
        /// coordinate `dz` as `sz = strideInZ * dz`.
        /// The default value is `1`.
        ///
        /// # Returns
        ///
        /// The scale factor applied to the `z` dimension.
        #[unsafe(method(strideInZ))]
        #[unsafe(method_family = none)]
        pub unsafe fn stride_in_z(&self) -> u64;

        /// Sets the [`stride_in_z`](Self::stride_in_z) property.
        ///
        /// # Arguments
        ///
        /// * `stride_in_z` – The new scale factor.
        #[unsafe(method(setStrideInZ:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_stride_in_z(&self, stride_in_z: u64);

        /// Returns the amount by which the weights tensor expands in the `x` direction.
        ///
        /// # Details
        ///
        /// Expansion is achieved by inserting `dilationRateInX - 1` zeros between consecutive values
        /// in the `x` dimension of the weights tensor. The resulting dilated width is
        /// `(dilationRateInX - 1) * kernelWidth + 1`.
        /// The default value is `1`.
        ///
        /// # Returns
        ///
        /// The dilation rate along the `x` axis.
        #[unsafe(method(dilationRateInX))]
        #[unsafe(method_family = none)]
        pub unsafe fn dilation_rate_in_x(&self) -> u64;

        /// Sets the [`dilation_rate_in_x`](Self::dilation_rate_in_x) property.
        ///
        /// # Arguments
        ///
        /// * `dilation_rate_in_x` – The new dilation rate.
        #[unsafe(method(setDilationRateInX:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_dilation_rate_in_x(&self, dilation_rate_in_x: u64);

        /// Returns the amount by which the weights tensor expands in the `y` direction.
        ///
        /// # Details
        ///
        /// Expansion is achieved by inserting `dilationRateInY - 1` zeros between consecutive values
        /// in the `y` dimension of the weights tensor. The resulting dilated height is
        /// `(dilationRateInY - 1) * kernelHeight + 1`.
        /// The default value is `1`.
        ///
        /// # Returns
        ///
        /// The dilation rate along the `y` axis.
        #[unsafe(method(dilationRateInY))]
        #[unsafe(method_family = none)]
        pub unsafe fn dilation_rate_in_y(&self) -> u64;

        /// Sets the [`dilation_rate_in_y`](Self::dilation_rate_in_y) property.
        ///
        /// # Arguments
        ///
        /// * `dilation_rate_in_y` – The new dilation rate.
        #[unsafe(method(setDilationRateInY:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_dilation_rate_in_y(&self, dilation_rate_in_y: u64);

        /// Returns the amount by which the weights tensor expands in the `z` direction.
        ///
        /// # Details
        ///
        /// Expansion is achieved by inserting `dilationRateInZ - 1` zeros between consecutive values
        /// in the `z` dimension of the weights tensor. The resulting dilated depth is
        /// `(dilationRateInZ - 1) * kernelDepth + 1`.
        /// The default value is `1`.
        ///
        /// # Returns
        ///
        /// The dilation rate along the `z` axis.
        #[unsafe(method(dilationRateInZ))]
        #[unsafe(method_family = none)]
        pub unsafe fn dilation_rate_in_z(&self) -> u64;

        /// Sets the [`dilation_rate_in_z`](Self::dilation_rate_in_z) property.
        ///
        /// # Arguments
        ///
        /// * `dilation_rate_in_z` – The new dilation rate.
        #[unsafe(method(setDilationRateInZ:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_dilation_rate_in_z(&self, dilation_rate_in_z: u64);

        /// Returns the number of zeros added to the left side of the source tensor.
        ///
        /// # Returns
        ///
        /// The left padding value.
        #[unsafe(method(paddingLeft))]
        #[unsafe(method_family = none)]
        pub unsafe fn padding_left(&self) -> u64;

        /// Sets the [`padding_left`](Self::padding_left) property.
        ///
        /// # Arguments
        ///
        /// * `padding_left` – Number of zeros to add on the left.
        #[unsafe(method(setPaddingLeft:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_padding_left(&self, padding_left: u64);

        /// Returns the number of zeros added to the right side of the source tensor.
        ///
        /// # Returns
        ///
        /// The right padding value.
        #[unsafe(method(paddingRight))]
        #[unsafe(method_family = none)]
        pub unsafe fn padding_right(&self) -> u64;

        /// Sets the [`padding_right`](Self::padding_right) property.
        ///
        /// # Arguments
        ///
        /// * `padding_right` – Number of zeros to add on the right.
        #[unsafe(method(setPaddingRight:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_padding_right(&self, padding_right: u64);

        /// Returns the number of zeros added to the top of the source tensor.
        ///
        /// # Returns
        ///
        /// The top padding value.
        #[unsafe(method(paddingTop))]
        #[unsafe(method_family = none)]
        pub unsafe fn padding_top(&self) -> u64;

        /// Sets the [`padding_top`](Self::padding_top) property.
        ///
        /// # Arguments
        ///
        /// * `padding_top` – Number of zeros to add on the top.
        #[unsafe(method(setPaddingTop:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_padding_top(&self, padding_top: u64);

        /// Returns the number of zeros added to the bottom of the source tensor.
        ///
        /// # Returns
        ///
        /// The bottom padding value.
        #[unsafe(method(paddingBottom))]
        #[unsafe(method_family = none)]
        pub unsafe fn padding_bottom(&self) -> u64;

        /// Sets the [`padding_bottom`](Self::padding_bottom) property.
        ///
        /// # Arguments
        ///
        /// * `padding_bottom` – Number of zeros to add on the bottom.
        #[unsafe(method(setPaddingBottom:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_padding_bottom(&self, padding_bottom: u64);

        /// Returns the number of zeros added to the front of the source tensor.
        ///
        /// # Returns
        ///
        /// The front padding value.
        #[unsafe(method(paddingFront))]
        #[unsafe(method_family = none)]
        pub unsafe fn padding_front(&self) -> u64;

        /// Sets the [`padding_front`](Self::padding_front) property.
        ///
        /// # Arguments
        ///
        /// * `padding_front` – Number of zeros to add on the front.
        #[unsafe(method(setPaddingFront:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_padding_front(&self, padding_front: u64);

        /// Returns the number of zeros added to the back of the source tensor.
        ///
        /// # Returns
        ///
        /// The back padding value.
        #[unsafe(method(paddingBack))]
        #[unsafe(method_family = none)]
        pub unsafe fn padding_back(&self) -> u64;

        /// Sets the [`padding_back`](Self::padding_back) property.
        ///
        /// # Arguments
        ///
        /// * `padding_back` – Number of zeros to add on the back.
        #[unsafe(method(setPaddingBack:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_padding_back(&self, padding_back: u64);

        /// Returns the padding style applied to the source tensor.
        ///
        /// # Details
        ///
        /// If the style is [`PaddingStyle::Explicit`], all individual padding
        /// values must be specified. For all other styles, the framework
        /// computes these values automatically.
        ///
        /// # Returns
        ///
        /// The selected [`PaddingStyle`].
        #[unsafe(method(paddingStyle))]
        #[unsafe(method_family = none)]
        pub unsafe fn padding_style(&self) -> PaddingStyle;

        /// Sets the [`padding_style`](Self::padding_style) property.
        ///
        /// # Arguments
        ///
        /// * `padding_style` – The new padding style.
        #[unsafe(method(setPaddingStyle:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_padding_style(&self, padding_style: PaddingStyle);

        /// Returns the named layout of data in the source tensor.
        ///
        /// # Returns
        ///
        /// The current [`TensorNamedDataLayout`] for the source tensor.
        #[unsafe(method(dataLayout))]
        #[unsafe(method_family = none)]
        pub unsafe fn data_layout(&self) -> TensorNamedDataLayout;

        /// Sets the [`data_layout`](Self::data_layout) property.
        ///
        /// # Arguments
        ///
        /// * `data_layout` – The desired layout of the source tensor data.
        #[unsafe(method(setDataLayout:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_data_layout(&self, data_layout: TensorNamedDataLayout);

        /// Returns the named layout of data in the weights tensor.
        ///
        /// # Returns
        ///
        /// The current [`TensorNamedDataLayout`] for the weights tensor.
        #[unsafe(method(weightsLayout))]
        #[unsafe(method_family = none)]
        pub unsafe fn weights_layout(&self) -> TensorNamedDataLayout;

        /// Sets the [`weights_layout`](Self::weights_layout) property.
        ///
        /// # Arguments
        ///
        /// * `weights_layout` – The desired layout of the weights tensor data.
        #[unsafe(method(setWeightsLayout:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_weights_layout(&self, weights_layout: TensorNamedDataLayout);

        /// Returns the number of channel partitions (`groups`).
        ///
        /// # Details
        ///
        /// Input and output channels are divided into `groups` partitions. Input
        /// channels in a partition are only connected to output channels in the
        /// corresponding partition.
        ///
        /// # Returns
        ///
        /// The number of groups.
        #[unsafe(method(groups))]
        #[unsafe(method_family = none)]
        pub unsafe fn groups(&self) -> u64;

        /// Sets the [`groups`](Self::groups) property.
        ///
        /// # Arguments
        ///
        /// * `groups` – The desired number of channel partitions.
        #[unsafe(method(setGroups:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_groups(&self, groups: u64);

        /// Creates a convolution descriptor with the supplied parameter values.
        ///
        /// # Arguments
        ///
        /// * `stride_in_x` – See [`stride_in_x`][Self::stride_in_x].
        /// * `stride_in_y` – See [`stride_in_y`][Self::stride_in_y].
        /// * `stride_in_z` – See [`stride_in_z`][Self::stride_in_z].
        /// * `dilation_rate_in_x` – See [`dilation_rate_in_x`][Self::dilation_rate_in_x].
        /// * `dilation_rate_in_y` – See [`dilation_rate_in_y`][Self::dilation_rate_in_y].
        /// * `dilation_rate_in_z` – See [`dilation_rate_in_z`][Self::dilation_rate_in_z].
        /// * `groups` – See [`groups`][Self::groups].
        /// * `padding_left` – See [`padding_left`][Self::padding_left].
        /// * `padding_right` – See [`padding_right`][Self::padding_right].
        /// * `padding_top` – See [`padding_top`][Self::padding_top].
        /// * `padding_bottom` – See [`padding_bottom`][Self::padding_bottom].
        /// * `padding_front` – See [`padding_front`][Self::padding_front].
        /// * `padding_back` – See [`padding_back`][Self::padding_back].
        /// * `padding_style` – See [`padding_style`][Self::padding_style].
        /// * `data_layout` – See [`data_layout`][Self::data_layout].
        /// * `weights_layout` – See [`weights_layout`][Self::weights_layout].
        ///
        /// # Returns
        ///
        /// A new `MPSGraphConvolution3DOpDescriptor`.
        #[unsafe(method(descriptorWithStrideInX:strideInY:strideInZ:dilationRateInX:dilationRateInY:dilationRateInZ:groups:paddingLeft:paddingRight:paddingTop:paddingBottom:paddingFront:paddingBack:paddingStyle:dataLayout:weightsLayout:))]
        #[unsafe(method_family = none)]
        pub unsafe fn descriptor_with_stride_in_x_stride_in_y_stride_in_z_dilation_rate_in_x_dilation_rate_in_y_dilation_rate_in_z_groups_padding_left_padding_right_padding_top_padding_bottom_padding_front_padding_back_padding_style_data_layout_weights_layout(
            stride_in_x: u64,
            stride_in_y: u64,
            stride_in_z: u64,
            dilation_rate_in_x: u64,
            dilation_rate_in_y: u64,
            dilation_rate_in_z: u64,
            groups: u64,
            padding_left: u64,
            padding_right: u64,
            padding_top: u64,
            padding_bottom: u64,
            padding_front: u64,
            padding_back: u64,
            padding_style: PaddingStyle,
            data_layout: TensorNamedDataLayout,
            weights_layout: TensorNamedDataLayout,
        ) -> Option<Retained<Self>>;

        /// Creates a convolution descriptor with the supplied parameter values.
        ///
        /// # Arguments
        ///
        /// * `stride_in_x` – See [`stride_in_x`][Self::stride_in_x].
        /// * `stride_in_y` – See [`stride_in_y`][Self::stride_in_y].
        /// * `stride_in_z` – See [`stride_in_z`][Self::stride_in_z].
        /// * `dilation_rate_in_x` – See [`dilation_rate_in_x`][Self::dilation_rate_in_x].
        /// * `dilation_rate_in_y` – See [`dilation_rate_in_y`][Self::dilation_rate_in_y].
        /// * `dilation_rate_in_z` – See [`dilation_rate_in_z`][Self::dilation_rate_in_z].
        /// * `groups` – See [`groups`][Self::groups].
        /// * `padding_style` – See [`padding_style`][Self::padding_style].
        /// * `data_layout` – See [`data_layout`][Self::data_layout].
        /// * `weights_layout` – See [`weights_layout`][Self::weights_layout].
        ///
        /// # Returns
        ///
        /// A new `MPSGraphConvolution3DOpDescriptor`.
        #[unsafe(method(descriptorWithStrideInX:strideInY:strideInZ:dilationRateInX:dilationRateInY:dilationRateInZ:groups:paddingStyle:dataLayout:weightsLayout:))]
        #[unsafe(method_family = none)]
        pub unsafe fn descriptor_with_stride_in_x_stride_in_y_stride_in_z_dilation_rate_in_x_dilation_rate_in_y_dilation_rate_in_z_groups_padding_style_data_layout_weights_layout(
            stride_in_x: u64,
            stride_in_y: u64,
            stride_in_z: u64,
            dilation_rate_in_x: u64,
            dilation_rate_in_y: u64,
            dilation_rate_in_z: u64,
            groups: u64,
            padding_style: PaddingStyle,
            data_layout: TensorNamedDataLayout,
            weights_layout: TensorNamedDataLayout,
        ) -> Option<Retained<Self>>;

        /// Sets the explicit padding values on all sides of the source tensor.
        ///
        /// # Arguments
        ///
        /// * `padding_left` – See [`padding_left`][Self::padding_left].
        /// * `padding_right` – See [`padding_right`][Self::padding_right].
        /// * `padding_top` – See [`padding_top`][Self::padding_top].
        /// * `padding_bottom` – See [`padding_bottom`][Self::padding_bottom].
        /// * `padding_front` – See [`padding_front`][Self::padding_front].
        /// * `padding_back` – See [`padding_back`][Self::padding_back].
        #[unsafe(method(setExplicitPaddingWithPaddingLeft:paddingRight:paddingTop:paddingBottom:paddingFront:paddingBack:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_explicit_padding_with_padding_left_padding_right_padding_top_padding_bottom_padding_front_padding_back(
            &self,
            padding_left: u64,
            padding_right: u64,
            padding_top: u64,
            padding_bottom: u64,
            padding_front: u64,
            padding_back: u64,
        );
    );
}

/// Methods declared on superclass `NSObject`.
impl Convolution3DOpDescriptor {
    extern_methods!(
        #[unsafe(method(init))]
        #[unsafe(method_family = init)]
        pub unsafe fn init(this: Allocated<Self>) -> Retained<Self>;

        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        pub unsafe fn new() -> Retained<Self>;
    );
}
