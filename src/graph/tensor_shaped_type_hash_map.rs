use std::collections::HashMap;

use objc2_foundation::NSDictionary;

use crate::ShapedType;
use crate::Tensor;

pub type TensorShapedTypeHashMap<'a> = HashMap<&'a Tensor, &'a ShapedType>;
pub type TensorShapedTypeDictionary = NSDictionary<Tensor, ShapedType>;
