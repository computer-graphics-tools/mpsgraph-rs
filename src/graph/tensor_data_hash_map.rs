use std::collections::HashMap;

use objc2::rc::Retained;
use objc2_foundation::NSDictionary;

use crate::Tensor;
use crate::TensorData;

pub type RetainedTensorDataHashMap = HashMap<Retained<Tensor>, Retained<TensorData>>;
pub type TensorDataHashMap<'a> = HashMap<&'a Tensor, &'a TensorData>;
pub type TensorDataDictionary = NSDictionary<Tensor, TensorData>;
