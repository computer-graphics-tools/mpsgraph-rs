use std::collections::HashMap;

use objc2::{Message, rc::Retained};
use objc2_foundation::{CopyingHelper, NSCopying, NSDictionary, NSObjectProtocol};
use std::hash::Hash;

/// Trait providing a conversion from a `HashMap<&K, &V>` to an Objective-C
/// `NSDictionary<K, V>` wrapped in `Retained`.
///
/// This is implemented generically for any key/value types that satisfy
/// the required Objective-C traits.
pub trait ToNSDictionary<K, V> {
    fn to_dictionary(&self) -> Retained<NSDictionary<K, V>>;
}

impl<'a, K, V> ToNSDictionary<K, V> for HashMap<&'a K, &'a V>
where
    K: NSCopying + NSObjectProtocol + Message + CopyingHelper<Result = K>,
    V: NSObjectProtocol + Message,
{
    fn to_dictionary(&self) -> Retained<NSDictionary<K, V>> {
        let keys: Vec<&K> = self.keys().copied().collect();
        let values: Vec<&V> = self.values().copied().collect();
        NSDictionary::from_slices(&keys, &values)
    }
}

/// Extension trait that converts an `NSDictionary<K, V>` back into an owned
/// `HashMap<Retained<K>, Retained<V>>`.
pub trait NSDictionaryExt<K, V> {
    fn to_hashmap(&self) -> HashMap<Retained<K>, Retained<V>>;
}

impl<K, V> NSDictionaryExt<K, V> for NSDictionary<K, V>
where
    K: NSObjectProtocol + Message,
    V: NSObjectProtocol + Message,
    Retained<K>: Eq + Hash,
{
    fn to_hashmap(&self) -> HashMap<Retained<K>, Retained<V>> {
        let (keys, values) = self.to_vecs();
        HashMap::from_iter(keys.into_iter().zip(values))
    }
}
