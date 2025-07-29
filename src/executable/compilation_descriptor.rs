use super::Executable;
use super::Optimization;
use super::OptimizationProfile;
use crate::ComputeDevice;
use objc2::{
    ClassType, extern_class, msg_send,
    rc::{Allocated, Retained},
    runtime::NSObject,
};
use objc2_foundation::{NSDictionary, NSObjectProtocol, NSString};
use std::{collections::HashMap, iter::zip, ptr::null};

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphCompilationDescriptor"]
    pub struct CompilationDescriptor;
);

unsafe impl NSObjectProtocol for CompilationDescriptor {}

impl CompilationDescriptor {
    /// Create a new compilation descriptor
    pub fn new() -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let allocated: Allocated<Self> = msg_send![class, alloc];
            let initialized: Retained<Self> = msg_send![allocated, init];
            initialized
        }
    }

    /// Set the optimization level
    pub fn set_optimization_level(&self, level: Optimization) {
        unsafe {
            let _: () = msg_send![self, setOptimizationLevel: level as u64];
        }
    }

    /// Set the optimization profile
    pub fn set_optimization_profile(&self, profile: OptimizationProfile) {
        unsafe {
            let _: () = msg_send![self, setOptimizationProfile: profile as u64];
        }
    }

    /// Set whether to debug compile
    pub fn set_debug_compile(&self, debug_compile: bool) {
        unsafe {
            let _: () = msg_send![self, setDebugCompile: debug_compile];
        }
    }

    /// Get the callables map as an `Option<HashMap>`.
    pub fn get_callables(&self) -> Option<HashMap<String, Retained<Executable>>> {
        let optional_dict: Option<Retained<NSDictionary<NSString, Executable>>> =
            unsafe { msg_send![self, callables] };
        optional_dict.map(|d| {
            let (keys, objects) = d.to_vecs();
            zip(keys.into_iter().map(|k| k.to_string()), objects.into_iter())
                .collect::<HashMap<_, _>>()
        })
    }

    /// Set the callables map using a Rust HashMap
    pub fn set_callables(&self, callables: &Option<HashMap<String, &Executable>>) {
        match callables {
            None => {
                // Set property to nil
                unsafe {
                    let _: () = msg_send![
                        self,
                        setCallables: null::<NSDictionary<NSString, Executable>>()
                    ];
                }
            }
            Some(map) if map.is_empty() => unsafe {
                let _: () = msg_send![
                    self,
                    setCallables: null::<NSDictionary<NSString, Executable>>()
                ];
            },
            Some(map) => {
                let mut ns_keys: Vec<Retained<NSString>> = Vec::with_capacity(map.len());
                let mut exec_refs: Vec<&Executable> = Vec::with_capacity(map.len());

                for (k, &exec) in map {
                    ns_keys.push(NSString::from_str(k));
                    exec_refs.push(exec);
                }

                let key_refs: Vec<&NSString> = ns_keys.iter().map(|k| &**k).collect();
                let dict: Retained<NSDictionary<NSString, Executable>> =
                    NSDictionary::from_slices(&key_refs, &exec_refs);

                unsafe {
                    let _: () = msg_send![self, setCallables: &*dict];
                }
            }
        }
    }

    pub fn set_compiler_options(&self, options: u64) {
        unsafe {
            let _: () = msg_send![self, setCompilerOptions: options];
        }
    }

    pub fn set_ane_compiler_spatial_splitting(&self, value: u64) {
        unsafe {
            let _: () = msg_send![self, setAneCompilerSpatialSplitting: value];
        }
    }

    pub fn set_enable_ane_fw_to_fw_signal(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableANEFWToFWSignal: enable];
        }
    }

    pub fn set_enable_ane_late_latch(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableANELateLatch: enable];
        }
    }

    pub fn set_print_ane_placement_analysis(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setPrintANEPlacementAnalysis: enable];
        }
    }

    pub fn set_preferred_device(&self, device: ComputeDevice) {
        unsafe {
            let _: () = msg_send![self, setPreferredDevice: device.bits()];
        }
    }

    pub fn set_allowed_compute_devices(&self, devices: ComputeDevice) {
        unsafe {
            let _: () = msg_send![self, setAllowedComputeDevices: devices.bits()];
        }
    }

    pub fn set_enable_parallel_encode(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableParallelEncode: enable];
        }
    }

    pub fn set_maximum_number_of_parallel_encoding_regions(&self, value: u64) {
        unsafe {
            let _: () = msg_send![self, setMaximumNumberOfParallelEncodingRegions: value];
        }
    }

    pub fn set_minimum_number_of_ops_in_parallel_region(&self, value: u64) {
        unsafe {
            let _: () = msg_send![self, setMinimumNumberOfOpsInParallelRegion: value];
        }
    }

    pub fn set_enable_mlir_diagnostics(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableMLIRDiagnostics: enable];
        }
    }

    pub fn set_enable_shape_equivalence(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableShapeEquivalence: enable];
        }
    }
}
