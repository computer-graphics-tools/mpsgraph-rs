fn main() {
    // Only compile on Apple platforms
    #[cfg(not(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
        target_os = "visionos"
    )))]
    {
        // Disable doctests when not on Apple platforms (for CI)
        println!("cargo:rustc-cfg=disable_doctests");
        panic!("This crate only supports Apple platforms (macOS, iOS, tvOS, watchOS, visionOS)");
    }

    // Print cargo metadata
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    println!("cargo:rerun-if-changed=build.rs");
}
