fn main() {
    // Run on both macOS and iOS
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    }
}
