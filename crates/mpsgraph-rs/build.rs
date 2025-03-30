fn main() {
    println!("cargo:rustc-link-search=framework=/System/Library/Frameworks");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");

    // Link with system library for Block_copy support
    println!("cargo:rustc-link-lib=System");
    
    // Disable doctests to avoid issues with Metal device requirements
    println!("cargo:rustc-cfg=mpsgraph_skip_doctests");
}
