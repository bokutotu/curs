extern crate cc;

use cuda_find_path::find_cuda;

fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode").flag("arch=compute_52,code=sm_52") // Generate code for Maxwell (GTX 970, 980, 980 Ti, Titan X).
        .flag("-gencode").flag("arch=compute_53,code=sm_53") // Generate code for Maxwell (Jetson TX1).
        .flag("-gencode").flag("arch=compute_61,code=sm_61") // Generate code for Pascal (GTX 1070, 1080, 1080 Ti, Titan Xp).
        .flag("-gencode").flag("arch=compute_60,code=sm_60") // Generate code for Pascal (Tesla P100).
        .flag("-gencode").flag("arch=compute_62,code=sm_62") // Generate code for Pascal (Jetson TX2).
        .flag("-gencode").flag("arch=compute_75,code=sm_75") // Generate code for Turing (RTX 2080Ti).
        .file("kernel/compare.cu")
        .compile("libcompare.a");

    for path in find_cuda() {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rerun-if-changed=build.rs");
}
