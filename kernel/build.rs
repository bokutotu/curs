extern crate cc;

use cuda_find_path::find_cuda;

fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode").flag("arch=compute_52,code=sm_52") // for (GTX 970, 980, 980 Ti, Titan X).
        .flag("-gencode").flag("arch=compute_53,code=sm_53") // for (Jetson TX1).
        .flag("-gencode").flag("arch=compute_61,code=sm_61") // for (GTX 1070,1080,1080Ti, Titan Xp).
        .flag("-gencode").flag("arch=compute_60,code=sm_60") // for (Tesla P100).
        .flag("-gencode").flag("arch=compute_62,code=sm_62") // for (Jetson TX2).
        .flag("-gencode").flag("arch=compute_75,code=sm_75") // for (RTX 2080Ti).
        .file("kernel/compare.cu")
        .compile("libcompare.a");

    for path in find_cuda() {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stub");
    println!("cargo:rustc-link-lib=cuda");
}
