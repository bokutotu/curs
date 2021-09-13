extern crate cc;

use cuda_find_path::find_cuda;
use std::{env, path::PathBuf};

fn main() {
    let cuda_files = vec![
        "kernel/compare.cu",
        "kernel/element_wise_operator.cu",
        "kernel/transpose.cu",
        "kernel/array_scalar_add.cu",
    ];

    for cuda_file in cuda_files.iter() {
        println!("{}", format!("cargo:rerun-if-changed={}", cuda_file));
    }
    println!("cargo:rerun-if-changed=kernel/kernel.h");
    println!("cargo:rerun-if-changed=build.rs");

    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        // .flag("-gencode")
        // .flag("arch=compute_52,code=sm_52") // for (GTX 970, 980, 980 Ti, Titan X).
        // .flag("-gencode")
        // .flag("arch=compute_53,code=sm_53") // for (Jetson TX1).
        // .flag("-gencode")
        // .flag("arch=compute_61,code=sm_61") // for (GTX 1070,1080,1080Ti, Titan Xp).
        // .flag("-gencode")
        // .flag("arch=compute_60,code=sm_60") // for (Tesla P100).
        // .flag("-gencode")
        // .flag("arch=compute_62,code=sm_62") // for (Jetson TX2).
        .flag("-gencode")
        .flag("arch=compute_75,code=sm_75") // for (RTX 2080Ti).
        .files(cuda_files)
        .include("kernel/")
        .compile("libkernel.a");
    println!("cargo:rustc-link-lib=kernel");

    for path in find_cuda() {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stub");
    println!("cargo:rustc-link-lib=cuda");

    let bindings = bindgen::Builder::default()
        .ctypes_prefix("::libc")
        .size_t_is_usize(true)
        .clang_arg("-I")
        .clang_arg("/usr/local/cuda/include".to_string())
        .header("kernel/kernel.h")
        .default_alias_style(bindgen::AliasVariation::TypeAlias)
        .rustfmt_bindings(true)
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Unable to write");
}
