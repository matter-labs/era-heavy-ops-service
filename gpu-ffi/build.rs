extern crate bindgen;
use std::{env, path::PathBuf};

// build.rs

fn main() {
    let bellman_cuda_path = if let Ok(path) = std::env::var("BELLMAN_CUDA_DIR") {
        path
    } else {
        // we need to instruct rustc so that it will find libbellman-cuda.a
        //   - if dep is resolved via git(cargo checks ~/.cargo/git/checkouts/)
        //   - if dep is resolved via local path
        //   - if you want to build on a macos or only for rust analyzer
        //      just `export BELLMAN_CUDA_DIR=$PWD/bellman-cuda`
        // so we will benefit from env variable for now
        todo!("set BELLMAN_CUDA_DIR=$PWD")
    };

    generate_bindings(&bellman_cuda_path);

    #[cfg(not(target_os = "macos"))]
    link_multiexp_library(&bellman_cuda_path);
}

fn generate_bindings(bellman_cuda_path: &str) {
    println!("generating bindings");
    let header_file = &format!("{}/src/bellman-cuda.h", bellman_cuda_path);
    const OUT_FILE: &str = "bindings.rs";
    println!("cargo:rerun-if-changed={}", header_file);

    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(header_file)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(format!(
        "{}/{}",
        env::current_dir().unwrap().to_str().unwrap(),
        "src"
    ));
    println!("out path {:?}", out_path.to_str());
    bindings
        .write_to_file(out_path.join(OUT_FILE))
        .expect("Couldn't write bindings!");
}

fn link_multiexp_library(bellman_cuda_path: &str) {
    let kind = "static";
    let name = "bellman-cuda";

    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=static=cudadevrt");
    println!(
        "cargo:rustc-link-search=native={}/build/src",
        bellman_cuda_path
    );
    println!("cargo:rustc-link-lib={}={}", kind, name);
}