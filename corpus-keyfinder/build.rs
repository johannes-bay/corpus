fn main() {
    // Compile the C++ wrapper
    cc::Build::new()
        .cpp(true)
        .file("wrapper.cpp")
        .include("/opt/homebrew/include")
        .flag("-std=c++11")
        .compile("keyfinder_wrapper");

    // Link against libkeyfinder
    println!("cargo:rustc-link-search=/opt/homebrew/lib");
    println!("cargo:rustc-link-lib=keyfinder");
    println!("cargo:rustc-link-lib=c++");
}
