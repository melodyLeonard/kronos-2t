plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.kronos.mobile"
    compileSdk = 36
    ndkVersion = "27.1.12297006"

    defaultConfig {
        applicationId = "com.kronos.mobile"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += "arm64-v8a" }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    sourceSets["main"].jniLibs.srcDir(layout.buildDirectory.dir("rust-jni"))
}

kotlin {
    compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
}

// The engine is one Rust library, built as a cdylib only for this app.
val buildEngine by tasks.registering {
    val repo = rootDir.parentFile
    val out = layout.buildDirectory.dir("rust-jni/arm64-v8a")
    val target = layout.buildDirectory.dir("rust")
    inputs.dir(repo.resolve("src"))
    inputs.file(repo.resolve("Cargo.toml"))
    outputs.dir(out)
    doLast {
        val bin = android.ndkDirectory.resolve("toolchains/llvm/prebuilt/darwin-x86_64/bin")
        val clang = bin.resolve("aarch64-linux-android26-clang").path
        val cargoDir = target.get().asFile
        val proc = ProcessBuilder("cargo", "rustc", "--release", "--lib", "--target", "aarch64-linux-android", "--crate-type", "cdylib")
            .directory(repo)
            .inheritIO()
        proc.environment().apply {
            put("CARGO_TARGET_DIR", cargoDir.path)
            put("CC_aarch64_linux_android", clang)
            put("CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER", clang)
            put("CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS", "-C link-arg=-Wl,-z,max-page-size=16384")
        }
        check(proc.start().waitFor() == 0) { "cargo build failed" }
        out.get().asFile.mkdirs()
        cargoDir.resolve("aarch64-linux-android/release/libkronos_2t.so")
            .copyTo(out.get().asFile.resolve("libkronos_2t.so"), overwrite = true)
    }
}

tasks.named("preBuild") { dependsOn(buildEngine) }
