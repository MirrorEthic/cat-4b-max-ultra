plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.mirrorethic.cat4b"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.mirrorethic.cat4b"
        minSdk = 31
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
    }

    // llama-server ships as a "native library" so Android extracts it somewhere executable
    // (apps may only exec from nativeLibraryDir since API 29).
    packaging { jniLibs { useLegacyPackaging = true } }

    sourceSets.getByName("main") {
        // the served system prompts, single-sourced from the repo's prompts/ (byte-exact, like the gateway)
        assets.srcDir("../../../prompts")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    buildTypes {
        release { isMinifyEnabled = false }
    }
}
