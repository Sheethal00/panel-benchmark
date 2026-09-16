plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.panelbench.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.panelbench.app"
        minSdk = 28  // matches target device profile: Android 9.0+ (Snapdragon 845-era, 2018+)
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"

        // Lets us launch the benchmark run via `adb shell am instrument` / intent extras
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    // Avoid re-compressing model files; some runtimes require them uncompressed in the APK
    androidResources {
        noCompress += listOf("tflite", "onnx", "ncnn", "param", "bin")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        resources {
            excludes += "META-INF/*"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")

    // --- Candidate runtimes: comment out ones you're not testing to keep APK small ---

    // TFLite (CPU + NNAPI + GPU delegate)
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
    // Required alongside tensorflow-lite-gpu: GpuDelegateFactory (and the Options class
    // GpuDelegate.Options now extends) live in this separate artifact. Without it, the
    // compiler can locate GpuDelegate.Options but fails to resolve its supertype --
    // a known packaging split (tensorflow/tensorflow#57934), not a version mismatch.
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")

    // ONNX Runtime Mobile (CPU + NNAPI)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")

    // ML Kit on-device text recognition (baseline OCR to compare against custom models)
    implementation("com.google.mlkit:text-recognition:16.0.1")
    // Tasks.await() used by MlKitOcrRuntime to run ML Kit's async API synchronously,
    // matching this harness's timing model. Usually pulled in transitively by ML Kit,
    // pinned explicitly here so the build doesn't rely on that transitive resolution.
    implementation("com.google.android.gms:play-services-tasks:18.2.0")

    // JSON for config + result serialization
    implementation("org.json:json:20240303")

    // CameraX (only needed if you benchmark end-to-end with live camera capture)
    val cameraxVersion = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
}