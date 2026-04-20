plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.lacosenet.benchmark"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.lacosenet.benchmark"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // D4: QNN HTP requires arm64-v8a; the onnxruntime-android-qnn AAR only
        // ships arm64-v8a. Declare explicitly so a future AAR/dep bundling other
        // ABIs cannot silently grow the APK.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    // Parity tests need the streaming library's STFT/StatefulInference.
    androidTestImplementation(project(":lacosenet-streaming"))
    androidTestImplementation(project(":bafnetplus-streaming"))

    // AndroidTest dependencies
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test:runner:1.5.2")
    androidTestImplementation("androidx.test:rules:1.5.0")
    androidTestImplementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.24.2")
}
