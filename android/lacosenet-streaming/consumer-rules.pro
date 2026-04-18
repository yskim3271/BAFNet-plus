# Consumer ProGuard rules for com.lacosenet.streaming.
# Applied automatically by downstream apps that enable minification.

# --- ONNX Runtime (ai.onnxruntime) ---------------------------------------
# ORT Java uses JNI and reflection extensively; stripping any of these
# classes causes UnsatisfiedLinkError at OrtEnvironment.getEnvironment().
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** {
    native <methods>;
}
-dontwarn ai.onnxruntime.**

# --- JTransforms FFT ------------------------------------------------------
# Used by StftProcessor for real-to-complex and inverse FFT (n_fft=400).
# The library uses reflection for its mixed-radix plan factory.
-keep class org.jtransforms.** { *; }
-keep class pl.edu.icm.jlargearrays.** { *; }
-dontwarn org.jtransforms.**
-dontwarn pl.edu.icm.jlargearrays.**

# --- Library public API ---------------------------------------------------
# Keep all public entry points consumers call.
-keep class com.lacosenet.streaming.StreamingEnhancer { *; }
-keep class com.lacosenet.streaming.core.** { *; }
-keep class com.lacosenet.streaming.backend.BackendSelector { *; }
-keep class com.lacosenet.streaming.backend.ExecutionBackend { *; }

# Gson-parsed config classes (@SerializedName relies on preserved field names).
-keepclassmembers class com.lacosenet.streaming.core.** {
    @com.google.gson.annotations.SerializedName <fields>;
}
