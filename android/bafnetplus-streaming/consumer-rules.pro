# Consumer ProGuard rules for com.bafnetplus.streaming.
# Applied automatically by downstream apps that enable minification.

# --- Library public API ---------------------------------------------------
-keep class com.bafnetplus.streaming.BAFNetPlusStreamingEnhancer { *; }
-keep class com.bafnetplus.streaming.core.** { *; }

# Gson-parsed config classes (@SerializedName relies on preserved field names).
-keepclassmembers class com.bafnetplus.streaming.core.** {
    @com.google.gson.annotations.SerializedName <fields>;
}
