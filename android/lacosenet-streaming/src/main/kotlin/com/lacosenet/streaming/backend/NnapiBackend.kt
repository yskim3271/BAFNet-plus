/**
 * NnapiBackend.kt
 *
 * Android Neural Networks API (NNAPI) Execution Provider backend.
 * Fallback for non-Qualcomm devices or when QNN is unavailable.
 *
 * Note: NNAPI is deprecated in Android 15+ (API 35).
 * Reference: https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html
 */
package com.lacosenet.streaming.backend

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Build
import android.util.Log
import com.lacosenet.streaming.core.StreamingConfig

/**
 * NNAPI Execution Provider backend.
 *
 * Key features:
 * - Hardware acceleration via Android Neural Networks API
 * - Supports various NPU/GPU/DSP backends depending on device
 * - Automatic fallback to CPU for unsupported operations
 *
 * Note: Consider using QNN EP for Qualcomm devices for better performance.
 *
 * @param context Android context (currently unused, kept for consistency)
 */
class NnapiBackend(private val context: Context) : BaseExecutionBackend() {

    companion object {
        private const val TAG = "NnapiBackend"
        private const val NNAPI_PROVIDER = "NnapiExecutionProvider"

        // NNAPI available from Android 8.1 (API 27)
        private const val NNAPI_MIN_API = 27

        // NNAPI deprecated from Android 15 (API 35)
        private const val NNAPI_DEPRECATED_API = 35
    }

    override val type: BackendType = BackendType.NNAPI

    override val isAvailable: Boolean
        get() = Build.VERSION.SDK_INT >= NNAPI_MIN_API

    /**
     * Whether NNAPI is deprecated on this Android version.
     */
    val isDeprecated: Boolean
        get() = Build.VERSION.SDK_INT >= NNAPI_DEPRECATED_API

    override fun initialize(
        env: OrtEnvironment,
        modelPath: String,
        config: StreamingConfig
    ): BackendInitResult {
        val startTime = System.nanoTime()

        try {
            // Check availability
            if (!isAvailable) {
                return BackendInitResult(
                    success = false,
                    backend = type,
                    errorMessage = "NNAPI requires Android 8.1+ (API 27)"
                )
            }

            // Log deprecation warning
            if (isDeprecated) {
                Log.w(TAG, "NNAPI is deprecated in Android 15+. Consider using QNN EP for Qualcomm devices.")
            }

            // Setup session options
            val sessionOptions = OrtSession.SessionOptions()

            // NNAPI-specific options
            // Disable CPU fallback within NNAPI (use ORT CPU instead for unsupported ops)
            // This avoids the inefficient nnapi-reference CPU implementation
            sessionOptions.addConfigEntry("nnapi.disable_cpu", "1")

            // Enable NNAPI
            sessionOptions.addNnapi()

            // Create session
            _session = env.createSession(modelPath, sessionOptions)

            val loadTimeMs = (System.nanoTime() - startTime) / 1_000_000f
            Log.i(TAG, "NNAPI backend initialized in ${loadTimeMs}ms (deprecated=$isDeprecated)")

            return BackendInitResult(
                success = true,
                backend = type,
                session = _session,
                loadTimeMs = loadTimeMs
            )

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize NNAPI backend", e)
            return BackendInitResult(
                success = false,
                backend = type,
                errorMessage = e.message
            )
        }
    }

    /**
     * Create session options with NNAPI EP configured.
     */
    @Suppress("UNUSED_PARAMETER")
    override fun createSessionOptions(config: StreamingConfig): OrtSession.SessionOptions {
        val sessionOptions = OrtSession.SessionOptions()
        sessionOptions.addConfigEntry("nnapi.disable_cpu", "1")
        sessionOptions.addNnapi()
        return sessionOptions
    }
}
