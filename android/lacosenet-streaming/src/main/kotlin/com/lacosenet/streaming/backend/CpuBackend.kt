/**
 * CpuBackend.kt
 *
 * CPU Execution Provider backend.
 * Always available as a fallback when hardware acceleration is unavailable.
 *
 * Reference: https://onnxruntime.ai/docs/execution-providers/CPU-ExecutionProvider.html
 */
package com.lacosenet.streaming.backend

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import com.lacosenet.streaming.core.StreamingConfig

/**
 * CPU Execution Provider backend.
 *
 * Key features:
 * - Always available on all devices
 * - Uses ONNX Runtime optimized CPU kernels
 * - Supports both FP32 and INT8 models
 * - Good for debugging and correctness verification
 */
class CpuBackend : BaseExecutionBackend() {

    companion object {
        private const val TAG = "CpuBackend"
        private const val CPU_PROVIDER = "CPUExecutionProvider"
    }

    override val type: BackendType = BackendType.CPU

    override val isAvailable: Boolean = true  // Always available

    override fun initialize(
        env: OrtEnvironment,
        modelPath: String,
        config: StreamingConfig
    ): BackendInitResult {
        val startTime = System.nanoTime()

        try {
            // Setup session options
            val sessionOptions = OrtSession.SessionOptions()

            // CPU-specific optimizations
            // Use all available CPU cores
            sessionOptions.setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())

            // Enable graph optimizations
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

            // Create session with CPU provider only
            _session = env.createSession(modelPath, sessionOptions)

            val loadTimeMs = (System.nanoTime() - startTime) / 1_000_000f
            Log.i(TAG, "CPU backend initialized in ${loadTimeMs}ms")

            return BackendInitResult(
                success = true,
                backend = type,
                session = _session,
                loadTimeMs = loadTimeMs
            )

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize CPU backend", e)
            return BackendInitResult(
                success = false,
                backend = type,
                errorMessage = e.message
            )
        }
    }

    /**
     * Create session options for CPU backend.
     */
    @Suppress("UNUSED_PARAMETER")
    override fun createSessionOptions(config: StreamingConfig): OrtSession.SessionOptions {
        val sessionOptions = OrtSession.SessionOptions()
        sessionOptions.setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
        sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        return sessionOptions
    }
}
