/**
 * ExecutionBackend.kt
 *
 * Abstract interface for ONNX Runtime execution backends.
 * Supports QNN (Qualcomm NPU), NNAPI, and CPU backends.
 */
package com.lacosenet.streaming.backend

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.lacosenet.streaming.core.StreamingConfig

/**
 * Supported execution backend types.
 */
enum class BackendType {
    /**
     * Qualcomm AI Engine Direct (QNN) with Hexagon Tensor Processor (HTP/NPU).
     * ONNX Runtime-based backend for Qualcomm Snapdragon SoCs.
     * Requires INT8 quantized ONNX model.
     */
    QNN_HTP,

    /**
     * Android Neural Networks API.
     * Fallback for non-Qualcomm devices or Android < 15.
     * Note: NNAPI is deprecated in Android 15+.
     */
    NNAPI,

    /**
     * CPU execution with ONNX Runtime optimized kernels.
     * Always available as final fallback.
     */
    CPU
}

/**
 * Result of backend initialization.
 */
data class BackendInitResult(
    val success: Boolean,
    val backend: BackendType,
    val session: OrtSession? = null,
    val errorMessage: String? = null,
    val loadTimeMs: Float = 0f
)

/**
 * Abstract interface for execution backends.
 *
 * Each backend implementation handles:
 * - Session creation with provider-specific options
 * - Context caching (for QNN)
 * - Fallback behavior
 */
interface ExecutionBackend {
    /**
     * Backend type identifier.
     */
    val type: BackendType

    /**
     * Whether this backend is available on the current device.
     */
    val isAvailable: Boolean

    /**
     * Whether the backend has been initialized with a model.
     */
    val isInitialized: Boolean

    /**
     * The ONNX Runtime session (null if not initialized).
     */
    val session: OrtSession?

    /**
     * Initialize the backend with a model.
     *
     * @param env ONNX Runtime environment
     * @param modelPath Path to the ONNX model file
     * @param config Streaming configuration
     * @return Initialization result
     */
    fun initialize(
        env: OrtEnvironment,
        modelPath: String,
        config: StreamingConfig
    ): BackendInitResult

    /**
     * Run inference on the model.
     *
     * Caller MUST close the returned OrtSession.Result (e.g. via .use { }) to
     * release the native tensor memory. Leaving Results uncollected accumulates
     * native heap — at QDQ Dual throughput that is ~14 MB per chunk (2026-04-19 B1).
     *
     * @param inputs Map of input name to tensor
     * @return OrtSession.Result (AutoCloseable). Output tensors are only valid
     *         until close(); extract data into Kotlin arrays before closing.
     */
    fun run(inputs: Map<String, OnnxTensor>): OrtSession.Result

    /**
     * Release resources.
     */
    fun release()

    /**
     * Get the last inference time in milliseconds.
     */
    val lastInferenceTimeMs: Float

    /**
     * Create session options for this backend.
     *
     * @param config Streaming configuration
     * @return Configured session options
     */
    fun createSessionOptions(config: StreamingConfig): OrtSession.SessionOptions
}

/**
 * Base implementation with common functionality.
 */
abstract class BaseExecutionBackend : ExecutionBackend {
    protected var _session: OrtSession? = null
    protected var _lastInferenceTimeMs: Float = 0f

    override val session: OrtSession?
        get() = _session

    override val isInitialized: Boolean
        get() = _session != null

    override val lastInferenceTimeMs: Float
        get() = _lastInferenceTimeMs

    override fun run(inputs: Map<String, OnnxTensor>): OrtSession.Result {
        val session = _session
            ?: throw IllegalStateException("Backend not initialized")

        val startTime = System.nanoTime()
        val result = session.run(inputs)
        val endTime = System.nanoTime()

        _lastInferenceTimeMs = (endTime - startTime) / 1_000_000f

        return result
    }

    override fun release() {
        _session?.close()
        _session = null
    }
}
