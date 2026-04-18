/**
 * QnnBackend.kt
 *
 * QNN (Qualcomm AI Engine Direct) Execution Provider backend.
 * Uses Hexagon Tensor Processor (HTP/NPU) for hardware-accelerated inference.
 *
 * Reference: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
 */
package com.lacosenet.streaming.backend

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.lacosenet.streaming.core.QnnConfig
import com.lacosenet.streaming.core.StreamingConfig
import java.io.File
import java.security.MessageDigest

/**
 * QNN Execution Provider backend for Qualcomm Snapdragon devices.
 *
 * Key features:
 * - HTP (Hexagon Tensor Processor) NPU acceleration
 * - INT8 quantized model support (required for HTP)
 * - Context binary caching for faster subsequent loads
 *
 * @param context Android context for cache directory access
 */
class QnnBackend(private val context: Context) : BaseExecutionBackend() {

    companion object {
        private const val TAG = "QnnBackend"
        private const val QNN_PROVIDER = "QNNExecutionProvider"
        private const val CONTEXT_CACHE_PREFIX = "qnn_ctx_"

        // B6: cache libQnnHtp.so availability so repeated isAvailable checks
        // (initialize(), logs, device info) do not re-enter System.loadLibrary.
        private val qnnLibLoaded: Boolean by lazy {
            try {
                System.loadLibrary("QnnHtp")
                true
            } catch (e: UnsatisfiedLinkError) {
                false
            }
        }
    }

    override val type: BackendType = BackendType.QNN_HTP

    override val isAvailable: Boolean
        get() = qnnLibLoaded

    private var contextCachePath: String? = null

    override fun initialize(
        env: OrtEnvironment,
        modelPath: String,
        config: StreamingConfig
    ): BackendInitResult {
        val startTime = System.nanoTime()

        try {
            // Check QNN availability
            if (!isAvailable) {
                return BackendInitResult(
                    success = false,
                    backend = type,
                    errorMessage = "QNN libraries not available"
                )
            }

            // Check model compatibility
            // QNN HTP supports: INT8 (best performance), FP16 (via enable_htp_fp16_precision)
            val quantization = config.modelInfo.quantization.lowercase()
            val isInt8 = quantization.contains("int8")
            val isFp32WithFp16 = (quantization.contains("fp32") || quantization.contains("fp16")) &&
                                  config.qnnConfig?.enableHtpFp16Precision == true

            if (!isInt8 && !isFp32WithFp16) {
                return BackendInitResult(
                    success = false,
                    backend = type,
                    errorMessage = "QNN HTP requires INT8 model or FP32 model with enable_htp_fp16_precision=true"
                )
            }

            // Setup session options
            val sessionOptions = OrtSession.SessionOptions()

            // Get QNN configuration
            val qnnConfig = config.qnnConfig

            // Build provider options
            val providerOptions = buildProviderOptions(qnnConfig)

            // Register QNN EP first (required for both cached and non-cached)
            sessionOptions.addQnn(providerOptions)
            Log.i(TAG, "QNN EP registered successfully")

            // Setup context caching
            if (qnnConfig?.contextCacheEnabled == true) {
                val cachePath = getOrCreateContextCachePath(modelPath)
                contextCachePath = cachePath

                // Check if cached context exists
                if (File(cachePath).exists()) {
                    Log.i(TAG, "Loading from cached QNN context: $cachePath")
                    // Load from cached context (QNN EP already registered)
                    _session = env.createSession(cachePath, sessionOptions)
                } else {
                    Log.i(TAG, "Creating new QNN context (will cache to: $cachePath)")
                    // Add context caching options
                    sessionOptions.addConfigEntry("ep.context_enable", "1")
                    sessionOptions.addConfigEntry("ep.context_file_path", cachePath)
                    sessionOptions.addConfigEntry("ep.context_embed_mode", "1")

                    _session = env.createSession(modelPath, sessionOptions)
                }
            } else {
                // No caching - create session directly
                _session = env.createSession(modelPath, sessionOptions)
            }

            val loadTimeMs = (System.nanoTime() - startTime) / 1_000_000f
            Log.i(TAG, "QNN backend initialized in ${loadTimeMs}ms")

            return BackendInitResult(
                success = true,
                backend = type,
                session = _session,
                loadTimeMs = loadTimeMs
            )

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize QNN backend", e)
            return BackendInitResult(
                success = false,
                backend = type,
                errorMessage = e.message
            )
        }
    }

    private fun buildProviderOptions(qnnConfig: QnnConfig?): Map<String, String> {
        val options = mutableMapOf<String, String>()

        // Get the native library directory for this app
        val nativeLibDir = context.applicationInfo.nativeLibraryDir
        val htpBackendPath = "$nativeLibDir/libQnnHtp.so"
        val skelPath = "$nativeLibDir/libQnnHtpV73Skel.so"
        val systemPath = "$nativeLibDir/libQnnSystem.so"

        Log.d(TAG, "Native library dir: $nativeLibDir")
        Log.d(TAG, "HTP backend path: $htpBackendPath (exists: ${File(htpBackendPath).exists()})")
        Log.d(TAG, "Skel path exists: ${File(skelPath).exists()}")
        Log.d(TAG, "System path exists: ${File(systemPath).exists()}")

        // Backend library path (required)
        options["backend_path"] = htpBackendPath

        // Apply QNN config options if available
        qnnConfig?.let { config ->
            // HTP performance mode: burst (max speed), high_performance, power_saver
            options["htp_performance_mode"] = config.htpPerformanceMode

            // Graph finalization optimization mode
            // 0=default, 1=faster compile, 2=longer, 3=longest (best runtime performance)
            options["htp_graph_finalization_optimization_mode"] = config.htpGraphFinalizationOptimizationMode.toString()

            // FP16 precision (0=disabled for INT8 model, 1=enabled)
            options["enable_htp_fp16_precision"] = if (config.enableHtpFp16Precision) "1" else "0"

            // Shared memory allocator - reduces memory copy overhead between CPU and HTP
            options["enable_htp_shared_memory_allocator"] = if (config.enableHtpSharedMemoryAllocator) "1" else "0"

            // VTCM (Vector Tensor Compute Memory) size in MB
            // SM8550 supports up to 16MB, improves internal buffer performance
            if (config.vtcmMb > 0) {
                options["vtcm_mb"] = config.vtcmMb.toString()
            }

            // Context priority for NPU scheduling (low, normal, normal_high, high)
            options["qnn_context_priority"] = config.contextPriority

            // Profiling level for performance analysis (off, basic, detailed)
            if (config.profilingLevel != "off") {
                options["profiling_level"] = config.profilingLevel
                // Optionally set profiling file path
                val profilingPath = "${context.cacheDir}/qnn_profile.csv"
                options["profiling_file_path"] = profilingPath
                Log.i(TAG, "QNN profiling enabled: level=${config.profilingLevel}, path=$profilingPath")
            }

            Log.d(TAG, "QNN config applied:")
            Log.d(TAG, "  htp_performance_mode: ${config.htpPerformanceMode}")
            Log.d(TAG, "  htp_graph_finalization_optimization_mode: ${config.htpGraphFinalizationOptimizationMode}")
            Log.d(TAG, "  enable_htp_fp16_precision: ${config.enableHtpFp16Precision}")
            Log.d(TAG, "  enable_htp_shared_memory_allocator: ${config.enableHtpSharedMemoryAllocator}")
            Log.d(TAG, "  vtcm_mb: ${config.vtcmMb}")
            Log.d(TAG, "  qnn_context_priority: ${config.contextPriority}")
            Log.d(TAG, "  profiling_level: ${config.profilingLevel}")
        }

        Log.d(TAG, "QNN provider options: $options")
        return options
    }

    /**
     * Get or create a context cache path for the model.
     * Uses model file hash to ensure cache invalidation on model changes.
     */
    private fun getOrCreateContextCachePath(modelPath: String): String {
        val modelFile = File(modelPath)
        val modelHash = computeFileHash(modelFile)
        val cacheFileName = "${CONTEXT_CACHE_PREFIX}${modelHash}.onnx"
        val cacheDir = context.cacheDir
        return File(cacheDir, cacheFileName).absolutePath
    }

    /**
     * Compute MD5 hash of file for cache key.
     */
    private fun computeFileHash(file: File): String {
        val md = MessageDigest.getInstance("MD5")
        file.inputStream().use { input ->
            val buffer = ByteArray(8192)
            var bytesRead: Int
            while (input.read(buffer).also { bytesRead = it } != -1) {
                md.update(buffer, 0, bytesRead)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }.take(16)
    }

    /**
     * Clear cached QNN context binaries.
     */
    fun clearContextCache() {
        context.cacheDir.listFiles()?.filter {
            it.name.startsWith(CONTEXT_CACHE_PREFIX)
        }?.forEach {
            Log.d(TAG, "Deleting cached context: ${it.name}")
            it.delete()
        }
    }

    override fun release() {
        super.release()
        contextCachePath = null
    }

    /**
     * Create session options with QNN EP configured.
     */
    override fun createSessionOptions(config: StreamingConfig): OrtSession.SessionOptions {
        val sessionOptions = OrtSession.SessionOptions()
        val qnnConfig = config.qnnConfig
        val providerOptions = buildProviderOptions(qnnConfig)
        sessionOptions.addQnn(providerOptions)
        return sessionOptions
    }
}
