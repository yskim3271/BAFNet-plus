/**
 * StreamingEnhancer.kt
 *
 * Main public API for LaCoSENet Android inference.
 * Provides high-level streaming speech enhancement functionality.
 */
package com.lacosenet.streaming

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.util.Log
import com.lacosenet.streaming.audio.AudioBuffer
import com.lacosenet.streaming.audio.FeatureBuffer
import com.lacosenet.streaming.audio.StftProcessor
import com.lacosenet.streaming.backend.*
import com.lacosenet.streaming.core.*
import com.lacosenet.streaming.session.StatefulInference
import java.io.File
import java.io.FileOutputStream

/**
 * Result of StreamingEnhancer initialization.
 */
data class InitResult(
    val success: Boolean,
    val backend: BackendType,
    val latencyMs: Float,
    val loadTimeMs: Float,
    val errorMessage: String? = null
)

/**
 * Main entry point for streaming speech enhancement on Android.
 *
 * This class provides:
 * - Automatic backend selection (QNN HTP > NNAPI > CPU)
 * - Streaming audio processing with state management
 * - STFT/iSTFT on host, neural network on accelerator
 *
 * Usage:
 * ```kotlin
 * val enhancer = StreamingEnhancer(context)
 * val result = enhancer.initialize("model.onnx", "streaming_config.json")
 *
 * if (result.success) {
 *     // Process audio in chunks
 *     val enhanced = enhancer.processChunk(audioChunk)
 *     if (enhanced != null) {
 *         // Play or save enhanced audio
 *     }
 * }
 *
 * enhancer.release()
 * ```
 *
 * @param context Android context for resource access
 */
class StreamingEnhancer(private val context: Context) {

    companion object {
        private const val TAG = "StreamingEnhancer"
    }

    // Components
    private var env: OrtEnvironment? = null
    private var backend: ExecutionBackend? = null
    private var inference: StatefulInference? = null
    private var stftProcessor: StftProcessor? = null
    private var config: StreamingConfig? = null

    // Buffers
    private var inputBuffer: AudioBuffer? = null
    private var featureBuffer: FeatureBuffer? = null

    // State
    private var isInitialized = false

    // Metrics
    private val metrics = StreamingMetrics()

    /**
     * Currently selected backend type.
     */
    val currentBackend: BackendType?
        get() = backend?.type

    /**
     * Algorithmic latency in milliseconds.
     */
    val latencyMs: Float
        get() = config?.latencyMs ?: 0f

    /**
     * Get current performance metrics.
     */
    fun getMetrics(): StreamingMetrics = metrics.copy()

    /**
     * Get the inference runner for direct benchmarking.
     * This bypasses STFT/iSTFT processing.
     */
    fun getInferenceRunner(): StatefulInference? = inference

    /**
     * Initialize the enhancer with model and configuration.
     *
     * @param modelPath Path to ONNX model (in assets or file system)
     * @param configPath Path to streaming_config.json (in assets or file system)
     * @param forceBackend Force a specific backend (null for auto-selection)
     * @return Initialization result
     */
    fun initialize(
        modelPath: String = "model.onnx",
        configPath: String = "streaming_config.json",
        forceBackend: BackendType? = null
    ): InitResult {
        val startTime = System.nanoTime()

        try {
            // Load configuration
            config = try {
                StreamingConfig.fromAssets(context, configPath)
            } catch (e: Exception) {
                Log.d(TAG, "Config not in assets, trying file path: $configPath")
                StreamingConfig.fromFile(configPath)
            }

            Log.i(TAG, "Loaded config: ${config!!.modelInfo.name}")

            // Copy model from assets if needed
            val modelFile = prepareModelFile(modelPath)

            // Create ONNX Runtime environment
            env = OrtEnvironment.getEnvironment()

            // Select backend
            val selector = BackendSelector(context)
            val selectedBackend = forceBackend ?: selector.selectBestBackend(config)
            backend = selector.createBackend(selectedBackend)

            Log.i(TAG, "Selected backend: $selectedBackend")

            // Initialize unified model
            val backendResult = backend!!.initialize(env!!, modelFile.absolutePath, config!!)

            if (!backendResult.success) {
                // Try fallback backends. Null out the primary so a second release()
                // path (e.g. tryFallbackBackends success or outer catch) does not
                // double-release the failed-init backend (B5).
                Log.w(TAG, "Primary backend failed: ${backendResult.errorMessage}")
                val failedType = backend!!.type
                backend?.release()
                backend = null
                val fallbackResult = tryFallbackBackends(selector, modelFile.absolutePath, failedType)
                if (!fallbackResult.success) {
                    return InitResult(
                        success = false,
                        backend = BackendType.CPU,
                        latencyMs = 0f,
                        loadTimeMs = 0f,
                        errorMessage = fallbackResult.errorMessage
                    )
                }
            }

            // Initialize inference runner
            inference = StatefulInference(env!!, backend!!, config!!)
            inference!!.initialize()

            // Initialize STFT processor
            stftProcessor = StftProcessor(config!!.stftConfig)

            // Initialize buffers
            val streamingParams = config!!.streamingConfig
            val stftConfig = config!!.stftConfig

            inputBuffer = AudioBuffer(config!!.samplesPerChunk * 4)
            featureBuffer = FeatureBuffer(
                stftConfig.freqBins,
                streamingParams.exportTimeFrames * 2
            )

            isInitialized = true
            metrics.reset()

            val loadTimeMs = (System.nanoTime() - startTime) / 1_000_000f

            Log.i(TAG, "Initialization complete in ${loadTimeMs}ms")
            Log.i(TAG, "  Backend: ${backend!!.type}")
            Log.i(TAG, "  Latency: ${config!!.latencyMs}ms")

            return InitResult(
                success = true,
                backend = backend!!.type,
                latencyMs = config!!.latencyMs,
                loadTimeMs = loadTimeMs
            )

        } catch (e: Exception) {
            Log.e(TAG, "Initialization failed", e)
            release()
            return InitResult(
                success = false,
                backend = BackendType.CPU,
                latencyMs = 0f,
                loadTimeMs = 0f,
                errorMessage = e.message
            )
        }
    }

    /**
     * Process a chunk of audio samples.
     *
     * @param samples Input audio samples (PCM float, 16kHz)
     * @return Enhanced audio samples, or null if still buffering
     */
    fun processChunk(samples: FloatArray): FloatArray? {
        if (!isInitialized) {
            throw IllegalStateException("Enhancer not initialized. Call initialize() first.")
        }
        // H1: reject NaN/Inf at the public API boundary. Passing non-finite samples
        // into ONNX yields backend-undefined behavior (silent zeros on CPU, HTP hang).
        require(samples.all { it.isFinite() }) {
            "processChunk: input contains NaN/Inf (size=${samples.size})"
        }

        // Push to input buffer
        inputBuffer!!.push(samples)

        // Check if we have enough samples
        if (!inputBuffer!!.hasEnough(config!!.samplesPerChunk)) {
            return null
        }

        // Get samples for processing
        val chunkSamples = inputBuffer!!.pop(config!!.samplesPerChunk)

        // Compute STFT. Advance by output_samples_per_chunk so the saved
        // stftContext tracks Python streaming's input_buffer slide boundary.
        val (mag, pha) = stftProcessor!!.stft(
            chunkSamples,
            advanceSamples = config!!.outputSamplesPerChunk,
        )

        // Add to feature buffer for lookahead
        val streamingParams = config!!.streamingConfig
        val frameSize = config!!.stftConfig.freqBins

        // Extract frames and add to buffer
        val numInputFrames = mag.size / frameSize
        for (t in 0 until numInputFrames) {
            val frameMag = FloatArray(frameSize)
            val framePha = FloatArray(frameSize)
            for (f in 0 until frameSize) {
                frameMag[f] = mag[f * numInputFrames + t]
                framePha[f] = pha[f * numInputFrames + t]
            }
            featureBuffer!!.push(frameMag, framePha)
        }

        // Check if we have enough frames for processing
        val totalNeeded = streamingParams.exportTimeFrames
        if (!featureBuffer!!.hasEnough(totalNeeded)) {
            return null
        }

        // Get features for neural network
        val (magInput, phaInput) = featureBuffer!!.get(totalNeeded)

        // Run inference
        val result = inference!!.run(magInput, phaInput)

        // Apply mask to get enhanced magnitude
        val estMag = inference!!.applyMask(magInput, result.estMask)

        // Crop mag/phase from totalNeeded (11) to chunk_size_frames (8) before iSTFT,
        // matching Python streaming pipeline (lacosenet.py: est_mag[:, :, :chunk_size]).
        val chunkFrames = streamingParams.chunkSizeFrames
        val cropMag = FloatArray(frameSize * chunkFrames)
        val cropPha = FloatArray(frameSize * chunkFrames)
        for (f in 0 until frameSize) {
            for (t in 0 until chunkFrames) {
                cropMag[f * chunkFrames + t] = estMag[f * totalNeeded + t]
                cropPha[f * chunkFrames + t] = result.estPhase[f * totalNeeded + t]
            }
        }

        // Streaming iSTFT with OLA tail carry-over: returns exactly chunkFrames * hopSize samples.
        val enhanced = stftProcessor!!.istftStreaming(cropMag, cropPha, chunkFrames)

        // Remove processed frames from buffer
        featureBuffer!!.removeFirst(chunkFrames)

        // Update metrics
        metrics.chunksProcessed++
        metrics.totalInferenceTimeMs += result.inferenceTimeMs
        if (result.inferenceTimeMs > metrics.peakInferenceTimeMs) {
            metrics.peakInferenceTimeMs = result.inferenceTimeMs
        }

        return enhanced
    }

    /**
     * Reset state for a new audio stream.
     */
    fun reset() {
        inference?.resetStates()
        stftProcessor?.reset()
        inputBuffer?.clear()
        featureBuffer?.clear()
        metrics.stateResets++
    }

    /**
     * Release all resources.
     */
    fun release() {
        inference?.release()
        backend?.release()
        env?.close()

        inference = null
        backend = null
        env = null
        stftProcessor = null
        inputBuffer = null
        featureBuffer = null
        config = null
        isInitialized = false
    }

    /**
     * Prepare model file (copy from assets if needed).
     */
    private fun prepareModelFile(modelPath: String): File {
        // Check if it's already a file path
        val directFile = File(modelPath)
        if (directFile.exists()) {
            return directFile
        }

        // Try to copy from assets
        val targetFile = File(context.filesDir, modelPath)
        if (!targetFile.exists()) {
            Log.d(TAG, "Copying model from assets: $modelPath")
            context.assets.open(modelPath).use { input ->
                FileOutputStream(targetFile).use { output ->
                    input.copyTo(output)
                }
            }
        }

        return targetFile
    }

    /**
     * Try fallback backends if primary fails.
     *
     * @param excludeType Backend type that already failed (skipped in the chain).
     */
    private fun tryFallbackBackends(
        selector: BackendSelector,
        modelPath: String,
        excludeType: BackendType? = null
    ): BackendInitResult {
        val backends = selector.createBackendChain(config)

        for (fallback in backends) {
            if (fallback.type == excludeType) continue  // Skip failed backend

            Log.i(TAG, "Trying fallback: ${fallback.type}")
            val result = fallback.initialize(env!!, modelPath, config!!)

            if (result.success) {
                backend = fallback
                return result
            }
        }

        return BackendInitResult(
            success = false,
            backend = BackendType.CPU,
            errorMessage = "All backends failed"
        )
    }

    /**
     * Get device capability information.
     */
    fun getDeviceInfo(): Map<String, String> {
        return BackendSelector(context).getDeviceInfo()
    }
}
