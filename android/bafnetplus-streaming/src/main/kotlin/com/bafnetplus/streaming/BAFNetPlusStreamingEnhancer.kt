/**
 * BAFNetPlusStreamingEnhancer.kt
 *
 * Dual-input (BCS + ACS) streaming enhancement driver.
 * Wraps the shared [StatefulInference] runtime with BAFNetPlus-specific STFT,
 * feature buffering, and complex-output iSTFT.
 */
package com.bafnetplus.streaming

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.util.Log
import com.bafnetplus.streaming.audio.DualChannelFeatureBuffer
import com.lacosenet.streaming.audio.AudioBuffer
import com.lacosenet.streaming.audio.StftProcessor
import com.lacosenet.streaming.backend.BackendInitResult
import com.lacosenet.streaming.backend.BackendSelector
import com.lacosenet.streaming.backend.BackendType
import com.lacosenet.streaming.backend.ExecutionBackend
import com.lacosenet.streaming.core.StreamingConfig
import com.lacosenet.streaming.core.StreamingMetrics
import com.lacosenet.streaming.session.StatefulInference
import java.io.File
import java.io.FileOutputStream
import kotlin.math.atan2

/**
 * Result of [BAFNetPlusStreamingEnhancer] initialization.
 */
data class InitResult(
    val success: Boolean,
    val backend: BackendType,
    val latencyMs: Float,
    val loadTimeMs: Float,
    val numStates: Int,
    val errorMessage: String? = null,
)

/**
 * Dual-input streaming enhancer for BAFNetPlus.
 *
 * Pipeline per chunk:
 *   bcs PCM + acs PCM  ─┐
 *                       ├─▶ per-channel STFT ─▶ DualChannelFeatureBuffer
 *                       │                             │ (chunk_size + encoder_lookahead)
 *                       │                             ▼
 *                       │                      StatefulInference.run(map4)
 *                       │                             │
 *                       │          (est_mag, est_com_real, est_com_imag)
 *                       │                             │
 *                       └──────────────▶ host iSTFT  ◀┘
 *                                               │
 *                                               ▼
 *                                        enhanced PCM (800 samples)
 *
 * Reads `bafnetplus_streaming_config.json` to discover 4 audio inputs + 166 states
 * + 3 primary outputs. Uses [StatefulInference] (shared with LaCoSENet) for the
 * ORT session / state double-buffer management.
 *
 * @param context Android context for asset access.
 */
class BAFNetPlusStreamingEnhancer(private val context: Context) {

    companion object {
        private const val TAG = "BAFNetPlusEnhancer"
        private const val DEFAULT_MODEL = "bafnetplus.onnx"
        private const val DEFAULT_CONFIG = "bafnetplus_streaming_config.json"
    }

    private var env: OrtEnvironment? = null
    private var backend: ExecutionBackend? = null
    private var inference: StatefulInference? = null
    private var stftBcs: StftProcessor? = null
    private var stftAcs: StftProcessor? = null
    private var config: StreamingConfig? = null

    private var bcsInputBuffer: AudioBuffer? = null
    private var acsInputBuffer: AudioBuffer? = null
    private var featureBuffer: DualChannelFeatureBuffer? = null

    private var isInitialized = false
    private val metrics = StreamingMetrics()

    /** Currently selected backend type. */
    val currentBackend: BackendType?
        get() = backend?.type

    /** Algorithmic latency in milliseconds. */
    val latencyMs: Float
        get() = config?.latencyMs ?: 0f

    /** Number of state tensors exposed by the underlying ONNX graph. */
    val numStates: Int
        get() = inference?.numStates ?: 0

    /** Current performance metrics snapshot. */
    fun getMetrics(): StreamingMetrics = metrics.copy()

    /**
     * Initialize the enhancer with BAFNetPlus model + config.
     */
    fun initialize(
        modelPath: String = DEFAULT_MODEL,
        configPath: String = DEFAULT_CONFIG,
        forceBackend: BackendType? = null,
    ): InitResult {
        val startTime = System.nanoTime()
        try {
            config = try {
                StreamingConfig.fromAssets(context, configPath)
            } catch (e: Exception) {
                Log.d(TAG, "Config not in assets, trying file path: $configPath")
                StreamingConfig.fromFile(configPath)
            }
            Log.i(TAG, "Loaded config: ${config!!.modelInfo.name}")

            val modelFile = prepareModelFile(modelPath)
            env = OrtEnvironment.getEnvironment()

            val selector = BackendSelector(context)
            val selectedBackend = forceBackend ?: selector.selectBestBackend(config)
            backend = selector.createBackend(selectedBackend)
            Log.i(TAG, "Selected backend: $selectedBackend")

            val backendResult = backend!!.initialize(env!!, modelFile.absolutePath, config!!)
            if (!backendResult.success) {
                val failedType = backend!!.type
                Log.w(TAG, "Primary backend failed: ${backendResult.errorMessage}")
                backend?.release()
                backend = null
                val fallback = tryFallbackBackends(selector, modelFile.absolutePath, failedType)
                if (!fallback.success) {
                    return InitResult(
                        success = false,
                        backend = BackendType.CPU,
                        latencyMs = 0f,
                        loadTimeMs = 0f,
                        numStates = 0,
                        errorMessage = fallback.errorMessage,
                    )
                }
            }

            inference = StatefulInference(env!!, backend!!, config!!)
            inference!!.initialize()

            // Expect 4 audio inputs (bcs_mag, bcs_pha, acs_mag, acs_pha) + 3 primary
            // outputs (est_mag, est_com_real, est_com_imag).
            val audioInputs = inference!!.audioInputNames.toSet()
            check(audioInputs == setOf("bcs_mag", "bcs_pha", "acs_mag", "acs_pha")) {
                "Unexpected BAFNetPlus audio inputs: $audioInputs"
            }
            val primaries = inference!!.primaryOutputNames.toSet()
            check(primaries == setOf("est_mag", "est_com_real", "est_com_imag")) {
                "Unexpected BAFNetPlus primary outputs: $primaries"
            }

            // Two STFT processors — independent streaming context (BCS + ACS).
            stftBcs = StftProcessor(config!!.stftConfig)
            stftAcs = StftProcessor(config!!.stftConfig)

            val streamingParams = config!!.streamingConfig
            val stftConfig = config!!.stftConfig
            bcsInputBuffer = AudioBuffer(config!!.samplesPerChunk * 4)
            acsInputBuffer = AudioBuffer(config!!.samplesPerChunk * 4)
            featureBuffer = DualChannelFeatureBuffer(
                stftConfig.freqBins,
                streamingParams.exportTimeFrames * 2,
            )

            isInitialized = true
            metrics.reset()

            val loadTimeMs = (System.nanoTime() - startTime) / 1_000_000f
            Log.i(TAG, "Initialization complete in ${loadTimeMs}ms")
            Log.i(TAG, "  Backend: ${backend!!.type}")
            Log.i(TAG, "  Num states: ${inference!!.numStates}")
            Log.i(TAG, "  Latency: ${config!!.latencyMs}ms")

            return InitResult(
                success = true,
                backend = backend!!.type,
                latencyMs = config!!.latencyMs,
                loadTimeMs = loadTimeMs,
                numStates = inference!!.numStates,
            )
        } catch (e: Exception) {
            Log.e(TAG, "Initialization failed", e)
            release()
            return InitResult(
                success = false,
                backend = BackendType.CPU,
                latencyMs = 0f,
                loadTimeMs = 0f,
                numStates = 0,
                errorMessage = e.message,
            )
        }
    }

    /**
     * Expose the inference runner for parity/benchmark tests (no ownership transfer).
     */
    fun getInferenceRunner(): StatefulInference? = inference

    /**
     * Expose the config (useful for tests to read state layout, STFT params etc.).
     */
    fun getConfig(): StreamingConfig? = config

    /**
     * Process a paired chunk of BCS + ACS samples.
     *
     * @param bcs Body-conducted (BCS) samples for this chunk.
     * @param acs Air-conducted (ACS) samples for this chunk. Must be the same
     *            length as [bcs] (both streams are frame-synchronous).
     * @return Enhanced PCM samples (exactly `outputSamplesPerChunk = 800` samples)
     *         once enough lookahead has accumulated; null while still buffering.
     */
    fun processChunk(bcs: FloatArray, acs: FloatArray): FloatArray? {
        check(isInitialized) { "Enhancer not initialized; call initialize() first" }
        require(bcs.size == acs.size) {
            "BCS/ACS length mismatch: ${bcs.size} vs ${acs.size}"
        }
        require(bcs.all { it.isFinite() }) { "BCS contains NaN/Inf (size=${bcs.size})" }
        require(acs.all { it.isFinite() }) { "ACS contains NaN/Inf (size=${acs.size})" }

        bcsInputBuffer!!.push(bcs)
        acsInputBuffer!!.push(acs)

        val samplesPerChunk = config!!.samplesPerChunk
        if (!bcsInputBuffer!!.hasEnough(samplesPerChunk) || !acsInputBuffer!!.hasEnough(samplesPerChunk)) {
            return null
        }

        val bcsChunk = bcsInputBuffer!!.pop(samplesPerChunk)
        val acsChunk = acsInputBuffer!!.pop(samplesPerChunk)

        val outputSamples = config!!.outputSamplesPerChunk
        val (bcsMag, bcsPha) = stftBcs!!.stft(bcsChunk, advanceSamples = outputSamples)
        val (acsMag, acsPha) = stftAcs!!.stft(acsChunk, advanceSamples = outputSamples)

        val streamingParams = config!!.streamingConfig
        val frameSize = config!!.stftConfig.freqBins

        // Push all transformed frames (same number for both channels).
        val numInputFrames = bcsMag.size / frameSize
        require(numInputFrames == acsMag.size / frameSize) {
            "BCS/ACS STFT frame count mismatch ($numInputFrames vs ${acsMag.size / frameSize})"
        }
        for (t in 0 until numInputFrames) {
            val bMag = FloatArray(frameSize)
            val bPha = FloatArray(frameSize)
            val aMag = FloatArray(frameSize)
            val aPha = FloatArray(frameSize)
            for (f in 0 until frameSize) {
                bMag[f] = bcsMag[f * numInputFrames + t]
                bPha[f] = bcsPha[f * numInputFrames + t]
                aMag[f] = acsMag[f * numInputFrames + t]
                aPha[f] = acsPha[f * numInputFrames + t]
            }
            featureBuffer!!.push(bMag, bPha, aMag, aPha)
        }

        val totalNeeded = streamingParams.exportTimeFrames
        if (!featureBuffer!!.hasEnough(totalNeeded)) {
            return null
        }

        val features = featureBuffer!!.get(totalNeeded)

        val inferStart = System.nanoTime()
        val outputs = inference!!.run(
            mapOf(
                "bcs_mag" to features.bcsMag,
                "bcs_pha" to features.bcsPha,
                "acs_mag" to features.acsMag,
                "acs_pha" to features.acsPha,
            )
        )
        val inferenceMs = (System.nanoTime() - inferStart) / 1_000_000f

        val estMag = outputs["est_mag"]
            ?: throw IllegalStateException("missing est_mag output")
        val estComReal = outputs["est_com_real"]
            ?: throw IllegalStateException("missing est_com_real output")
        val estComImag = outputs["est_com_imag"]
            ?: throw IllegalStateException("missing est_com_imag output")

        // Compute host-side phase from complex outputs (mirrors LaCoSENet's
        // complex phase mode). BAFNetPlus has already fused mag → est_mag; no
        // mask multiplication here. The ONNX graph crops to chunk_size_frames
        // internally, so estMag/estCom* already have T = chunk_size_frames.
        val chunkFrames = streamingParams.chunkSizeFrames
        require(estMag.size == frameSize * chunkFrames) {
            "est_mag size ${estMag.size} != frameSize * chunkFrames (${frameSize * chunkFrames})"
        }
        val estPha = FloatArray(estComReal.size) { i ->
            atan2(estComImag[i] + 1e-8f, estComReal[i] + 1e-8f)
        }

        // Shared host iSTFT — BCS stream carries the output OLA buffer.
        val enhanced = stftBcs!!.istftStreaming(estMag, estPha, chunkFrames)

        featureBuffer!!.removeFirst(chunkFrames)

        metrics.chunksProcessed++
        metrics.totalInferenceTimeMs += inferenceMs
        if (inferenceMs > metrics.peakInferenceTimeMs) {
            metrics.peakInferenceTimeMs = inferenceMs
        }

        return enhanced
    }

    /** Reset state for a new audio stream. */
    fun reset() {
        inference?.resetStates()
        stftBcs?.reset()
        stftAcs?.reset()
        bcsInputBuffer?.clear()
        acsInputBuffer?.clear()
        featureBuffer?.clear()
        metrics.stateResets++
    }

    /** Release all resources. */
    fun release() {
        inference?.release()
        backend?.release()
        env?.close()

        inference = null
        backend = null
        env = null
        stftBcs = null
        stftAcs = null
        bcsInputBuffer = null
        acsInputBuffer = null
        featureBuffer = null
        config = null
        isInitialized = false
    }

    private fun prepareModelFile(modelPath: String): File {
        val directFile = File(modelPath)
        if (directFile.exists()) {
            return directFile
        }
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

    private fun tryFallbackBackends(
        selector: BackendSelector,
        modelPath: String,
        excludeType: BackendType? = null,
    ): BackendInitResult {
        val backends = selector.createBackendChain(config)
        for (fallback in backends) {
            if (fallback.type == excludeType) continue
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
            errorMessage = "All backends failed",
        )
    }
}
