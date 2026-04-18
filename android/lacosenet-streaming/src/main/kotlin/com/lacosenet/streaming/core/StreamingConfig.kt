/**
 * StreamingConfig.kt
 *
 * Configuration data classes for LaCoSENet Android inference.
 * Loaded from streaming_config.json shipped with the model.
 */
package com.lacosenet.streaming.core

import android.content.Context
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import java.io.InputStreamReader

/**
 * Root configuration loaded from streaming_config.json.
 */
data class StreamingConfig(
    @SerializedName("model_info")
    val modelInfo: ModelInfo,

    @SerializedName("stft_config")
    val stftConfig: StftConfig,

    @SerializedName("streaming_config")
    val streamingConfig: StreamingParams,

    @SerializedName("qnn_config")
    val qnnConfig: QnnConfig? = null,

    @SerializedName("state_info")
    val stateInfo: StateInfo? = null,

    @SerializedName("export_info")
    val exportInfo: ExportInfo? = null
) {
    companion object {
        /**
         * Load configuration from assets.
         */
        fun fromAssets(context: Context, path: String = "streaming_config.json"): StreamingConfig {
            return context.assets.open(path).use { stream ->
                InputStreamReader(stream).use { reader ->
                    Gson().fromJson(reader, StreamingConfig::class.java)
                }
            }.also { it.validate(path) }
        }

        /**
         * Load configuration from a file path.
         */
        fun fromFile(path: String): StreamingConfig {
            return java.io.File(path).reader().use { reader ->
                Gson().fromJson(reader, StreamingConfig::class.java)
            }.also { it.validate(path) }
        }
    }

    /**
     * Verify that required fields were actually present in the source JSON (C2).
     * Gson silently leaves missing Int fields at 0, which previously allowed a
     * bundled model to load with wildly different streaming geometry than the
     * one it was exported for. Throw immediately so mismatches are loud.
     */
    fun validate(source: String = "streaming_config.json") {
        val p = streamingConfig
        require(p.chunkSizeFrames > 0) {
            "$source: streaming_config.chunk_size_frames missing or invalid (got ${p.chunkSizeFrames})"
        }
        require(p.encoderLookahead >= 0) {
            "$source: streaming_config.encoder_lookahead missing (got ${p.encoderLookahead})"
        }
        require(p.decoderLookahead >= 0) {
            "$source: streaming_config.decoder_lookahead missing (got ${p.decoderLookahead})"
        }
        require(p.exportTimeFrames > 0) {
            "$source: streaming_config.export_time_frames missing or invalid (got ${p.exportTimeFrames})"
        }
        require(p.freqBins > 0) {
            "$source: streaming_config.freq_bins missing or invalid (got ${p.freqBins})"
        }
        require(p.freqBinsEncoded > 0) {
            "$source: streaming_config.freq_bins_encoded missing or invalid (got ${p.freqBinsEncoded})"
        }
        require(p.freqBins == stftConfig.freqBins) {
            "$source: streaming_config.freq_bins=${p.freqBins} != stft_config.freqBins=${stftConfig.freqBins}"
        }
    }

    /**
     * Calculate samples per chunk based on STFT and streaming parameters.
     */
    val samplesPerChunk: Int
        get() {
            val totalFrames = streamingConfig.chunkSizeFrames + streamingConfig.inputLookaheadFrames
            return (totalFrames - 1) * stftConfig.hopSize + stftConfig.winLength / 2
        }

    /**
     * Calculate output samples per chunk.
     */
    val outputSamplesPerChunk: Int
        get() = streamingConfig.chunkSizeFrames * stftConfig.hopSize

    /**
     * Calculate algorithmic latency in milliseconds.
     */
    val latencyMs: Float
        get() {
            val lookaheadSamples = streamingConfig.totalLookahead * stftConfig.hopSize
            return lookaheadSamples.toFloat() / stftConfig.sampleRate * 1000f
        }
}

/**
 * Model metadata.
 */
data class ModelInfo(
    @SerializedName("name")
    val name: String,

    @SerializedName("version")
    val version: String = "1.0.0",

    @SerializedName("export_format")
    val exportFormat: String = "stateful_nncore",

    @SerializedName("quantization")
    val quantization: String = "int8_qdq",

    @SerializedName("phase_output_mode")
    val phaseOutputMode: String = "complex",

    @SerializedName("qnn_compatible")
    val qnnCompatible: Boolean = true,

    @SerializedName("supported_backends")
    val supportedBackends: List<String> = listOf("qnn_htp", "nnapi", "cpu"),

    @SerializedName("infer_type")
    val inferType: String = "masking"
)

/**
 * STFT configuration.
 */
data class StftConfig(
    @SerializedName("n_fft")
    val nFft: Int = 400,

    @SerializedName("hop_size")
    val hopSize: Int = 100,

    @SerializedName("win_length")
    val winLength: Int = 400,

    @SerializedName("sample_rate")
    val sampleRate: Int = 16000,

    @SerializedName("center")
    val center: Boolean = true,

    @SerializedName("compress_factor")
    val compressFactor: Float = 0.3f
) {
    /**
     * Number of frequency bins (n_fft / 2 + 1).
     */
    val freqBins: Int
        get() = nFft / 2 + 1
}

/**
 * Streaming inference parameters.
 */
data class StreamingParams(
    // Model-specific streaming params: NO defaults. Mismatched fallback values
    // (previously chunk_size_frames=32 vs JSON=8 etc.) silently produced a
    // completely different model geometry. Use 0-sentinel + validate() to force
    // failure when fields are missing from JSON (C2).
    @SerializedName("chunk_size_frames")
    val chunkSizeFrames: Int = 0,

    @SerializedName("encoder_lookahead")
    val encoderLookahead: Int = -1,

    @SerializedName("decoder_lookahead")
    val decoderLookahead: Int = -1,

    @SerializedName("export_time_frames")
    val exportTimeFrames: Int = 0,

    @SerializedName("freq_bins")
    val freqBins: Int = 0,

    @SerializedName("freq_bins_encoded")
    val freqBinsEncoded: Int = 0
) {
    /**
     * Input lookahead in frames.
     *
     * C3: matches Python `self.input_lookahead_frames = int(encoder_lookahead)` in
     * src/models/streaming/lacosenet.py:162. The previous Kotlin formula
     * `maxOf(stftLookaheadFrames=1, encoderLookahead)` inflated lookahead when
     * encoder_lookahead was 0 because the STFT lookahead contribution is already
     * counted in `samplesPerChunk` via `+ winLength / 2` (i.e. `stft_future_samples`).
     */
    val inputLookaheadFrames: Int
        get() = encoderLookahead

    /**
     * Total lookahead = input_lookahead + decoder_lookahead.
     */
    val totalLookahead: Int
        get() = inputLookaheadFrames + decoderLookahead
}

/**
 * QNN Execution Provider configuration.
 */
data class QnnConfig(
    @SerializedName("target_soc")
    val targetSoc: String? = "SM8550",

    @SerializedName("htp_performance_mode")
    val htpPerformanceMode: String = "burst",

    @SerializedName("context_cache_enabled")
    val contextCacheEnabled: Boolean = true,

    @SerializedName("vtcm_mb")
    val vtcmMb: Int = 16,

    @SerializedName("enable_htp_fp16_precision")
    val enableHtpFp16Precision: Boolean = false,

    /** Graph finalization optimization mode: 0=default, 1=faster, 2=longer, 3=longest (best performance) */
    @SerializedName("htp_graph_finalization_optimization_mode")
    val htpGraphFinalizationOptimizationMode: Int = 3,

    /** Enable shared memory allocator for reduced memory copy overhead */
    @SerializedName("enable_htp_shared_memory_allocator")
    val enableHtpSharedMemoryAllocator: Boolean = true,

    /** Context priority for NPU scheduling: low, normal, normal_high, high */
    @SerializedName("qnn_context_priority")
    val contextPriority: String = "high",

    /** Enable profiling for performance analysis: off, basic, detailed */
    @SerializedName("profiling_level")
    val profilingLevel: String = "off"
)

/**
 * State tensor information.
 */
data class StateInfo(
    @SerializedName("num_states")
    val numStates: Int,

    @SerializedName("state_names")
    val stateNames: List<String> = emptyList()
)

/**
 * Export provenance information.
 */
data class ExportInfo(
    @SerializedName("timestamp")
    val timestamp: String? = null,

    @SerializedName("checkpoint_md5")
    val checkpointMd5: String? = null,

    @SerializedName("git_commit")
    val gitCommit: String? = null
)
