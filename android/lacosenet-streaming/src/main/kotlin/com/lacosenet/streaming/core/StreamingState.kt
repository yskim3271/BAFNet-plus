/**
 * StreamingState.kt
 *
 * Data classes for streaming inference results and metrics.
 */
package com.lacosenet.streaming.core

/**
 * Result of a single inference step.
 */
data class InferenceResult(
    /**
     * Estimated magnitude mask or mapping [1, F, T].
     */
    val estMask: FloatArray,

    /**
     * Estimated phase [1, F, T] (computed from phase_real/phase_imag via atan2).
     */
    val estPhase: FloatArray,

    /**
     * Inference time in milliseconds.
     */
    val inferenceTimeMs: Float = 0f
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as InferenceResult
        return estMask.contentEquals(other.estMask) &&
               estPhase.contentEquals(other.estPhase)
    }

    override fun hashCode(): Int {
        var result = estMask.contentHashCode()
        result = 31 * result + estPhase.contentHashCode()
        return result
    }
}

/**
 * Metrics collected during inference.
 */
data class StreamingMetrics(
    /**
     * Number of chunks processed.
     */
    var chunksProcessed: Long = 0,

    /**
     * Total inference time in milliseconds.
     */
    var totalInferenceTimeMs: Float = 0f,

    /**
     * Peak inference time in milliseconds.
     */
    var peakInferenceTimeMs: Float = 0f,

    /**
     * Number of state resets.
     */
    var stateResets: Int = 0
) {
    /**
     * Average inference time per chunk.
     */
    val avgInferenceTimeMs: Float
        get() = if (chunksProcessed > 0) totalInferenceTimeMs / chunksProcessed else 0f

    /**
     * Reset all metrics.
     */
    fun reset() {
        chunksProcessed = 0
        totalInferenceTimeMs = 0f
        peakInferenceTimeMs = 0f
        stateResets = 0
    }
}
