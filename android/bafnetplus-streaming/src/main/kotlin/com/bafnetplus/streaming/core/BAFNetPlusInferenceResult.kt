/**
 * BAFNetPlusInferenceResult.kt
 *
 * Result of a single BAFNetPlus inference step.
 * Unlike LaCoSENet's [com.lacosenet.streaming.core.InferenceResult] — which carries
 * `(estMask, estPhase)` requiring `mag * estMask` on host — BAFNetPlus outputs the
 * post-fusion magnitude directly plus the complex (real, imag) phase. No mask
 * multiplication required; the tuple feeds into iSTFT directly.
 */
package com.bafnetplus.streaming.core

import com.lacosenet.streaming.session.StatefulInference

/**
 * Per-chunk BAFNetPlus output tuple + timing.
 *
 * Arrays reference [StatefulInference]'s preallocated buffers and are overwritten
 * on the next [StatefulInference.run] call. Do not retain across invocations —
 * clone if a persistent copy is needed.
 *
 * @property estMag Enhanced magnitude `[1, freqBins, exportTimeFrames]` (power-law
 *                  compressed, matches STFT input range).
 * @property estComReal Real part of fused complex output, same shape as estMag.
 * @property estComImag Imaginary part of fused complex output, same shape.
 * @property inferenceTimeMs Wall-clock ONNX run time in ms.
 */
data class BAFNetPlusInferenceResult(
    val estMag: FloatArray,
    val estComReal: FloatArray,
    val estComImag: FloatArray,
    val inferenceTimeMs: Float = 0f,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as BAFNetPlusInferenceResult
        return estMag.contentEquals(other.estMag) &&
            estComReal.contentEquals(other.estComReal) &&
            estComImag.contentEquals(other.estComImag)
    }

    override fun hashCode(): Int {
        var result = estMag.contentHashCode()
        result = 31 * result + estComReal.contentHashCode()
        result = 31 * result + estComImag.contentHashCode()
        return result
    }

    companion object {
        /** Required graph output names. */
        val REQUIRED_OUTPUTS = setOf("est_mag", "est_com_real", "est_com_imag")

        /** Required audio input names. */
        val REQUIRED_INPUTS = setOf("bcs_mag", "bcs_pha", "acs_mag", "acs_pha")

        /** Convenience — extract a [BAFNetPlusInferenceResult] from a run() output map. */
        fun fromOutputs(outputs: Map<String, FloatArray>, inferenceTimeMs: Float = 0f): BAFNetPlusInferenceResult {
            val estMag = outputs["est_mag"]
                ?: throw IllegalStateException("missing est_mag; got ${outputs.keys}")
            val estComReal = outputs["est_com_real"]
                ?: throw IllegalStateException("missing est_com_real; got ${outputs.keys}")
            val estComImag = outputs["est_com_imag"]
                ?: throw IllegalStateException("missing est_com_imag; got ${outputs.keys}")
            return BAFNetPlusInferenceResult(estMag, estComReal, estComImag, inferenceTimeMs)
        }
    }
}
