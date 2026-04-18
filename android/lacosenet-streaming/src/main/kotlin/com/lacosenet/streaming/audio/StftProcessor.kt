/**
 * StftProcessor.kt
 *
 * Pure Kotlin STFT/iSTFT implementation for streaming audio processing.
 * Matches the Python implementation in src/stft.py.
 */
package com.lacosenet.streaming.audio

import com.lacosenet.streaming.core.StftConfig
import kotlin.math.*
import org.jtransforms.fft.FloatFFT_1D

/**
 * STFT/iSTFT processor for streaming audio enhancement.
 *
 * Implements:
 * - Short-Time Fourier Transform (STFT)
 * - Inverse STFT with overlap-add
 * - Magnitude/phase representation with power-law compression
 *
 * @param config STFT configuration
 */
class StftProcessor(private val config: StftConfig) {

    private val nFft = config.nFft
    private val hopSize = config.hopSize
    private val winLength = config.winLength
    private val compressFactor = config.compressFactor

    // Precomputed window
    private val window = createHannWindow(winLength)

    // Precomputed FFT plan (mixed-radix for non-power-of-two n_fft like 400).
    private val fft = FloatFFT_1D(nFft.toLong())

    // Overlap-add tail buffer for streaming iSTFT (matches Python manual_istft_ola).
    private val olaTailSize = winLength - hopSize
    private val olaBuffer = FloatArray(olaTailSize)
    private val olaNorm = FloatArray(olaTailSize)
    private var olaInitialized = false

    // Previous STFT context for streaming
    private var stftContext: FloatArray? = null
    private val contextSize = winLength / 2

    /**
     * Compute STFT and return magnitude and phase.
     *
     * @param audio Input audio samples
     * @param center If true, pad audio to center frames
     * @param advanceSamples If > 0, the next streaming context is saved from
     *     `audio[advanceSamples - contextSize : advanceSamples]` (matches
     *     Python streaming `input_buffer[advance - context_size : advance]`).
     *     If <= 0 (default), falls back to the tail of `audio`.
     * @return Pair of (magnitude, phase) arrays [F, T]
     */
    fun stft(audio: FloatArray, center: Boolean = true, advanceSamples: Int = -1): Pair<FloatArray, FloatArray> {
        // H1: reject non-finite samples before they reach the FFT / ONNX. JTransforms
        // and ORT propagate NaN/Inf silently and downstream metrics become meaningless.
        require(audio.all { it.isFinite() }) {
            "StftProcessor.stft: audio contains NaN/Inf (size=${audio.size})"
        }

        // Optionally prepend context for streaming continuity
        val processAudio = if (stftContext != null && stftContext!!.isNotEmpty()) {
            FloatArray(stftContext!!.size + audio.size).also {
                stftContext!!.copyInto(it)
                audio.copyInto(it, stftContext!!.size)
            }
        } else if (center) {
            // Zero-prepend: matches Python streaming's initial stft_context = zeros(win_size/2).
            // Python never reflect-pads the trailing side in streaming mode.
            FloatArray(contextSize + audio.size).also {
                audio.copyInto(it, contextSize)
            }
        } else {
            audio
        }

        // Calculate number of frames — guard against short input (B2).
        // Without context or with short audio, `size - winLength` can be negative
        // producing numFrames <= 0 and crashing the FloatArray allocation below.
        require(processAudio.size >= winLength) {
            "STFT input too short: ${processAudio.size} samples < winLength=$winLength " +
                "(audio=${audio.size}, context=${stftContext?.size ?: if (center) contextSize else 0})"
        }
        val numFrames = (processAudio.size - winLength) / hopSize + 1
        val freqBins = nFft / 2 + 1

        // Output arrays [F, T] flattened
        val magnitude = FloatArray(freqBins * numFrames)
        val phase = FloatArray(freqBins * numFrames)

        // Process each frame
        for (t in 0 until numFrames) {
            val startIdx = t * hopSize

            // Extract and window frame; realForward writes the DFT in-place.
            val frame = FloatArray(nFft)
            for (i in 0 until winLength) {
                frame[i] = processAudio[startIdx + i] * window[i]
            }
            fft.realForward(frame)
            // JTransforms realForward layout for even nFft:
            //   frame[0]         = Re(X[0])        (DC)
            //   frame[1]         = Re(X[N/2])      (Nyquist)
            //   frame[2*k]       = Re(X[k])        for k in 1..N/2-1
            //   frame[2*k + 1]   = Im(X[k])        for k in 1..N/2-1

            for (f in 0 until freqBins) {
                val re: Float
                val im: Float
                when (f) {
                    0 -> { re = frame[0]; im = 0f }
                    freqBins - 1 -> { re = frame[1]; im = 0f }
                    else -> { re = frame[2 * f]; im = frame[2 * f + 1] }
                }
                val mag = sqrt(re * re + im * im)
                val compressedMag = mag.toDouble().pow(compressFactor.toDouble()).toFloat()
                val pha = atan2(im + 1e-8f, re + 1e-8f)

                val idx = f * numFrames + t
                magnitude[idx] = compressedMag
                phase[idx] = pha
            }
        }

        // Update context for next call.
        val advance = if (advanceSamples > 0) advanceSamples else audio.size
        if (advance >= contextSize) {
            stftContext = audio.copyOfRange(advance - contextSize, advance)
        }

        return Pair(magnitude, phase)
    }

    /**
     * Streaming iSTFT with OLA tail carry-over. Matches `manual_istft_ola`
     * in src/stft.py: returns exactly numFrames * hopSize samples and keeps
     * the trailing (winLength - hopSize) samples internally for the next call.
     */
    fun istftStreaming(magnitude: FloatArray, phase: FloatArray, numFrames: Int): FloatArray {
        val freqBins = nFft / 2 + 1
        val outputSamples = numFrames * hopSize
        val totalSize = outputSamples + olaTailSize

        val buf = FloatArray(totalSize)
        val norm = FloatArray(totalSize)

        olaBuffer.copyInto(buf, 0, 0, olaTailSize)
        olaNorm.copyInto(norm, 0, 0, olaTailSize)

        for (t in 0 until numFrames) {
            // Build JTransforms realForward layout from (mag, phase) for this frame.
            val spectrum = FloatArray(nFft)
            for (f in 0 until freqBins) {
                val compressedMag = magnitude[f * numFrames + t]
                val decMag = compressedMag.toDouble().pow(1.0 / compressFactor).toFloat()
                val pha = phase[f * numFrames + t]
                val re = decMag * cos(pha)
                val im = decMag * sin(pha)
                when (f) {
                    0 -> { spectrum[0] = re }
                    freqBins - 1 -> { spectrum[1] = re } // Nyquist
                    else -> {
                        spectrum[2 * f] = re
                        spectrum[2 * f + 1] = im
                    }
                }
            }
            // scale=true divides by N, matching numpy.fft.irfft convention.
            fft.realInverse(spectrum, true)

            val startIdx = t * hopSize
            for (i in 0 until winLength) {
                buf[startIdx + i] += spectrum[i] * window[i]
                norm[startIdx + i] += window[i] * window[i]
            }
        }

        val output = FloatArray(outputSamples)
        for (i in 0 until outputSamples) {
            output[i] = if (norm[i] > 1e-8f) buf[i] / norm[i] else buf[i]
        }

        for (i in 0 until olaTailSize) {
            olaBuffer[i] = buf[outputSamples + i]
            olaNorm[i] = norm[outputSamples + i]
        }

        return output
    }

    /**
     * Reset processor state for new stream.
     */
    fun reset() {
        olaBuffer.fill(0f)
        olaNorm.fill(0f)
        olaInitialized = false
        stftContext = null
    }

    companion object {
        /**
         * Create Hann window.
         */
        fun createHannWindow(size: Int): FloatArray {
            return FloatArray(size) { i ->
                (0.5 * (1 - cos(2 * PI * i / size))).toFloat()
            }
        }
    }
}
