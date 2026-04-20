/**
 * DualChannelFeatureBuffer.kt
 *
 * Frame-synchronous feature buffer for BAFNetPlus (BCS + ACS streams).
 * Stores per-frame (bcs_mag, bcs_pha, acs_mag, acs_pha) tuples and exposes
 * them as concatenated [F, T] flattened tensors matching the ONNX graph input
 * layout.
 */
package com.bafnetplus.streaming.audio

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Thread-safe feature buffer holding paired BCS/ACS frames.
 *
 * Matches [com.lacosenet.streaming.audio.FeatureBuffer]'s semantics but stores
 * two channels per frame. The flattened output preserves the Python streaming
 * layout: `[freq_bin, time_frame]` row-major, so the ONNX graph sees
 * `[1, freqBins, numFrames]` when reshaped.
 *
 * @param freqBins Number of frequency bins per frame.
 * @param maxFrames Soft cap — oldest frames are evicted on overflow.
 */
class DualChannelFeatureBuffer(
    private val freqBins: Int,
    private val maxFrames: Int,
) {
    private data class Frame(
        val bcsMag: FloatArray,
        val bcsPha: FloatArray,
        val acsMag: FloatArray,
        val acsPha: FloatArray,
    )

    /**
     * Concatenated [F, T] flattened tensors for a window of frames.
     */
    data class Features(
        val bcsMag: FloatArray,
        val bcsPha: FloatArray,
        val acsMag: FloatArray,
        val acsPha: FloatArray,
    )

    private val frames = mutableListOf<Frame>()
    private val lock = ReentrantLock()

    /** Number of frames currently buffered. */
    val bufferedFrames: Int
        get() = lock.withLock { frames.size }

    /** Push a paired frame; each array must have size == [freqBins]. */
    fun push(
        bcsMag: FloatArray,
        bcsPha: FloatArray,
        acsMag: FloatArray,
        acsPha: FloatArray,
    ) = lock.withLock {
        require(bcsMag.size == freqBins) { "bcsMag size ${bcsMag.size} != $freqBins" }
        require(bcsPha.size == freqBins) { "bcsPha size ${bcsPha.size} != $freqBins" }
        require(acsMag.size == freqBins) { "acsMag size ${acsMag.size} != $freqBins" }
        require(acsPha.size == freqBins) { "acsPha size ${acsPha.size} != $freqBins" }

        frames.add(
            Frame(
                bcsMag = bcsMag.clone(),
                bcsPha = bcsPha.clone(),
                acsMag = acsMag.clone(),
                acsPha = acsPha.clone(),
            )
        )
        while (frames.size > maxFrames) {
            frames.removeAt(0)
        }
    }

    /** True if [count] frames are available. */
    fun hasEnough(count: Int): Boolean = lock.withLock { frames.size >= count }

    /**
     * Return concatenated [F, T] views for the first [count] frames.
     * Layout: `output[f * count + t] = frames[t].channel[f]`.
     */
    fun get(count: Int): Features = lock.withLock {
        require(count > 0) { "count must be > 0" }
        require(count <= frames.size) { "requested $count frames; only ${frames.size} buffered" }

        val total = count * freqBins
        val bcsMag = FloatArray(total)
        val bcsPha = FloatArray(total)
        val acsMag = FloatArray(total)
        val acsPha = FloatArray(total)

        for (t in 0 until count) {
            val frame = frames[t]
            for (f in 0 until freqBins) {
                val idx = f * count + t
                bcsMag[idx] = frame.bcsMag[f]
                bcsPha[idx] = frame.bcsPha[f]
                acsMag[idx] = frame.acsMag[f]
                acsPha[idx] = frame.acsPha[f]
            }
        }

        Features(bcsMag, bcsPha, acsMag, acsPha)
    }

    /** Drop the first [count] frames. */
    fun removeFirst(count: Int) = lock.withLock {
        repeat(minOf(count, frames.size)) {
            frames.removeAt(0)
        }
    }

    /** Clear all buffered frames. */
    fun clear() = lock.withLock {
        frames.clear()
    }
}
