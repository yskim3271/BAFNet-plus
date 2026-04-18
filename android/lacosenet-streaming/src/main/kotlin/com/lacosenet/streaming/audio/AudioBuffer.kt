/**
 * AudioBuffer.kt
 *
 * Ring buffer implementations for streaming audio processing.
 */
package com.lacosenet.streaming.audio

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Thread-safe ring buffer for audio samples.
 *
 * @param capacity Maximum number of samples to store
 */
class AudioBuffer(private val capacity: Int) {
    private val buffer = FloatArray(capacity)
    private var writePos = 0
    private var readPos = 0
    private var size = 0
    private val lock = ReentrantLock()

    /**
     * Number of samples currently in the buffer.
     */
    val available: Int
        get() = lock.withLock { size }

    /**
     * Check if buffer has at least the specified number of samples.
     */
    fun hasEnough(count: Int): Boolean = lock.withLock { size >= count }

    /**
     * Push samples into the buffer.
     *
     * @param samples Samples to push
     * @return Number of samples actually pushed (may be less if buffer is full)
     */
    fun push(samples: FloatArray): Int = lock.withLock {
        val count = minOf(samples.size, capacity - size)

        for (i in 0 until count) {
            buffer[writePos] = samples[i]
            writePos = (writePos + 1) % capacity
        }
        size += count

        count
    }

    /**
     * Pop samples from the buffer.
     *
     * @param count Number of samples to pop
     * @return Array of popped samples (may be smaller if not enough available)
     */
    fun pop(count: Int): FloatArray = lock.withLock {
        val actualCount = minOf(count, size)
        val result = FloatArray(actualCount)

        for (i in 0 until actualCount) {
            result[i] = buffer[readPos]
            readPos = (readPos + 1) % capacity
        }
        size -= actualCount

        result
    }

    /**
     * Clear all samples from the buffer.
     */
    fun clear() = lock.withLock {
        writePos = 0
        readPos = 0
        size = 0
    }
}

/**
 * Feature buffer for storing spectral frames.
 * Used for decoder lookahead buffering.
 */
class FeatureBuffer(
    private val freqBins: Int,
    private val maxFrames: Int
) {
    private data class Frame(
        val mag: FloatArray,
        val pha: FloatArray
    )

    private val frames = mutableListOf<Frame>()
    private val lock = ReentrantLock()

    /**
     * Number of frames currently buffered.
     */
    val bufferedFrames: Int
        get() = lock.withLock { frames.size }

    /**
     * Push a single frame.
     */
    fun push(mag: FloatArray, pha: FloatArray) = lock.withLock {
        frames.add(Frame(mag.clone(), pha.clone()))

        // Limit buffer size
        while (frames.size > maxFrames) {
            frames.removeAt(0)
        }
    }

    /**
     * Check if enough frames are available.
     */
    fun hasEnough(count: Int): Boolean = lock.withLock { frames.size >= count }

    /**
     * Get concatenated features for the specified number of frames.
     *
     * @param count Number of frames to get
     * @return Pair of (mag, pha) arrays concatenated across frames
     */
    fun get(count: Int): Pair<FloatArray, FloatArray> = lock.withLock {
        val actualCount = minOf(count, frames.size)
        val totalSize = actualCount * freqBins

        val mag = FloatArray(totalSize)
        val pha = FloatArray(totalSize)

        for (i in 0 until actualCount) {
            frames[i].mag.copyInto(mag, i * freqBins)
            frames[i].pha.copyInto(pha, i * freqBins)
        }

        Pair(mag, pha)
    }

    /**
     * Remove the first N frames.
     */
    fun removeFirst(count: Int) = lock.withLock {
        repeat(minOf(count, frames.size)) {
            frames.removeAt(0)
        }
    }

    /**
     * Clear all frames.
     */
    fun clear() = lock.withLock {
        frames.clear()
    }
}
