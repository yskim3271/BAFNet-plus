package com.lacosenet.benchmark.parity

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.lacosenet.streaming.audio.StftProcessor
import com.lacosenet.streaming.core.StftConfig
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Parity checks between Kotlin [StftProcessor.istftStreaming] and the Python
 * `manual_istft_ola` reference (src/stft.py). The fixture generator dumps
 * est_mag_crop / est_pha_crop as iSTFT inputs and istft_output / ola_buffer_out
 * / ola_norm_out as the expected outputs per chunk.
 *
 * Tolerances are looser than StatefulInference tests because both sides use
 * naive O(N^2) inverse DFT with different accumulation order (Kotlin: explicit
 * loop, Python: numpy.fft.irfft). Post-A7 (FFT transition) these should drop
 * to float32 noise (~1e-7).
 */
@RunWith(AndroidJUnit4::class)
class IstftParityTest {

    companion object {
        private const val TAG = "IstftParityTest"
        private const val RMS_TOLERANCE = 1e-4f
        private const val MAX_TOLERANCE = 2e-3f
    }

    private lateinit var loader: FixtureLoader

    @Before
    fun setUp() {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        loader = FixtureLoader(testContext)
    }

    private fun makeStftConfig(): StftConfig {
        val m = loader.manifest
        return StftConfig(
            nFft = m.nFft,
            hopSize = m.hopSize,
            winLength = m.winSize,
            compressFactor = m.compressFactor,
            sampleRate = 16000,
            center = true,
        )
    }

    /**
     * Single-chunk iSTFT parity at chunk 0 (zero OLA start). Kotlin processor
     * is reset, then fed the fixture's est_mag_crop + est_pha_crop; output
     * is compared to istft_output.bin.
     */
    @Test
    fun istftStreaming_chunk000_zeroOlaStart() {
        val m = loader.manifest
        val chunk = m.chunks[0]

        val mag = loader.readTensor(chunk.files.getValue("est_mag_crop"))
        val pha = loader.readTensor(chunk.files.getValue("est_pha_crop"))
        val expectedOut = loader.readTensor(chunk.files.getValue("istft_output"))

        val proc = StftProcessor(makeStftConfig())
        proc.reset()

        val numFrames = m.chunkSizeFrames
        val actualOut = proc.istftStreaming(mag, pha, numFrames)

        assertTrue(
            "Size mismatch: expected ${expectedOut.size}, got ${actualOut.size}",
            actualOut.size == expectedOut.size,
        )

        val rms = rmsDiff(expectedOut, actualOut)
        val max = maxAbsDiff(expectedOut, actualOut)
        Log.i(TAG, "chunk0 istft: RMS=$rms, MaxAbs=$max")
        assertTrue("chunk0 RMS=$rms > $RMS_TOLERANCE", rms < RMS_TOLERANCE)
        assertTrue("chunk0 MaxAbs=$max > $MAX_TOLERANCE", max < MAX_TOLERANCE)
    }

    /**
     * Cross-chunk iSTFT parity: after chunk 0 the processor must keep the
     * 300-sample OLA tail internally and apply it to chunk 1, matching the
     * Python reference's (ola_buffer, ola_norm) carry-over.
     */
    @Test
    fun istftStreaming_chunk001_afterChunk000() {
        val m = loader.manifest
        require(m.numChunks >= 2) { "Need >=2 chunks" }

        val mag0 = loader.readTensor(m.chunks[0].files.getValue("est_mag_crop"))
        val pha0 = loader.readTensor(m.chunks[0].files.getValue("est_pha_crop"))
        val mag1 = loader.readTensor(m.chunks[1].files.getValue("est_mag_crop"))
        val pha1 = loader.readTensor(m.chunks[1].files.getValue("est_pha_crop"))
        val expected1 = loader.readTensor(m.chunks[1].files.getValue("istft_output"))

        val proc = StftProcessor(makeStftConfig())
        proc.reset()

        val numFrames = m.chunkSizeFrames
        proc.istftStreaming(mag0, pha0, numFrames)
        val out1 = proc.istftStreaming(mag1, pha1, numFrames)

        val rms = rmsDiff(expected1, out1)
        val max = maxAbsDiff(expected1, out1)
        Log.i(TAG, "chunk1 istft: RMS=$rms, MaxAbs=$max")
        assertTrue("chunk1 RMS=$rms > $RMS_TOLERANCE", rms < RMS_TOLERANCE)
        assertTrue("chunk1 MaxAbs=$max > $MAX_TOLERANCE", max < MAX_TOLERANCE)
    }
}
