package com.lacosenet.benchmark.parity

import android.content.Context
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.lacosenet.streaming.audio.StftProcessor
import com.lacosenet.streaming.core.StftConfig
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Parity checks between Kotlin [StftProcessor] and the Python reference dumped
 * by scripts/make_streaming_golden.py.
 *
 * These tests are expected to FAIL on current HEAD — that is the point of
 * Stage 2: the failures are the regression signal for A1-A7 fixes.
 *
 * Baseline tolerances:
 *   - RMS diff  < 1e-4
 *   - max diff  < 2e-3 (looser to allow numerical drift from naive DFT; fix A7
 *                       would push this down to ~1e-5)
 *
 * Run with:
 *   ./gradlew :benchmark-app:connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.parity.StftParityTest
 */
@RunWith(AndroidJUnit4::class)
class StftParityTest {

    companion object {
        private const val TAG = "StftParityTest"
        private const val RMS_TOLERANCE = 1e-4f
        private const val MAX_TOLERANCE = 2e-3f
    }

    private lateinit var testContext: Context
    private lateinit var loader: FixtureLoader

    @Before
    fun setUp() {
        testContext = InstrumentationRegistry.getInstrumentation().context
        loader = FixtureLoader(testContext)
    }

    private fun makeStftConfig(): StftConfig {
        val m = loader.manifest
        return StftConfig(
            nFft = m.nFft,
            hopSize = m.hopSize,
            winLength = m.winSize,
            sampleRate = 16000,
            center = true,
            compressFactor = m.compressFactor,
        )
    }

    /**
     * Raw DFT math parity: bypasses Kotlin's internal context/reflect-padding
     * by feeding the already-prepared `stft_input` (1400 samples) with
     * `center=false`. Isolates numerical differences in the core DFT math
     * itself — periodic Hann (A2), atan2 epsilon (A5), and the naive O(N^2)
     * DFT's cumulative float error (A7).
     */
    @Test
    fun stftRawMath_chunk000() {
        val m = loader.manifest
        require(m.numChunks > 0) { "No fixture chunks" }
        val chunk = m.chunks[0]

        val stftInput = loader.readTensor(chunk.files.getValue("stft_input"))
        val expectedMag = loader.readTensor(chunk.files.getValue("stft_mag"))
        val expectedPha = loader.readTensor(chunk.files.getValue("stft_pha"))

        val proc = StftProcessor(makeStftConfig())
        val (mag, pha) = proc.stft(stftInput, center = false)

        reportAndAssert("mag@chunk000 raw", expectedMag, mag)
        reportAndAssert("pha@chunk000 raw", expectedPha, pha)
    }

    /**
     * Streaming-pipeline parity: feeds the 1200-sample `input_samples` (chunk
     * 0's Python-semantics chunk) to a fresh Kotlin [StftProcessor]. For chunk
     * 0 the Python reference prepended 200 zeros (zero-initialized
     * stft_context) and used `center=false`; the Kotlin processor's default is
     * `center=true` with reflect padding, so this test exposes A1/A2/A4/A5
     * together. Expected to fail on HEAD.
     */
    @Test
    fun stftStreaming_chunk000() {
        val m = loader.manifest
        val chunk = m.chunks[0]

        val inputSamples = loader.readTensor(chunk.files.getValue("input_samples"))
        val expectedMag = loader.readTensor(chunk.files.getValue("stft_mag"))
        val expectedPha = loader.readTensor(chunk.files.getValue("stft_pha"))

        val proc = StftProcessor(makeStftConfig())
        val (mag, pha) = proc.stft(inputSamples, center = true)

        // Shape sanity: Kotlin may not yield 11 frames with reflect padding on
        // 1200 samples, which itself is an A-axis bug. Log the size before
        // comparing so failures are easy to interpret.
        Log.i(TAG, "Expected mag size=${expectedMag.size}, actual=${mag.size}")
        if (mag.size != expectedMag.size) {
            fail(
                "Size mismatch: expected ${expectedMag.size} " +
                    "(${m.freqBins} bins × ${m.totalFramesNeeded} frames), " +
                    "got ${mag.size}. Likely A1/A4: samples_per_chunk or " +
                    "reflect-pad yields a different frame count."
            )
        }
        reportAndAssert("mag@chunk000 streaming", expectedMag, mag)
        reportAndAssert("pha@chunk000 streaming", expectedPha, pha)
    }

    /**
     * Cross-chunk streaming parity: after processing chunk 0 via the
     * processor, chunk 1 must use the carry-over context established
     * internally. In the Python reference chunk 1's
     * `stft_context_in` is the last 200 samples of chunk 0's audio; this test
     * asserts Kotlin reaches the same mag/pha for chunk 1 when fed `input_samples`
     * sequentially. Exposes A3 (missing OLA carry-over is not here but the STFT
     * context-management path is exercised) and A1/A2/A4/A5 from the previous test.
     */
    @Test
    fun stftStreaming_chunk001_afterChunk000() {
        val m = loader.manifest
        require(m.numChunks >= 2) { "Need at least 2 chunks" }

        val chunk0Input = loader.readTensor(m.chunks[0].files.getValue("input_samples"))
        val chunk1Input = loader.readTensor(m.chunks[1].files.getValue("input_samples"))
        val expectedMag = loader.readTensor(m.chunks[1].files.getValue("stft_mag"))
        val expectedPha = loader.readTensor(m.chunks[1].files.getValue("stft_pha"))

        val proc = StftProcessor(makeStftConfig())
        // Warm up the processor — discard chunk 0's STFT output, we only care
        // about what internal context gets saved for chunk 1. advanceSamples
        // must match Python streaming's advance (= output_samples_per_chunk).
        val advance = m.outputSamplesPerChunk
        proc.stft(chunk0Input, center = true, advanceSamples = advance)
        val (mag, pha) = proc.stft(chunk1Input, center = true, advanceSamples = advance)

        if (mag.size != expectedMag.size) {
            fail("Size mismatch at chunk001: expected ${expectedMag.size}, got ${mag.size}")
        }
        reportAndAssert("mag@chunk001 streaming", expectedMag, mag)
        reportAndAssert("pha@chunk001 streaming", expectedPha, pha)
    }

    private fun reportAndAssert(label: String, expected: FloatArray, actual: FloatArray) {
        val rms = rmsDiff(expected, actual)
        val max = maxAbsDiff(expected, actual)
        Log.i(TAG, "$label: RMS=$rms, MaxAbs=$max")
        assertTrue(
            "$label RMS diff=$rms exceeds tolerance $RMS_TOLERANCE",
            rms < RMS_TOLERANCE,
        )
        assertTrue(
            "$label MaxAbs diff=$max exceeds tolerance $MAX_TOLERANCE",
            max < MAX_TOLERANCE,
        )
    }
}
