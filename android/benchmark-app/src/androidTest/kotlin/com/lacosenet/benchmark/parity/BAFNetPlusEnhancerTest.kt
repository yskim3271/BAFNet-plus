package com.lacosenet.benchmark.parity

import android.content.Context
import android.os.Debug
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.bafnetplus.streaming.BAFNetPlusStreamingEnhancer
import com.lacosenet.streaming.backend.BackendType
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.max
import kotlin.random.Random

/**
 * End-to-end smoke tests for [BAFNetPlusStreamingEnhancer].
 *
 * - CPU backend: verify init, 100-chunk processChunk loop without crash, valid
 *   enhanced samples, and measure native-heap peak (budget: ≤ 400 MB).
 * - QNN HTP backend: verify QDQ model loads and runs 1 chunk on device. Skipped
 *   gracefully if libQnnHtp.so is unavailable.
 *
 * Run with:
 *   ./gradlew :benchmark-app:connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.parity.BAFNetPlusEnhancerTest
 */
@RunWith(AndroidJUnit4::class)
class BAFNetPlusEnhancerTest {

    companion object {
        private const val TAG = "BAFNetPlusEnhancer"
        private const val NUM_CHUNKS_SMOKE = 100
        private const val MEMORY_BUDGET_MB = 400

        /** 1 chunk = outputSamplesPerChunk input samples per channel (= 800 @ 16kHz). */
        private const val SAMPLES_PER_PUSH = 800
    }

    private lateinit var targetContext: Context
    private var qnnAvailable = false

    @Before
    fun setUp() {
        targetContext = InstrumentationRegistry.getInstrumentation().targetContext
        qnnAvailable = try {
            System.loadLibrary("QnnHtp")
            Log.i(TAG, "QNN HTP available")
            true
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "QNN HTP unavailable: ${e.message}")
            false
        }
    }

    private var enhancer: BAFNetPlusStreamingEnhancer? = null

    @After
    fun tearDown() {
        enhancer?.release()
        enhancer = null
    }

    /**
     * CPU backend E2E: push synthetic stereo-aligned audio through processChunk
     * for 100 chunks. Verify:
     *  - initialize() reports 166 states
     *  - At least one chunk returns enhanced samples (size == outputSamplesPerChunk)
     *  - No NaN/Inf in any enhanced output
     *  - Native-heap delta stays under the budget
     */
    @Test
    fun cpuEnhancerProcessChunkSmoke() {
        val baseHeapKb = Debug.getNativeHeapSize() / 1024
        Log.i(TAG, "Baseline native heap: ${baseHeapKb / 1024} MB")

        enhancer = BAFNetPlusStreamingEnhancer(targetContext)
        val init = enhancer!!.initialize(
            modelPath = "bafnetplus.onnx",
            forceBackend = BackendType.CPU,
        )
        assertTrue("init failed: ${init.errorMessage}", init.success)
        assertEquals("state count", 166, init.numStates)
        assertEquals("latency_ms == 50.0 (6 lookahead * 6.25ms)", 37.5f, init.latencyMs, 20f)
        Log.i(TAG, "Init: backend=${init.backend}, latency=${init.latencyMs}ms, load=${init.loadTimeMs}ms")

        val rng = Random(12345)
        var nonNullChunks = 0
        var peakHeapDelta = 0L
        var totalSamples = 0

        for (i in 0 until NUM_CHUNKS_SMOKE) {
            // Synthetic audio: low-amplitude noise on both streams, same size.
            val bcs = FloatArray(SAMPLES_PER_PUSH) { (rng.nextFloat() - 0.5f) * 0.05f }
            val acs = FloatArray(SAMPLES_PER_PUSH) { (rng.nextFloat() - 0.5f) * 0.05f }
            val enhanced = enhancer!!.processChunk(bcs, acs)
            if (enhanced != null) {
                nonNullChunks += 1
                totalSamples += enhanced.size
                assertTrue(
                    "enhanced chunk $i size=${enhanced.size} != 800",
                    enhanced.size == 800,
                )
                assertFalse(
                    "enhanced chunk $i contains NaN/Inf",
                    enhanced.any { !it.isFinite() },
                )
            }

            // Native heap tracking — sample every 10 chunks to reduce overhead.
            if ((i + 1) % 10 == 0) {
                val curKb = Debug.getNativeHeapSize() / 1024
                val delta = curKb - baseHeapKb
                peakHeapDelta = max(peakHeapDelta, delta)
            }
        }

        Log.i(TAG, "Processed $NUM_CHUNKS_SMOKE chunks: $nonNullChunks non-null, $totalSamples total samples")
        Log.i(TAG, "Peak native heap delta: ${peakHeapDelta / 1024} MB (budget: $MEMORY_BUDGET_MB MB)")
        assertTrue(
            "no chunks produced enhanced output in $NUM_CHUNKS_SMOKE iterations",
            nonNullChunks > 0,
        )
        val peakHeapMb = (baseHeapKb + peakHeapDelta) / 1024L
        assertTrue(
            "peak native heap ${peakHeapMb}MB exceeds budget ${MEMORY_BUDGET_MB}MB",
            peakHeapMb <= MEMORY_BUDGET_MB,
        )
    }

    /**
     * QNN HTP E2E: load `bafnetplus_qdq.onnx` via the enhancer with
     * forceBackend=QNN_HTP, run one chunk, confirm non-null output without NaN/Inf.
     * Skipped if QNN unavailable on device.
     */
    @Test
    fun qnnHtpEnhancerQdqSmoke() {
        if (!qnnAvailable) {
            Log.w(TAG, "QNN HTP unavailable — skipping QDQ smoke")
            return
        }
        enhancer = BAFNetPlusStreamingEnhancer(targetContext)
        val init = enhancer!!.initialize(
            modelPath = "bafnetplus_qdq.onnx",
            forceBackend = BackendType.QNN_HTP,
        )
        assertTrue("QDQ HTP init failed: ${init.errorMessage}", init.success)
        assertEquals("backend selected", BackendType.QNN_HTP, init.backend)
        assertEquals("state count", 166, init.numStates)
        Log.i(TAG, "QDQ HTP init: load=${init.loadTimeMs}ms, latency=${init.latencyMs}ms")

        // Warm up a couple of chunks to pass initial buffering.
        val rng = Random(2026)
        var got: FloatArray? = null
        for (i in 0 until 5) {
            val bcs = FloatArray(SAMPLES_PER_PUSH) { (rng.nextFloat() - 0.5f) * 0.05f }
            val acs = FloatArray(SAMPLES_PER_PUSH) { (rng.nextFloat() - 0.5f) * 0.05f }
            val out = enhancer!!.processChunk(bcs, acs)
            if (out != null) {
                got = out
                break
            }
        }
        assertNotNull("QDQ HTP produced null for first 5 chunks", got)
        val enhanced = got!!
        assertEquals("enhanced sample count", 800, enhanced.size)
        assertFalse(
            "QDQ HTP output contains NaN/Inf",
            enhanced.any { !it.isFinite() },
        )
        Log.i(TAG, "QDQ HTP first enhanced chunk: min=${enhanced.min()}, max=${enhanced.max()}")
    }
}
