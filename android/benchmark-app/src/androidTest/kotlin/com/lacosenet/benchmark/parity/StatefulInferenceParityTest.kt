package com.lacosenet.benchmark.parity

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.lacosenet.streaming.backend.CpuBackend
import com.lacosenet.streaming.core.StreamingConfig
import com.lacosenet.streaming.session.StatefulInference
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream

/**
 * Parity checks between Kotlin [StatefulInference] running on the CPU backend
 * and the Python/ONNX Runtime reference dumped by scripts/make_streaming_golden.py.
 *
 * Since both sides execute the *same* ONNX model file with the CPU provider,
 * numerical differences should be limited to JNI data-copy quirks. Tolerances
 * are tight:
 *   - RMS diff  < 1e-5
 *   - max diff  < 1e-4
 *
 * The streaming state is advanced by StatefulInference's double-buffer
 * internally; we feed chunks sequentially and compare each chunk's output.
 *
 * Note: the current [com.lacosenet.streaming.backend.BaseExecutionBackend.run]
 * leaks `OrtSession.Result` (B1 in REPORT.md). This test runs for the fixed
 * fixture chunk count so the leak is bounded, but a long-running variant is
 * deferred to a separate B1 leak test.
 *
 * Run with:
 *   ./gradlew :benchmark-app:connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.parity.StatefulInferenceParityTest
 */
@RunWith(AndroidJUnit4::class)
class StatefulInferenceParityTest {

    companion object {
        private const val TAG = "StatefulInferenceParity"
        private const val MODEL_FILE = "model.onnx"
        private const val CONFIG_FILE = "streaming_config.json"
        private const val RMS_TOLERANCE = 1e-5f
        private const val MAX_TOLERANCE = 1e-4f
        private const val MAX_CHUNKS_TO_TEST = 5
    }

    private lateinit var testContext: Context
    private lateinit var targetContext: Context
    private lateinit var loader: FixtureLoader
    private lateinit var ortEnv: OrtEnvironment
    private var backend: CpuBackend? = null
    private var inference: StatefulInference? = null
    private var config: StreamingConfig? = null

    @Before
    fun setUp() {
        testContext = InstrumentationRegistry.getInstrumentation().context
        targetContext = InstrumentationRegistry.getInstrumentation().targetContext
        loader = FixtureLoader(testContext)
        ortEnv = OrtEnvironment.getEnvironment()

        // Main APK assets (model, streaming_config.json) live in benchmark-app's
        // main source set, so we need targetContext to read them.
        config = StreamingConfig.fromAssets(targetContext, CONFIG_FILE)

        val modelFile = prepareModelFile(MODEL_FILE)
        backend = CpuBackend()
        val initResult = backend!!.initialize(ortEnv, modelFile.absolutePath, config!!)
        assertTrue(
            "CPU backend failed to initialize: ${initResult.errorMessage}",
            initResult.success,
        )

        inference = StatefulInference(ortEnv, backend!!, config!!)
        inference!!.initialize()
    }

    @After
    fun tearDown() {
        inference?.release()
        backend?.release()
    }

    private fun prepareModelFile(name: String): File {
        val target = File(targetContext.filesDir, name)
        if (!target.exists()) {
            targetContext.assets.open(name).use { input ->
                FileOutputStream(target).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return target
    }

    /**
     * Feed the first N fixture chunks sequentially through Kotlin's
     * StatefulInference, compare mask and phase output to the fixture's ORT
     * reference per chunk. State is advanced implicitly by Kotlin's double
     * buffer and Python's next_state feedback loop in the fixture generator.
     *
     * Expected behavior on current HEAD: no assertion failures (both sides run
     * the same CPU ONNX graph). If this test fails, it indicates a real
     * StatefulInference issue such as wrong state ordering or the double-buffer
     * swap skipping a state.
     */
    @Test
    fun sequentialStreamingParity_firstChunks() {
        val m = loader.manifest
        val nToCheck = minOf(MAX_CHUNKS_TO_TEST, m.numChunks)
        require(nToCheck > 0) { "No fixture chunks available" }

        for (i in 0 until nToCheck) {
            val chunk = m.chunks[i]
            val magIn = loader.readTensor(chunk.files.getValue("model_mag_in"))
            val phaIn = loader.readTensor(chunk.files.getValue("model_pha_in"))
            val expectedMask = loader.readTensor(chunk.files.getValue("est_mask"))
            val expectedPha = loader.readTensor(chunk.files.getValue("est_pha"))

            val result = inference!!.run(magIn, phaIn)

            assertEquals("estMask size@chunk$i", expectedMask.size, result.estMask.size)
            assertEquals("estPhase size@chunk$i", expectedPha.size, result.estPhase.size)

            val maskRms = rmsDiff(expectedMask, result.estMask)
            val maskMax = maxAbsDiff(expectedMask, result.estMask)
            val phaRms = rmsDiff(expectedPha, result.estPhase)
            val phaMax = maxAbsDiff(expectedPha, result.estPhase)
            Log.i(
                TAG,
                "chunk=$i mask RMS=$maskRms max=$maskMax | pha RMS=$phaRms max=$phaMax",
            )

            assertTrue(
                "mask@chunk$i RMS=$maskRms > $RMS_TOLERANCE",
                maskRms < RMS_TOLERANCE,
            )
            assertTrue(
                "mask@chunk$i max=$maskMax > $MAX_TOLERANCE",
                maskMax < MAX_TOLERANCE,
            )
            assertTrue(
                "pha@chunk$i RMS=$phaRms > $RMS_TOLERANCE",
                phaRms < RMS_TOLERANCE,
            )
            assertTrue(
                "pha@chunk$i max=$phaMax > $MAX_TOLERANCE",
                phaMax < MAX_TOLERANCE,
            )
        }
    }

    /**
     * Reset state between chunks — the first chunk's output must then match the
     * fixture's first chunk output even after multiple resets. This exercises
     * StatefulInference.resetStates() and confirms state buffers are zeroed.
     */
    @Test
    fun resetStatesRestoresChunk0Output() {
        val m = loader.manifest
        val chunk0 = m.chunks[0]
        val magIn = loader.readTensor(chunk0.files.getValue("model_mag_in"))
        val phaIn = loader.readTensor(chunk0.files.getValue("model_pha_in"))
        val expectedMask = loader.readTensor(chunk0.files.getValue("est_mask"))

        val r1 = inference!!.run(magIn, phaIn)
        // Advance state by a second run.
        if (m.numChunks >= 2) {
            val c1 = m.chunks[1]
            inference!!.run(
                loader.readTensor(c1.files.getValue("model_mag_in")),
                loader.readTensor(c1.files.getValue("model_pha_in")),
            )
        }
        inference!!.resetStates()
        val r2 = inference!!.run(magIn, phaIn)

        val rmsFirst = rmsDiff(expectedMask, r1.estMask)
        val rmsAfterReset = rmsDiff(expectedMask, r2.estMask)
        Log.i(TAG, "reset: first-run RMS=$rmsFirst, after-reset RMS=$rmsAfterReset")
        assertTrue(
            "First run RMS=$rmsFirst exceeds $RMS_TOLERANCE",
            rmsFirst < RMS_TOLERANCE,
        )
        assertTrue(
            "After-reset RMS=$rmsAfterReset exceeds $RMS_TOLERANCE (state leaked across reset)",
            rmsAfterReset < RMS_TOLERANCE,
        )
    }
}
