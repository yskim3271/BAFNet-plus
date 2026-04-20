package com.lacosenet.benchmark.parity

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.FloatBuffer

/**
 * BAFNetPlus HTP session-init + 1-chunk probe (Stage 3 S3-θ).
 *
 * Loads `bafnetplus_qdq.onnx` on QNN HTP and runs a single chunk with
 * zero-initialized inputs + zero-state to verify that:
 *   1. The QNN EP can load the quantized graph (no unsupported ops).
 *   2. A single inference produces the expected 3 primary outputs plus
 *      166 next_state tensors without NaN/Inf.
 *
 * Stage 3 scope is strictly load + 1 chunk + shape sanity. Numerical QDQ
 * drift is deferred to Stage 5. If the device lacks QNN HTP (libQnnHtp.so
 * missing), the test is skipped gracefully (matches existing LaCoSENet
 * parity test conventions).
 *
 * Run with:
 *   ./gradlew :benchmark-app:connectedAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=\
 *        com.lacosenet.benchmark.parity.BafnetPlusHtpProbeTest
 */
@RunWith(AndroidJUnit4::class)
class BafnetPlusHtpProbeTest {

    companion object {
        private const val TAG = "BafnetPlusHtpProbe"
        private const val MODEL_QDQ = "bafnetplus_qdq.onnx"
        private const val CONFIG = "bafnetplus_streaming_config.json"
        private const val EXPECTED_NUM_STATES = 166
        private const val EXPECTED_PRIMARY_OUTPUTS = 3 // est_mag, est_com_real, est_com_imag
    }

    private lateinit var context: Context
    private lateinit var ortEnv: OrtEnvironment
    private var qnnAvailable = false

    @Before
    fun setUp() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        ortEnv = OrtEnvironment.getEnvironment()
        qnnAvailable = try {
            System.loadLibrary("QnnHtp")
            Log.i(TAG, "QNN HTP library loaded successfully")
            true
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "QNN HTP not available: ${e.message}")
            false
        }
    }

    @After
    fun tearDown() {
        ortEnv.close()
    }

    private fun copyAssetToFilesDir(assetName: String): File {
        val outFile = File(context.filesDir, assetName)
        context.assets.open(assetName).use { input ->
            outFile.outputStream().use { output -> input.copyTo(output) }
        }
        return outFile
    }

    private fun readStateShapes(): Map<String, IntArray> {
        val jsonStr = context.assets.open(CONFIG).bufferedReader().readText()
        val cfg = JSONObject(jsonStr)
        val stateInfo = cfg.getJSONObject("state_info")
        val layout = stateInfo.getJSONArray("state_layout")
        val map = LinkedHashMap<String, IntArray>()
        for (i in 0 until layout.length()) {
            val entry = layout.getJSONObject(i)
            val name = entry.getString("name")
            val shapeJson = entry.getJSONArray("shape")
            val shape = IntArray(shapeJson.length()) { shapeJson.getInt(it) }
            map[name] = shape
        }
        return map
    }

    private fun zerosTensor(shape: IntArray): OnnxTensor {
        val size = shape.fold(1) { acc, d -> acc * d }
        val buf = FloatBuffer.allocate(size)
        for (i in 0 until size) buf.put(0f)
        buf.rewind()
        return OnnxTensor.createTensor(
            ortEnv,
            buf,
            shape.map { it.toLong() }.toLongArray(),
        )
    }

    private fun hasNanInf(arr: FloatArray): Pair<Boolean, Boolean> {
        var nan = false
        var inf = false
        for (v in arr) {
            if (v.isNaN()) nan = true
            if (v.isInfinite()) inf = true
            if (nan && inf) return true to true
        }
        return nan to inf
    }

    @Test
    fun bafnetplusQdqLoadsAndRunsOneChunkOnHtp() {
        if (!qnnAvailable) {
            Log.w(TAG, "QNN HTP unavailable on this device — skipping probe")
            return
        }

        val modelFile = copyAssetToFilesDir(MODEL_QDQ)
        val modelPath = modelFile.absolutePath

        val providerOptions = mapOf(
            "backend_path" to "libQnnHtp.so",
            "htp_performance_mode" to "burst",
            "enable_htp_fp16_precision" to "0", // QDQ uses INT8, not FP16
            "enable_htp_shared_memory_allocator" to "1",
            "vtcm_mb" to "8",
        )

        val sessionOptions = OrtSession.SessionOptions().apply {
            try {
                addQnn(providerOptions)
            } catch (e: Exception) {
                fail("Failed to register QNN EP: ${e.message}")
            }
        }

        val session: OrtSession = try {
            ortEnv.createSession(modelPath, sessionOptions)
        } catch (e: Exception) {
            fail("Failed to create QNN HTP session: ${e.message}")
            return
        }

        Log.i(TAG, "QNN HTP session created for $MODEL_QDQ")
        Log.i(TAG, "  inputs=${session.inputInfo.size}, outputs=${session.outputInfo.size}")

        // Zero-input probe: bcs_mag/pha, acs_mag/pha all zeros + state zeros
        val stateShapes = readStateShapes()
        assertTrue(
            "Unexpected state count from config: ${stateShapes.size} (expected $EXPECTED_NUM_STATES)",
            stateShapes.size == EXPECTED_NUM_STATES,
        )

        val inputs = HashMap<String, OnnxTensor>()
        val audioShape = intArrayOf(1, 201, 11)
        inputs["bcs_mag"] = zerosTensor(audioShape)
        inputs["bcs_pha"] = zerosTensor(audioShape)
        inputs["acs_mag"] = zerosTensor(audioShape)
        inputs["acs_pha"] = zerosTensor(audioShape)
        for ((name, shape) in stateShapes) {
            inputs[name] = zerosTensor(shape)
        }

        val result = try {
            session.run(inputs)
        } catch (e: Exception) {
            fail("QNN HTP inference failed: ${e.message}")
            return
        }

        try {
            val numOutputs = result.size()
            val expected = EXPECTED_PRIMARY_OUTPUTS + EXPECTED_NUM_STATES
            assertTrue(
                "Output count $numOutputs != expected $expected",
                numOutputs == expected,
            )

            // Check primary outputs (est_mag, est_com_real, est_com_imag) for NaN/Inf
            for (outName in listOf("est_mag", "est_com_real", "est_com_imag")) {
                val t = result.get(outName).get() as OnnxTensor
                val flat = t.floatBuffer.array()
                val (nan, inf) = hasNanInf(flat)
                assertFalse("$outName contains NaN", nan)
                assertFalse("$outName contains Inf", inf)
                Log.i(TAG, "  $outName shape=${t.info.shape.toList()} (NaN/Inf free)")
            }

            Log.i(TAG, "HTP probe PASSED: $numOutputs outputs, 0 NaN/Inf in primary outputs")
        } finally {
            result.close()
            inputs.values.forEach { it.close() }
            session.close()
        }
    }
}
