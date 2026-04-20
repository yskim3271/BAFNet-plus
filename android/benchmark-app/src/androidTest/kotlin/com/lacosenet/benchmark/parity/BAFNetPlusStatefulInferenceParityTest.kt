package com.lacosenet.benchmark.parity

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
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
import java.nio.FloatBuffer

/**
 * Parity between Kotlin [StatefulInference] on the CPU backend and a raw ORT
 * CPU session for the BAFNetPlus dual-input graph. Both paths execute the same
 * ONNX model; numerical differences should be limited to JNI data-copy noise.
 *
 * Tolerances (same as LaCoSENet):
 *   RMS diff < 1e-5, max diff < 1e-4.
 *
 * This verifies that Kotlin correctly:
 *   1. Orders + flattens all 166 state tensors into the graph in the order the
 *      ONNX export produced.
 *   2. Copies the 4 audio inputs (bcs_mag, bcs_pha, acs_mag, acs_pha) to the
 *      right JNI tensors.
 *   3. Swaps the state double-buffer without losing any next_state slot.
 *   4. Resets state deterministically so chunk 0 produces the same output
 *      after a mid-stream reset.
 *
 * Run with:
 *   ./gradlew :benchmark-app:connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.parity.BAFNetPlusStatefulInferenceParityTest
 */
@RunWith(AndroidJUnit4::class)
class BAFNetPlusStatefulInferenceParityTest {

    companion object {
        private const val TAG = "BAFNetPlusParity"
        private const val MODEL_FILE = "bafnetplus.onnx"
        private const val CONFIG_FILE = "bafnetplus_streaming_config.json"
        private const val RMS_TOLERANCE = 1e-5f
        private const val MAX_TOLERANCE = 1e-4f
        private const val MAX_CHUNKS_TO_TEST = 3
        private val PRIMARY_OUTPUTS = listOf("est_mag", "est_com_real", "est_com_imag")
    }

    private lateinit var testContext: Context
    private lateinit var targetContext: Context
    private lateinit var fixtureLoader: BafnetPlusFixtureLoader
    private lateinit var ortEnv: OrtEnvironment
    private var config: StreamingConfig? = null
    private var modelFile: File? = null
    private var backend: CpuBackend? = null
    private var inference: StatefulInference? = null

    @Before
    fun setUp() {
        testContext = InstrumentationRegistry.getInstrumentation().context
        targetContext = InstrumentationRegistry.getInstrumentation().targetContext
        fixtureLoader = BafnetPlusFixtureLoader(testContext)
        ortEnv = OrtEnvironment.getEnvironment()

        config = StreamingConfig.fromAssets(targetContext, CONFIG_FILE)

        modelFile = prepareModelFile(MODEL_FILE)
        backend = CpuBackend()
        val initResult = backend!!.initialize(ortEnv, modelFile!!.absolutePath, config!!)
        assertTrue(
            "CPU backend failed to initialize: ${initResult.errorMessage}",
            initResult.success,
        )

        inference = StatefulInference(ortEnv, backend!!, config!!)
        inference!!.initialize()

        // Sanity: BAFNetPlus should expose 166 states + 4 audio inputs + 3 primaries.
        assertEquals("state count", 166, inference!!.numStates)
        assertEquals(
            "audio inputs",
            setOf("bcs_mag", "bcs_pha", "acs_mag", "acs_pha"),
            inference!!.audioInputNames.toSet(),
        )
        assertEquals(
            "primary outputs",
            PRIMARY_OUTPUTS.toSet(),
            inference!!.primaryOutputNames.toSet(),
        )
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
     * Read the bcs/acs mag/pha tensors for a given chunk.
     */
    private fun loadChunkInputs(chunkIdx: Int): Map<String, FloatArray> {
        val chunk = fixtureLoader.manifest.chunks[chunkIdx]
        return mapOf(
            "bcs_mag" to fixtureLoader.readTensor(chunk.files.getValue("bcs_mag")),
            "bcs_pha" to fixtureLoader.readTensor(chunk.files.getValue("bcs_pha")),
            "acs_mag" to fixtureLoader.readTensor(chunk.files.getValue("acs_mag")),
            "acs_pha" to fixtureLoader.readTensor(chunk.files.getValue("acs_pha")),
        )
    }

    /**
     * Run a single raw ORT session inference with the provided audio + state
     * inputs. Returns (primary outputs, next_state outputs) each as independent
     * FloatArray copies.
     */
    private fun runRaw(
        rawSession: OrtSession,
        audioInputs: Map<String, FloatArray>,
        audioShapes: Map<String, LongArray>,
        stateValues: Map<String, FloatArray>,
        stateShapes: Map<String, LongArray>,
    ): Pair<Map<String, FloatArray>, Map<String, FloatArray>> {
        val tensors = mutableListOf<OnnxTensor>()
        val inputMap = HashMap<String, OnnxTensor>()

        try {
            for ((name, data) in audioInputs) {
                val shape = audioShapes.getValue(name)
                val tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(data), shape)
                tensors.add(tensor)
                inputMap[name] = tensor
            }
            for ((name, data) in stateValues) {
                val shape = stateShapes.getValue(name)
                val tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(data), shape)
                tensors.add(tensor)
                inputMap[name] = tensor
            }

            rawSession.run(inputMap).use { result ->
                val primary = HashMap<String, FloatArray>()
                val nextStates = HashMap<String, FloatArray>()
                for (entry in result) {
                    val name = entry.key
                    val tensor = entry.value as OnnxTensor
                    val info = tensor.info as ai.onnxruntime.TensorInfo
                    val size = info.shape.fold(1L) { a, d -> a * d }.toInt()
                    val out = FloatArray(size)
                    tensor.floatBuffer.get(out, 0, size)
                    if (name.startsWith("next_state_")) {
                        nextStates[name.removePrefix("next_")] = out
                    } else {
                        primary[name] = out
                    }
                }
                return primary to nextStates
            }
        } finally {
            tensors.forEach { it.close() }
        }
    }

    /**
     * Returns the session's audio input shapes (bcs_mag/pha, acs_mag/pha).
     */
    private fun audioShapesFromSession(session: OrtSession): Map<String, LongArray> {
        val out = HashMap<String, LongArray>()
        for ((name, info) in session.inputInfo) {
            if (name.startsWith("state_")) continue
            val tensorInfo = info.info as ai.onnxruntime.TensorInfo
            out[name] = tensorInfo.shape
        }
        return out
    }

    /**
     * Returns the state shapes as declared by the session (state_* inputs).
     */
    private fun stateShapesFromSession(session: OrtSession): Map<String, LongArray> {
        val out = LinkedHashMap<String, LongArray>()
        for ((name, info) in session.inputInfo) {
            if (!name.startsWith("state_")) continue
            val tensorInfo = info.info as ai.onnxruntime.TensorInfo
            out[name] = tensorInfo.shape
        }
        return out
    }

    /**
     * Test 1 — sequential parity for the first 3 fixture chunks, comparing
     * Kotlin [StatefulInference] against a raw ORT CPU session. Both sides
     * start from zero state; each side feeds its own next_state outputs.
     */
    @Test
    fun sequentialParityFirstChunks() {
        val rawSession = ortEnv.createSession(
            modelFile!!.absolutePath,
            OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            },
        )
        try {
            val audioShapes = audioShapesFromSession(rawSession)
            val stateShapes = stateShapesFromSession(rawSession)
            var rawStateValues: Map<String, FloatArray> = stateShapes.mapValues { (_, shape) ->
                FloatArray(shape.fold(1L) { a, d -> a * d }.toInt())
            }

            val numChunks = minOf(MAX_CHUNKS_TO_TEST, fixtureLoader.manifest.chunks.size)
            require(numChunks > 0) { "no fixture chunks to test" }

            for (i in 0 until numChunks) {
                val audio = loadChunkInputs(i)

                // Raw ORT side (reference)
                val (rawPrimary, rawNextStates) = runRaw(
                    rawSession,
                    audioInputs = audio,
                    audioShapes = audioShapes,
                    stateValues = rawStateValues,
                    stateShapes = stateShapes,
                )
                rawStateValues = rawNextStates

                // Kotlin side (StatefulInference). Clone outputs because the returned
                // arrays reference preallocated buffers that get overwritten next call.
                val kotlinOutputs = inference!!.run(audio)
                val kPrimary = kotlinOutputs.mapValues { it.value.copyOf() }

                for (name in PRIMARY_OUTPUTS) {
                    val ref = rawPrimary[name]
                        ?: throw IllegalStateException("missing raw output $name at chunk $i")
                    val kot = kPrimary[name]
                        ?: throw IllegalStateException("missing kotlin output $name at chunk $i")
                    assertEquals("$name size @chunk$i", ref.size, kot.size)
                    val rms = rmsDiff(ref, kot)
                    val max = maxAbsDiff(ref, kot)
                    Log.i(TAG, "chunk=$i $name RMS=$rms max=$max")
                    assertTrue(
                        "$name@chunk$i RMS=$rms > $RMS_TOLERANCE",
                        rms < RMS_TOLERANCE,
                    )
                    assertTrue(
                        "$name@chunk$i max=$max > $MAX_TOLERANCE",
                        max < MAX_TOLERANCE,
                    )
                }
            }
        } finally {
            rawSession.close()
        }
    }

    /**
     * Test 2 — reset between chunks. After advancing state with chunk 1 and
     * calling resetStates(), re-running chunk 0 must produce bit-identical
     * output to the first chunk-0 run.
     */
    @Test
    fun resetRestoresFirstChunkOutput() {
        val audio0 = loadChunkInputs(0)
        val audio1 = loadChunkInputs(1)

        val first = inference!!.run(audio0).mapValues { it.value.copyOf() }

        // Advance state with chunk 1
        inference!!.run(audio1)

        inference!!.resetStates()

        val afterReset = inference!!.run(audio0).mapValues { it.value.copyOf() }

        for (name in PRIMARY_OUTPUTS) {
            val f = first[name]
                ?: throw IllegalStateException("missing first-run $name")
            val r = afterReset[name]
                ?: throw IllegalStateException("missing post-reset $name")
            val rms = rmsDiff(f, r)
            val max = maxAbsDiff(f, r)
            Log.i(TAG, "reset: $name RMS=$rms max=$max")
            // Bit-identical is ideal but we allow the same small tolerance
            // since fresh ONNX runs may have non-deterministic ORDER of fused ops.
            assertTrue(
                "$name reset RMS=$rms > $RMS_TOLERANCE (state leaked across reset)",
                rms < RMS_TOLERANCE,
            )
            assertTrue(
                "$name reset max=$max > $MAX_TOLERANCE",
                max < MAX_TOLERANCE,
            )
        }
    }
}
