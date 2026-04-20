/**
 * StatefulInference.kt
 *
 * Stateful inference runner for LaCoSENet / BAFNetPlus ONNX streaming models.
 * Manages explicit state I/O matching StatefulExportableNNCore from Python.
 *
 * Optimizations:
 * - Tensor pooling: Pre-allocated Direct ByteBuffers for zero-copy JNI transfer
 * - Double buffering: Swap state buffers without copying
 * - Reusable tensors: Minimize allocation overhead per inference
 *
 * Two run() overloads:
 * - `run(mag, pha)` — LaCoSENet-specific convenience, returns [InferenceResult]
 *   with atan2-derived phase (when `phase_output_mode == "complex"`).
 * - `run(audioInputs: Map<String, FloatArray>)` — generic multi-input API used by
 *   BAFNetPlus (bcs_mag, bcs_pha, acs_mag, acs_pha). Returns a Map of primary
 *   outputs keyed by graph output name.
 */
package com.lacosenet.streaming.session

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.util.Log
import com.lacosenet.streaming.backend.ExecutionBackend
import com.lacosenet.streaming.core.InferenceResult
import com.lacosenet.streaming.core.StreamingConfig
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.atan2

/**
 * Stateful inference runner for streaming enhancement.
 *
 * Audio inputs (non-`state_*`) and primary outputs (non-`next_state_*`) are
 * discovered from the session metadata — both LaCoSENet ("mag"/"pha") and
 * BAFNetPlus ("bcs_mag"/"bcs_pha"/"acs_mag"/"acs_pha") are supported by the
 * same class.
 *
 * @param env ONNX Runtime environment
 * @param backend Execution backend (QNN, NNAPI, or CPU)
 * @param config Streaming configuration
 */
class StatefulInference(
    private val env: OrtEnvironment,
    private val backend: ExecutionBackend,
    private val config: StreamingConfig
) {
    companion object {
        private const val TAG = "StatefulInference"
        private const val FLOAT_BYTES = 4
    }

    // State management
    private val stateNames = mutableListOf<String>()
    private val nextStateNames = mutableListOf<String>()
    private val stateShapes = mutableMapOf<String, LongArray>()
    private val stateSizes = mutableMapOf<String, Int>()
    private var isInitialized = false

    // Double buffering for states (swap between A and B)
    private var stateBuffersA = mutableMapOf<String, ByteBuffer>()
    private var stateBuffersB = mutableMapOf<String, ByteBuffer>()
    private var stateTensorsA = mutableMapOf<String, OnnxTensor>()
    private var stateTensorsB = mutableMapOf<String, OnnxTensor>()
    private var useBufferA = true  // Current active buffer set

    // Set when ORT throws during run(); forces resetStates() before the next run (H2).
    private var isStateInvalid = false

    // Generic audio input registry — keyed by input name (e.g. "mag"/"pha" for
    // LaCoSENet, "bcs_mag"/"bcs_pha"/"acs_mag"/"acs_pha" for BAFNetPlus).
    private val audioBuffers = mutableMapOf<String, ByteBuffer>()
    private val audioTensors = mutableMapOf<String, OnnxTensor>()
    private val audioShapes = mutableMapOf<String, LongArray>()
    private val audioSizes = mutableMapOf<String, Int>()

    // Generic primary-output registry — keyed by output name, excludes `next_state_*`.
    // Each entry is a preallocated FloatArray reused across inference calls.
    private val primaryOutputArrays = mutableMapOf<String, FloatArray>()

    // LaCoSENet `phase_output_mode = "complex"` post-processing cache — atan2 from
    // phase_real / phase_imag into estPhaCache, returned via [InferenceResult.estPhase].
    private var estPhaCache: FloatArray? = null

    // Inference type
    private val inferType = config.modelInfo.inferType
    private val phaseOutputMode = config.modelInfo.phaseOutputMode

    // Last run() wall-clock inference time in ms (populated by the generic path).
    private var lastInferenceTimeMs: Float = 0f

    /** Input names discovered at initialize() time (audio, non-state). */
    val audioInputNames: List<String>
        get() = audioBuffers.keys.toList()

    /** Primary output names discovered at initialize() time (non-next_state). */
    val primaryOutputNames: List<String>
        get() = primaryOutputArrays.keys.toList()

    /** Number of state tensors managed by this instance. */
    val numStates: Int
        get() = stateNames.size

    /**
     * Initialize state tensors from session metadata.
     * Pre-allocates Direct ByteBuffers for tensor pooling.
     */
    fun initialize() {
        val session = backend.session
            ?: throw IllegalStateException("Backend not initialized")

        // Discover state + audio inputs from session metadata.
        for (inputInfo in session.inputInfo) {
            val name = inputInfo.key
            val tensorInfo = inputInfo.value.info as TensorInfo
            if (name.startsWith("state_")) {
                stateNames.add(name)
                nextStateNames.add("next_$name")
                stateShapes[name] = tensorInfo.shape
                stateSizes[name] = tensorInfo.shape.fold(1L) { acc, dim -> acc * dim }.toInt()
            } else {
                audioShapes[name] = tensorInfo.shape
                audioSizes[name] = tensorInfo.shape.fold(1L) { acc, dim -> acc * dim }.toInt()
            }
        }

        // Sort for deterministic order
        stateNames.sort()
        nextStateNames.sortBy { it.removePrefix("next_") }

        // Assert sorted ONNX state names match streaming_config.json state_info.state_names.
        // Parity (C4) protects against silent drift when either side is re-exported.
        val expectedStateNames = config.stateInfo?.stateNames.orEmpty()
        if (expectedStateNames.isNotEmpty()) {
            val expectedSorted = expectedStateNames.sorted()
            check(stateNames == expectedSorted) {
                val onnxOnly = stateNames - expectedSorted.toSet()
                val configOnly = expectedSorted - stateNames.toSet()
                "State registry mismatch between ONNX and streaming_config.json. " +
                    "ONNX has ${stateNames.size} state_* inputs; config declares " +
                    "${expectedSorted.size}. In ONNX but not config: $onnxOnly. " +
                    "In config but not ONNX: $configOnly."
            }
        } else {
            Log.w(TAG, "streaming_config.json has no state_info.state_names; skipping registry assertion")
        }

        Log.d(TAG, "Found ${stateNames.size} state tensors, ${audioShapes.size} audio inputs")
        for ((name, shape) in audioShapes) {
            Log.d(TAG, "  audio input '$name': shape=${shape.contentToString()}")
        }

        // Pre-allocate audio input buffers + tensors (one per non-state input).
        for ((name, shape) in audioShapes) {
            val size = audioSizes.getValue(name)
            val buffer = allocateDirectBuffer(size)
            audioBuffers[name] = buffer
            audioTensors[name] = OnnxTensor.createTensor(env, buffer.asFloatBuffer(), shape)
        }

        // Pre-allocate double buffers for states
        allocateStateBuffers()

        // Pre-allocate output arrays for primary outputs (non-next_state).
        for (outputInfo in session.outputInfo) {
            val name = outputInfo.key
            if (name.startsWith("next_state_")) continue
            val tensorInfo = outputInfo.value.info as TensorInfo
            val size = tensorInfo.shape.fold(1L) { acc, dim -> acc * dim }.toInt()
            primaryOutputArrays[name] = FloatArray(size)
        }

        // Initialize states to zeros
        resetStates()
        isInitialized = true

        val totalAudioBytes = audioSizes.values.sum() * FLOAT_BYTES
        val totalStateBytes = stateSizes.values.sum() * FLOAT_BYTES * 2
        Log.i(TAG, "Tensor pooling initialized:")
        Log.i(TAG, "  - Audio buffers: ${audioBuffers.size} inputs, ${totalAudioBytes} bytes")
        Log.i(TAG, "  - State buffers: ${stateNames.size} x 2 (double buffering)")
        Log.i(TAG, "  - Total state memory: ${totalStateBytes / 1024} KB")
        Log.i(TAG, "  - Primary outputs: ${primaryOutputArrays.size} names")
    }

    /**
     * Allocate a Direct ByteBuffer with native byte order.
     */
    private fun allocateDirectBuffer(floatCount: Int): ByteBuffer {
        return ByteBuffer.allocateDirect(floatCount * FLOAT_BYTES)
            .order(ByteOrder.nativeOrder())
    }

    /**
     * Pre-allocate double buffers for all state tensors.
     */
    private fun allocateStateBuffers() {
        for (name in stateNames) {
            val size = stateSizes[name]!!
            val shape = stateShapes[name]!!

            // Allocate buffer set A
            val bufferA = allocateDirectBuffer(size)
            stateBuffersA[name] = bufferA
            stateTensorsA[name] = OnnxTensor.createTensor(env, bufferA.asFloatBuffer(), shape)

            // Allocate buffer set B
            val bufferB = allocateDirectBuffer(size)
            stateBuffersB[name] = bufferB
            stateTensorsB[name] = OnnxTensor.createTensor(env, bufferB.asFloatBuffer(), shape)
        }
    }

    /**
     * Reset all states to zeros.
     * With tensor pooling, this just clears the buffer contents without reallocation.
     */
    fun resetStates() {
        // Zero out the active buffer set
        val activeBuffers = if (useBufferA) stateBuffersA else stateBuffersB

        for (name in stateNames) {
            val buffer = activeBuffers[name]!!
            buffer.clear()
            // Fill with zeros
            val floatBuffer = buffer.asFloatBuffer()
            val size = stateSizes[name]!!
            for (i in 0 until size) {
                floatBuffer.put(0f)
            }
            buffer.rewind()
        }

        // Reset to use buffer A
        useBufferA = true
        isStateInvalid = false
    }

    /**
     * Run inference with arbitrary named audio inputs (generic path).
     *
     * Caller supplies a [FloatArray] per non-state input; the method copies each
     * array into the preallocated Direct ByteBuffer of the matching tensor, then
     * runs the backend using the active state tensor set and swaps the state
     * double-buffer from the `next_state_*` outputs.
     *
     * Returns the preallocated primary-output arrays keyed by output name. The
     * returned arrays are reused on the next call — **do not retain references**
     * across invocations.
     *
     * @throws IllegalArgumentException if `audioInputs` keys don't match the
     *         session's non-state inputs, or if any array size mismatches.
     */
    fun run(audioInputs: Map<String, FloatArray>): Map<String, FloatArray> {
        check(isInitialized) { "StatefulInference not initialized" }
        check(!isStateInvalid) {
            "StatefulInference state invalidated by previous error; call resetStates() to recover"
        }
        require(audioInputs.keys == audioBuffers.keys) {
            "audioInputs keys mismatch: got ${audioInputs.keys}, expected ${audioBuffers.keys}"
        }
        for ((name, data) in audioInputs) {
            val expected = audioSizes.getValue(name)
            require(data.size == expected) {
                "audio input '$name' size mismatch: got ${data.size}, expected $expected " +
                    "(shape=${audioShapes.getValue(name).contentToString()})"
            }
        }

        val startTime = System.nanoTime()

        // Copy input data to preallocated buffers (zero-copy to JNI).
        for ((name, data) in audioInputs) {
            fillInputBuffer(audioBuffers.getValue(name), data)
        }

        // Active state tensor set (swapped each call).
        val activeStateTensors = if (useBufferA) stateTensorsA else stateTensorsB

        // Build input map: audio + state tensors.
        val inputs = HashMap<String, OnnxTensor>(audioTensors.size + stateNames.size)
        inputs.putAll(audioTensors)
        for (name in stateNames) {
            inputs[name] = activeStateTensors.getValue(name)
        }

        // Run inference. H2: on ORT exception, state buffers / preallocated outputs
        // may be partially written; mark state invalid so caller must resetStates().
        try {
            backend.run(inputs).use { result ->
                val outputs = result.associate { it.key to (it.value as OnnxTensor) }

                // Extract primary outputs into preallocated arrays.
                for ((name, array) in primaryOutputArrays) {
                    val tensor = outputs[name]
                        ?: throw IllegalStateException("Missing primary output: $name")
                    extractIntoArray(tensor, array)
                }

                // Swap state double-buffer from next_state_* outputs.
                updateStatesDoubleBuffer(outputs)

                lastInferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000f

                // Return a snapshot map — values still reference preallocated arrays.
                return HashMap(primaryOutputArrays)
            }
        } catch (e: Exception) {
            isStateInvalid = true
            throw e
        }
    }

    /**
     * Run inference on a single frame (LaCoSENet backward-compat path).
     * Delegates to the generic `run(audioInputs)` path and post-processes the
     * primary outputs into the LaCoSENet-specific [InferenceResult] shape
     * (applying atan2 when `phase_output_mode == "complex"`).
     *
     * @param mag Magnitude spectrogram [1, F, T]
     * @param pha Phase spectrogram [1, F, T]
     * @return Inference result with enhanced mask and phase
     */
    fun run(mag: FloatArray, pha: FloatArray): InferenceResult {
        val outputs = run(mapOf("mag" to mag, "pha" to pha))

        val estMask = outputs["est_mask"]
            ?: throw IllegalStateException(
                "LaCoSENet run(mag, pha) requires 'est_mask' output; available: ${outputs.keys}"
            )

        val estPha: FloatArray = if (phaseOutputMode == "complex") {
            val phaseReal = outputs["phase_real"]
                ?: throw IllegalStateException("complex mode requires 'phase_real' output")
            val phaseImag = outputs["phase_imag"]
                ?: throw IllegalStateException("complex mode requires 'phase_imag' output")
            val target = estPhaCache
                ?: FloatArray(phaseReal.size).also { estPhaCache = it }
            computeAtan2InPlace(phaseImag, phaseReal, target)
        } else {
            outputs["est_pha"]
                ?: throw IllegalStateException("non-complex mode requires 'est_pha' output")
        }

        return InferenceResult(
            estMask = estMask,
            estPhase = estPha,
            inferenceTimeMs = lastInferenceTimeMs
        )
    }

    /**
     * Fill pre-allocated buffer with input data.
     * Handles padding/trimming to expected size.
     */
    private fun fillInputBuffer(buffer: ByteBuffer, data: FloatArray) {
        buffer.clear()
        val floatBuffer = buffer.asFloatBuffer()
        val expectedSize = floatBuffer.capacity()

        when {
            data.size == expectedSize -> {
                floatBuffer.put(data)
            }
            data.size < expectedSize -> {
                // Pad with zeros
                floatBuffer.put(data)
                for (i in data.size until expectedSize) {
                    floatBuffer.put(0f)
                }
            }
            else -> {
                // Trim
                floatBuffer.put(data, 0, expectedSize)
            }
        }
        buffer.rewind()
    }

    /**
     * Apply mask to magnitude.
     */
    fun applyMask(mag: FloatArray, mask: FloatArray): FloatArray {
        return if (inferType == "masking") {
            FloatArray(mag.size) { i -> mag[i] * mask[i] }
        } else {
            mask
        }
    }

    /**
     * Extract float data from ONNX tensor into a pre-allocated array (zero-alloc).
     */
    private fun extractIntoArray(tensor: OnnxTensor, target: FloatArray): FloatArray {
        val buffer = tensor.floatBuffer
        buffer.get(target, 0, target.size)
        return target
    }

    /**
     * Compute atan2(imag, real) in-place into target array (zero-alloc).
     */
    private fun computeAtan2InPlace(imag: FloatArray, real: FloatArray, target: FloatArray): FloatArray {
        for (i in target.indices) {
            target[i] = atan2(imag[i] + 1e-8f, real[i] + 1e-8f)
        }
        return target
    }

    /**
     * Update state tensors using double buffering.
     * Copies next_state outputs to the inactive buffer set, then swaps.
     * This avoids tensor allocation per inference.
     */
    private fun updateStatesDoubleBuffer(outputs: Map<String, OnnxTensor>) {
        // Get the inactive buffer set (will become active after swap)
        val inactiveBuffers = if (useBufferA) stateBuffersB else stateBuffersA

        for (i in stateNames.indices) {
            val stateName = stateNames[i]
            val nextStateName = nextStateNames[i]

            val nextState = outputs[nextStateName]
                ?: throw IllegalStateException("Missing output: $nextStateName")

            // Copy next_state data to inactive buffer
            val buffer = inactiveBuffers[stateName]!!
            buffer.clear()
            val floatBuffer = buffer.asFloatBuffer()

            // Extract data from output tensor and copy to buffer
            val outputBuffer = nextState.floatBuffer
            floatBuffer.put(outputBuffer)
            buffer.rewind()
        }

        // Swap active buffer set
        useBufferA = !useBufferA
    }

    /**
     * Release all resources.
     * Closes all pre-allocated tensors.
     */
    fun release() {
        // Close audio tensors + clear maps
        audioTensors.values.forEach { it.close() }
        audioTensors.clear()
        audioBuffers.clear()
        audioShapes.clear()
        audioSizes.clear()

        // Close state tensors (both buffer sets)
        stateTensorsA.values.forEach { it.close() }
        stateTensorsB.values.forEach { it.close() }
        stateTensorsA.clear()
        stateTensorsB.clear()

        // Clear state buffers
        stateBuffersA.clear()
        stateBuffersB.clear()

        // Clear preallocated output arrays
        primaryOutputArrays.clear()
        estPhaCache = null

        // Clear metadata
        stateNames.clear()
        nextStateNames.clear()
        stateShapes.clear()
        stateSizes.clear()

        isInitialized = false
    }
}
