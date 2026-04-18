/**
 * StatefulInference.kt
 *
 * Stateful inference runner for LaCoSENet.
 * Manages explicit state I/O matching StatefulExportableNNCore from Python.
 *
 * Optimizations:
 * - Tensor pooling: Pre-allocated Direct ByteBuffers for zero-copy JNI transfer
 * - Double buffering: Swap state buffers without copying
 * - Reusable tensors: Minimize allocation overhead per inference
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
import java.nio.FloatBuffer
import kotlin.math.atan2

/**
 * Stateful inference runner for streaming enhancement.
 *
 * This class handles:
 * - State tensor initialization and management
 * - Running inference with proper I/O formatting
 * - Phase reconstruction from complex outputs (atan2 on host)
 * - Mask application for enhanced magnitude
 *
 * The ONNX model has explicit state I/O:
 * - Inputs: mag, pha, state_enc_conv0, state_enc_conv1, ..., state_ts_time_0, ...
 * - Outputs: est_mask, phase_real, phase_imag, next_state_enc_conv0, ...
 *
 * Performance optimizations:
 * - Tensor pooling with Direct ByteBuffers for zero-copy JNI data transfer
 * - Double buffering for state tensors (swap without copying)
 * - Pre-allocated input tensors (mag, pha) reused across inferences
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

    // Pre-allocated input tensors (mag, pha)
    private var magBuffer: ByteBuffer? = null
    private var phaBuffer: ByteBuffer? = null
    private var magTensor: OnnxTensor? = null
    private var phaTensor: OnnxTensor? = null

    // Input/output shapes
    private var magShape: LongArray = longArrayOf(1, config.stftConfig.freqBins.toLong(), config.streamingConfig.exportTimeFrames.toLong())
    private var phaShape: LongArray = magShape.clone()
    private var magSize: Int = 0
    private var phaSize: Int = 0

    // Inference type
    private val inferType = config.modelInfo.inferType
    private val phaseOutputMode = config.modelInfo.phaseOutputMode

    // Pre-allocated output arrays (avoid per-inference allocation)
    private var estMaskArray: FloatArray? = null
    private var estPhaArray: FloatArray? = null
    private var phaseRealArray: FloatArray? = null
    private var phaseImagArray: FloatArray? = null

    /**
     * Initialize state tensors from session metadata.
     * Pre-allocates Direct ByteBuffers for tensor pooling.
     */
    fun initialize() {
        val session = backend.session
            ?: throw IllegalStateException("Backend not initialized")

        // Extract state names from inputs
        for (inputInfo in session.inputInfo) {
            val name = inputInfo.key
            if (name.startsWith("state_")) {
                stateNames.add(name)
                nextStateNames.add("next_$name")

                val tensorInfo = inputInfo.value.info as TensorInfo
                stateShapes[name] = tensorInfo.shape
                stateSizes[name] = tensorInfo.shape.fold(1L) { acc, dim -> acc * dim }.toInt()
            } else if (name == "mag") {
                val tensorInfo = inputInfo.value.info as TensorInfo
                magShape = tensorInfo.shape
                phaShape = tensorInfo.shape.clone()
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

        Log.d(TAG, "Found ${stateNames.size} state tensors")
        Log.d(TAG, "Mag shape: ${magShape.contentToString()}")

        // Calculate input sizes
        magSize = magShape.fold(1L) { acc, dim -> acc * dim }.toInt()
        phaSize = phaShape.fold(1L) { acc, dim -> acc * dim }.toInt()

        // Pre-allocate input buffers (Direct ByteBuffer for zero-copy)
        magBuffer = allocateDirectBuffer(magSize)
        phaBuffer = allocateDirectBuffer(phaSize)

        // Create reusable input tensors
        magTensor = OnnxTensor.createTensor(env, magBuffer!!.asFloatBuffer(), magShape)
        phaTensor = OnnxTensor.createTensor(env, phaBuffer!!.asFloatBuffer(), phaShape)

        // Pre-allocate double buffers for states
        allocateStateBuffers()

        // Pre-allocate output arrays from session output info
        for (outputInfo in session.outputInfo) {
            val name = outputInfo.key
            val tensorInfo = outputInfo.value.info as TensorInfo
            val size = tensorInfo.shape.fold(1L) { acc, dim -> acc * dim }.toInt()
            when (name) {
                "est_mask" -> estMaskArray = FloatArray(size)
                "est_pha" -> estPhaArray = FloatArray(size)
                "phase_real" -> phaseRealArray = FloatArray(size)
                "phase_imag" -> phaseImagArray = FloatArray(size)
            }
        }
        if (phaseOutputMode == "complex" && estPhaArray == null) {
            estPhaArray = FloatArray(magSize)
        }

        // Initialize states to zeros
        resetStates()
        isInitialized = true

        Log.i(TAG, "Tensor pooling initialized:")
        Log.i(TAG, "  - Mag/Pha buffers: ${magSize * FLOAT_BYTES} bytes each")
        Log.i(TAG, "  - State buffers: ${stateNames.size} x 2 (double buffering)")
        val totalStateBytes = stateSizes.values.sum() * FLOAT_BYTES * 2
        Log.i(TAG, "  - Total state memory: ${totalStateBytes / 1024} KB")
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
     * Run inference on a single frame.
     * Uses pre-allocated tensors and double buffering for minimal allocation overhead.
     *
     * @param mag Magnitude spectrogram [1, F, T]
     * @param pha Phase spectrogram [1, F, T]
     * @return Inference result with enhanced mask and phase
     */
    fun run(mag: FloatArray, pha: FloatArray): InferenceResult {
        if (!isInitialized) {
            throw IllegalStateException("StatefulInference not initialized")
        }
        check(!isStateInvalid) {
            "StatefulInference state invalidated by previous error; call resetStates() to recover"
        }
        // B4: enforce strict shape match — fillInputBuffer() previously silently
        // zero-padded/trimmed, hiding upstream geometry bugs.
        require(mag.size == magSize) {
            "mag size mismatch: got ${mag.size}, expected $magSize (shape=${magShape.contentToString()})"
        }
        require(pha.size == phaSize) {
            "pha size mismatch: got ${pha.size}, expected $phaSize (shape=${phaShape.contentToString()})"
        }

        val startTime = System.nanoTime()

        // Copy input data to pre-allocated buffers (zero-copy to JNI)
        fillInputBuffer(magBuffer!!, mag)
        fillInputBuffer(phaBuffer!!, pha)

        // Get active state tensors
        val activeStateTensors = if (useBufferA) stateTensorsA else stateTensorsB

        // Build input map using pre-allocated tensors
        val inputs = mutableMapOf<String, OnnxTensor>()
        inputs["mag"] = magTensor!!
        inputs["pha"] = phaTensor!!
        for (name in stateNames) {
            inputs[name] = activeStateTensors[name]!!
        }

        // Run inference — OrtSession.Result must be closed to release native tensor memory.
        // H2: on ORT exception, state buffers / pre-allocated outputs may be partially
        // written; mark state invalid so caller must resetStates() before next run.
        try {
            backend.run(inputs).use { result ->
                val outputs = result.associate { it.key to (it.value as OnnxTensor) }

                // Parse outputs based on phase_output_mode using pre-allocated arrays
                val estMask = extractIntoArray(outputs["est_mask"]!!, estMaskArray!!)

                val estPha: FloatArray
                if (phaseOutputMode == "complex") {
                    val phaseReal = extractIntoArray(outputs["phase_real"]!!, phaseRealArray!!)
                    val phaseImag = extractIntoArray(outputs["phase_imag"]!!, phaseImagArray!!)
                    estPha = computeAtan2InPlace(phaseImag, phaseReal, estPhaArray!!)
                } else {
                    estPha = extractIntoArray(outputs["est_pha"]!!, estPhaArray!!)
                }

                // Update states using double buffering (swap without copying)
                updateStatesDoubleBuffer(outputs)

                val inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000f

                return InferenceResult(
                    estMask = estMask,
                    estPhase = estPha,
                    inferenceTimeMs = inferenceTimeMs
                )
            }
        } catch (e: Exception) {
            isStateInvalid = true
            throw e
        }
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
        // Close pre-allocated input tensors
        magTensor?.close()
        phaTensor?.close()
        magTensor = null
        phaTensor = null

        // Clear input buffers (Direct ByteBuffers are GC'd)
        magBuffer = null
        phaBuffer = null

        // Close state tensors (both buffer sets)
        stateTensorsA.values.forEach { it.close() }
        stateTensorsB.values.forEach { it.close() }
        stateTensorsA.clear()
        stateTensorsB.clear()

        // Clear state buffers
        stateBuffersA.clear()
        stateBuffersB.clear()

        // Clear pre-allocated output arrays
        estMaskArray = null
        estPhaArray = null
        phaseRealArray = null
        phaseImagArray = null

        // Clear metadata
        stateNames.clear()
        nextStateNames.clear()
        stateShapes.clear()
        stateSizes.clear()

        isInitialized = false
    }
}
