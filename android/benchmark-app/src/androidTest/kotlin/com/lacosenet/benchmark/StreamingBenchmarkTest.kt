package com.lacosenet.benchmark

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.FloatBuffer
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * LaCoSENet Streaming Benchmark.
 *
 * Benchmarks unified model inference with stateful session pattern.
 *
 * Real-time constraint: chunk_size_frames * 6.25ms budget per chunk (from streaming_config.json)
 *
 * Run with:
 *   ./gradlew :benchmark-app:connectedAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.StreamingBenchmarkTest
 */
@RunWith(AndroidJUnit4::class)
class StreamingBenchmarkTest {

    companion object {
        private const val TAG = "StreamingBenchmark"
        private const val MODEL_FILE = "model.onnx"
        private const val CHUNKS_PER_SESSION = 4
        private const val WARMUP_ITERATIONS = 10
        private const val BENCHMARK_ITERATIONS = 50

        // Real-time constraint
        private const val HOP_SIZE_MS = 6.25f    // 100 samples @ 16kHz
    }

    private lateinit var context: Context
    private lateinit var ortEnv: OrtEnvironment
    private var qnnAvailable = false
    private var modelConfig: ModelConfig? = null

    data class ModelConfig(
        val name: String,
        val version: String,
        val quantization: String,
        val numStates: Int,
        val encoderLookahead: Int,
        val decoderLookahead: Int,
        val chunkSizeFrames: Int,
        val exportTimeFrames: Int,
        val exportTimestamp: String?,
        val checkpointMd5: String?,
        val gitCommit: String?
    )

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

    private fun loadModelConfig(): ModelConfig? {
        return try {
            val jsonStr = context.assets.open("streaming_config.json").bufferedReader().readText()
            val json = JSONObject(jsonStr)

            val modelInfo = json.optJSONObject("model_info")
            val streamingConfig = json.optJSONObject("streaming_config")
            val exportInfo = json.optJSONObject("export_info")
            val stateInfo = json.optJSONObject("state_info")

            ModelConfig(
                name = modelInfo?.optString("name", "unknown") ?: "unknown",
                version = modelInfo?.optString("version", "unknown") ?: "unknown",
                quantization = modelInfo?.optString("quantization", "unknown") ?: "unknown",
                numStates = stateInfo?.optInt("num_states", 0) ?: 0,
                encoderLookahead = streamingConfig?.optInt("encoder_lookahead", 0) ?: 0,
                decoderLookahead = streamingConfig?.optInt("decoder_lookahead", 0) ?: 0,
                chunkSizeFrames = streamingConfig?.optInt("chunk_size_frames", 32) ?: 32,
                exportTimeFrames = streamingConfig?.optInt("export_time_frames", 40) ?: 40,
                exportTimestamp = exportInfo?.optString("timestamp"),
                checkpointMd5 = exportInfo?.optString("checkpoint_md5"),
                gitCommit = exportInfo?.optString("git_commit")
            )
        } catch (e: Exception) {
            Log.w(TAG, "Could not load streaming_config.json: ${e.message}")
            null
        }
    }

    private fun logModelConfig(config: ModelConfig) {
        Log.i(TAG, "-".repeat(70))
        Log.i(TAG, "MODEL IDENTIFICATION")
        Log.i(TAG, "-".repeat(70))
        Log.i(TAG, "  Name:             ${config.name}")
        Log.i(TAG, "  Version:          ${config.version}")
        Log.i(TAG, "  Quantization:     ${config.quantization}")
        Log.i(TAG, "  Num states:       ${config.numStates}")
        Log.i(TAG, "  Encoder lookahead: ${config.encoderLookahead}")
        Log.i(TAG, "  Decoder lookahead: ${config.decoderLookahead}")
        if (config.exportTimestamp != null) {
            Log.i(TAG, "  Export date:      ${config.exportTimestamp}")
        }
        if (config.checkpointMd5 != null) {
            Log.i(TAG, "  Checkpoint MD5:   ${config.checkpointMd5}")
        }
        if (config.gitCommit != null) {
            Log.i(TAG, "  Git commit:       ${config.gitCommit}")
        }
        Log.i(TAG, "")
    }

    @Test
    fun benchmarkCpu() {
        val sessionOptions = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(4)
        }
        runBenchmark("CPU (4 threads)", sessionOptions)
    }

    @Test
    fun benchmarkCpuThreadSweep() {
        val threadCounts = intArrayOf(1, 2, 4, 6, 8)
        for (numThreads in threadCounts) {
            val sessionOptions = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(numThreads)
            }
            runBenchmark("CPU ($numThreads threads)", sessionOptions)
        }
    }

    @Test
    fun benchmarkCpuMemoryPattern() {
        val sessionOptions = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(4)
            setMemoryPatternOptimization(true)
        }
        runBenchmark("CPU (4 threads, memPattern)", sessionOptions)
    }

    @Test
    fun benchmarkQnnHtp() {
        if (!qnnAvailable) {
            Log.w(TAG, "QNN HTP not available, skipping benchmark")
            return
        }

        Log.i(TAG, "QNN backend_path: libQnnHtp.so (full options)")

        val providerOptions = mapOf(
            "backend_path" to "libQnnHtp.so",
            "htp_performance_mode" to "burst",
            "htp_graph_finalization_optimization_mode" to "3",
            "enable_htp_fp16_precision" to "1",
            "enable_htp_shared_memory_allocator" to "1",
            "vtcm_mb" to "8",
            "qnn_context_priority" to "high",
        )

        val sessionOptions = OrtSession.SessionOptions()
        try {
            sessionOptions.addQnn(providerOptions)
            Log.i(TAG, "QNN EP registered with full options")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add QNN EP: ${e.message}")
            return
        }
        runBenchmark("QNN HTP (FP16, full opts)", sessionOptions)
    }

    @Test
    fun benchmarkQnnHtpWithProfiling() {
        if (!qnnAvailable) {
            Log.w(TAG, "QNN HTP not available, skipping benchmark")
            return
        }

        val profilingPath = "${context.cacheDir}/qnn_profile.csv"
        Log.i(TAG, "QNN profiling output: $profilingPath")

        val providerOptions = mapOf(
            "backend_path" to "libQnnHtp.so",
            "htp_performance_mode" to "burst",
            "htp_graph_finalization_optimization_mode" to "3",
            "enable_htp_fp16_precision" to "1",
            "enable_htp_shared_memory_allocator" to "1",
            "vtcm_mb" to "8",
            "qnn_context_priority" to "high",
            "profiling_level" to "basic",
            "profiling_file_path" to profilingPath,
        )

        val sessionOptions = OrtSession.SessionOptions()
        try {
            sessionOptions.addQnn(providerOptions)
            Log.i(TAG, "QNN EP registered with profiling")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add QNN EP: ${e.message}")
            return
        }
        runBenchmark("QNN HTP (FP16, profiling)", sessionOptions)

        Log.i(TAG, "Profiling CSV written to: $profilingPath")
        Log.i(TAG, "Pull with: adb pull $profilingPath ./qnn_profile.csv")
    }

    @Test
    fun benchmarkQnnHtpCached() {
        if (!qnnAvailable) {
            Log.w(TAG, "QNN HTP not available, skipping benchmark")
            return
        }

        // Copy model from assets to filesDir (assets are read-only, context cache needs writable path)
        val modelFile = copyModelToFilesDir(MODEL_FILE)
        if (modelFile == null) {
            Log.e(TAG, "Failed to copy model to filesDir")
            return
        }
        val modelPath = modelFile.absolutePath
        val contextCachePath = "${context.filesDir}/qnn_ctx_cache.onnx"

        // Detect if this is a QDQ model (check quantization from config)
        val config = loadModelConfig()
        val isQdq = config?.quantization?.contains("qdq", ignoreCase = true) == true ||
                     config?.quantization?.contains("int8", ignoreCase = true) == true
        val fp16Flag = if (isQdq) "0" else "1"
        val label = if (isQdq) "QDQ INT8" else "FP16"

        Log.i(TAG, "Context cache benchmark: model=$modelPath, cache=$contextCachePath")
        Log.i(TAG, "  Model type: $label, enable_htp_fp16_precision=$fp16Flag")

        val providerOptions = mapOf(
            "backend_path" to "libQnnHtp.so",
            "htp_performance_mode" to "burst",
            "htp_graph_finalization_optimization_mode" to "3",
            "enable_htp_fp16_precision" to fp16Flag,
            "enable_htp_shared_memory_allocator" to "1",
            "vtcm_mb" to "8",
            "qnn_context_priority" to "high",
        )

        // Delete old cache if exists
        File(contextCachePath).delete()

        // Phase 1: Create session with context caching enabled (compiles graph, saves cache)
        Log.i(TAG, "Phase 1: Compiling HTP graph and caching context...")
        val compileStart = System.nanoTime()
        val sessionOptions1 = OrtSession.SessionOptions().apply {
            addQnn(providerOptions)
            addConfigEntry("ep.context_enable", "1")
            addConfigEntry("ep.context_file_path", contextCachePath)
            addConfigEntry("ep.context_embed_mode", "1")
        }
        val session1 = ortEnv.createSession(modelPath, sessionOptions1)
        val compileTimeMs = (System.nanoTime() - compileStart) / 1_000_000.0
        session1.close()
        Log.i(TAG, "  Graph compilation + cache: %.1fms".format(compileTimeMs))

        val cacheFile = File(contextCachePath)
        if (cacheFile.exists()) {
            Log.i(TAG, "  Cache file size: %.1f MB".format(cacheFile.length() / (1024.0 * 1024.0)))
        } else {
            Log.e(TAG, "  Context cache file not created!")
            return
        }

        // Phase 2: Load from cached context and benchmark
        Log.i(TAG, "Phase 2: Loading from cached context and benchmarking...")
        val loadStart = System.nanoTime()
        val sessionOptions2 = OrtSession.SessionOptions().apply {
            addQnn(providerOptions)
        }
        val cachedModelPath = contextCachePath
        val loadTimeMs = (System.nanoTime() - loadStart) / 1_000_000.0
        Log.i(TAG, "  Session options setup: %.1fms".format(loadTimeMs))

        runBenchmarkWithPath("QNN HTP ($label, cached)", sessionOptions2, cachedModelPath)

        // Cleanup cache file
        cacheFile.delete()
    }

    @Test
    fun benchmarkQnnHtpQdq() {
        if (!qnnAvailable) {
            Log.w(TAG, "QNN HTP not available, skipping benchmark")
            return
        }

        // Check if QDQ model exists in assets
        val qdqModelFile = "model_qdq.onnx"
        val qdqExists = try {
            context.assets.open(qdqModelFile).close()
            true
        } catch (e: Exception) {
            false
        }

        if (!qdqExists) {
            Log.w(TAG, "QDQ model not found in assets: $qdqModelFile, skipping")
            return
        }

        Log.i(TAG, "QDQ INT8 benchmark (native HTP path)")

        val providerOptions = mapOf(
            "backend_path" to "libQnnHtp.so",
            "htp_performance_mode" to "burst",
            "htp_graph_finalization_optimization_mode" to "3",
            "enable_htp_fp16_precision" to "0",  // INT8 native, no FP16 conversion
            "enable_htp_shared_memory_allocator" to "1",
            "vtcm_mb" to "8",
            "qnn_context_priority" to "high",
        )

        val sessionOptions = OrtSession.SessionOptions()
        try {
            sessionOptions.addQnn(providerOptions)
            Log.i(TAG, "QNN EP registered for QDQ INT8")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add QNN EP: ${e.message}")
            return
        }
        runBenchmark("QNN HTP (QDQ INT8, full opts)", sessionOptions, qdqModelFile)
    }

    @Test
    fun benchmarkQnnHtpQdqCached() {
        if (!qnnAvailable) {
            Log.w(TAG, "QNN HTP not available, skipping benchmark")
            return
        }

        // Check if QDQ model exists in assets
        val qdqModelFile = "model_qdq.onnx"
        val qdqExists = try {
            context.assets.open(qdqModelFile).close()
            true
        } catch (e: Exception) {
            false
        }

        if (!qdqExists) {
            Log.w(TAG, "QDQ model not found in assets: $qdqModelFile, skipping")
            return
        }

        // Copy QDQ model to filesDir
        val modelFile = copyModelToFilesDir(qdqModelFile)
        if (modelFile == null) {
            Log.e(TAG, "Failed to copy QDQ model to filesDir")
            return
        }
        val modelPath = modelFile.absolutePath
        val contextCachePath = "${context.filesDir}/qnn_ctx_cache_qdq.onnx"

        val providerOptions = mapOf(
            "backend_path" to "libQnnHtp.so",
            "htp_performance_mode" to "burst",
            "htp_graph_finalization_optimization_mode" to "3",
            "enable_htp_fp16_precision" to "0",
            "enable_htp_shared_memory_allocator" to "1",
            "vtcm_mb" to "8",
            "qnn_context_priority" to "high",
        )

        // Delete old cache
        File(contextCachePath).delete()

        // Phase 1: Compile and cache
        Log.i(TAG, "QDQ INT8 + context cache: compiling...")
        val compileStart = System.nanoTime()
        val sessionOptions1 = OrtSession.SessionOptions().apply {
            addQnn(providerOptions)
            addConfigEntry("ep.context_enable", "1")
            addConfigEntry("ep.context_file_path", contextCachePath)
            addConfigEntry("ep.context_embed_mode", "1")
        }
        val session1 = ortEnv.createSession(modelPath, sessionOptions1)
        val compileTimeMs = (System.nanoTime() - compileStart) / 1_000_000.0
        session1.close()
        Log.i(TAG, "  Compile + cache: %.1fms".format(compileTimeMs))

        val cacheFile = File(contextCachePath)
        if (!cacheFile.exists()) {
            Log.e(TAG, "  Context cache file not created!")
            return
        }
        Log.i(TAG, "  Cache file size: %.1f MB".format(cacheFile.length() / (1024.0 * 1024.0)))

        // Phase 2: Load from cache and benchmark
        val sessionOptions2 = OrtSession.SessionOptions().apply {
            addQnn(providerOptions)
        }
        runBenchmarkWithPath("QNN HTP (QDQ INT8, cached)", sessionOptions2, contextCachePath)

        cacheFile.delete()
    }

    /**
     * Copy a model file from assets to filesDir for writable access.
     */
    private fun copyModelToFilesDir(assetName: String): File? {
        return try {
            val outFile = File(context.filesDir, assetName)
            context.assets.open(assetName).use { input ->
                outFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            Log.i(TAG, "Copied $assetName to ${outFile.absolutePath} (${outFile.length()} bytes)")
            outFile
        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy $assetName: ${e.message}")
            null
        }
    }

    @Test
    fun benchmarkNnapi() {
        val sessionOptions = OrtSession.SessionOptions()
        try {
            sessionOptions.addNnapi()
            Log.i(TAG, "NNAPI EP registered successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add NNAPI EP: ${e.message}")
            return
        }
        runBenchmark("NNAPI", sessionOptions)
    }

    private fun runBenchmark(
        backendLabel: String,
        sessionOptions: OrtSession.SessionOptions,
        modelFileName: String = MODEL_FILE,
    ) {
        // Load model from assets as bytes
        val modelExists = try {
            context.assets.open(modelFileName).close()
            true
        } catch (e: Exception) {
            false
        }
        if (!modelExists) {
            Log.e(TAG, "Model file not found in assets: $modelFileName")
            return
        }
        val modelBytes = context.assets.open(modelFileName).readBytes()
        val session = ortEnv.createSession(modelBytes, sessionOptions)
        runBenchmarkWithSession(backendLabel, session)
    }

    private fun runBenchmarkWithPath(
        backendLabel: String,
        sessionOptions: OrtSession.SessionOptions,
        modelPath: String,
    ) {
        val session = ortEnv.createSession(modelPath, sessionOptions)
        runBenchmarkWithSession(backendLabel, session)
    }

    private fun runBenchmarkWithSession(backendLabel: String, session: OrtSession) {
        // Load and log model config for traceability
        modelConfig = loadModelConfig()

        val chunkSizeFrames = modelConfig?.chunkSizeFrames ?: 32
        val exportTimeFrames = modelConfig?.exportTimeFrames ?: 40
        val realtimeBudgetMs = chunkSizeFrames * HOP_SIZE_MS  // chunk audio length = real-time budget

        Log.i(TAG, "=" .repeat(70))
        Log.i(TAG, "BENCHMARK: $backendLabel")
        Log.i(TAG, "=" .repeat(70))
        Log.i(TAG, "Real-time budget: ${realtimeBudgetMs}ms per chunk")
        Log.i(TAG, "  Chunk size frames: $chunkSizeFrames")
        Log.i(TAG, "  Export time frames: $exportTimeFrames")
        Log.i(TAG, "  Hop size: ${HOP_SIZE_MS}ms")
        Log.i(TAG, "  Chunks per session: $CHUNKS_PER_SESSION")
        Log.i(TAG, "  Warmup sessions: $WARMUP_ITERATIONS")
        Log.i(TAG, "  Benchmark sessions: $BENCHMARK_ITERATIONS")
        Log.i(TAG, "")
        modelConfig?.let { logModelConfig(it) }

        // Build input map from session inputInfo (auto-detect shapes)
        val inputMap = mutableMapOf<String, OnnxTensor>()
        val tensors = mutableListOf<OnnxTensor>()

        for ((name, nodeInfo) in session.inputInfo) {
            val tensorInfo = nodeInfo.info as ai.onnxruntime.TensorInfo
            val shape = tensorInfo.shape
            val size = shape.reduce { a, b -> a * b }.toInt()

            val data = if (name.startsWith("state_")) {
                FloatArray(size)  // zero-initialized for state
            } else {
                FloatArray(size) { Random.nextFloat() }
            }

            val tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(data), shape)
            inputMap[name] = tensor
            tensors.add(tensor)

            Log.i(TAG, "  Input '$name': shape=${shape.toList()}")
        }

        // Warmup
        Log.i(TAG, "Warmup: $WARMUP_ITERATIONS sessions x $CHUNKS_PER_SESSION chunks")
        repeat(WARMUP_ITERATIONS) {
            repeat(CHUNKS_PER_SESSION) {
                session.run(inputMap).close()
            }
        }

        // Benchmark
        Log.i(TAG, "Benchmark: $BENCHMARK_ITERATIONS sessions x $CHUNKS_PER_SESSION chunks")
        val perChunkLatencies = mutableListOf<Double>()
        val perSessionLatencies = mutableListOf<Double>()
        val chunkPositionLatencies = List(CHUNKS_PER_SESSION) { mutableListOf<Double>() }

        repeat(BENCHMARK_ITERATIONS) { sessionIdx ->
            var sessionTotal = 0.0

            repeat(CHUNKS_PER_SESSION) { chunkIdx ->
                val start = System.nanoTime()
                val result = session.run(inputMap)
                val elapsed = (System.nanoTime() - start) / 1_000_000.0
                result.close()

                perChunkLatencies.add(elapsed)
                chunkPositionLatencies[chunkIdx].add(elapsed)
                sessionTotal += elapsed
            }

            perSessionLatencies.add(sessionTotal)

            if ((sessionIdx + 1) % 10 == 0) {
                Log.i(TAG, "Progress: ${sessionIdx + 1}/$BENCHMARK_ITERATIONS sessions")
            }
        }

        // Cleanup
        tensors.forEach { it.close() }
        session.close()

        // Report results
        val stats = calculateStats(perChunkLatencies)
        val isRealtime = stats.p95 < realtimeBudgetMs
        val rtStatus = if (isRealtime) "REALTIME" else "NOT REALTIME"

        Log.i(TAG, "")
        Log.i(TAG, "=" .repeat(70))
        Log.i(TAG, "RESULTS [$backendLabel] [$rtStatus]")
        Log.i(TAG, "=" .repeat(70))

        val config = modelConfig
        if (config != null) {
            Log.i(TAG, "Model: ${config.name}, Git: ${config.gitCommit?.take(8) ?: "N/A"}")
            Log.i(TAG, "")
        }

        Log.i(TAG, String.format("  Per-chunk (%d samples):", perChunkLatencies.size))
        Log.i(TAG, String.format("    Mean: %.1fms, P95: %.1fms, P99: %.1fms", stats.mean, stats.p95, stats.p99))
        Log.i(TAG, String.format("  Per-session (%d sessions):", perSessionLatencies.size))
        Log.i(TAG, String.format("    %d-chunk session mean: %.1fms", CHUNKS_PER_SESSION, perSessionLatencies.average()))
        Log.i(TAG, "  Chunk position analysis:")
        for (i in 0 until CHUNKS_PER_SESSION) {
            Log.i(TAG, String.format("    chunk[%d]: %.1fms", i, chunkPositionLatencies[i].average()))
        }

        Log.i(TAG, String.format("  Budget: %.1fms per chunk", realtimeBudgetMs))
        Log.i(TAG, "")
        Log.i(TAG, "=" .repeat(70))
    }

    private fun calculateStats(latencies: List<Double>): Stats {
        val mean = latencies.average()
        val std = sqrt(latencies.map { (it - mean) * (it - mean) }.average())
        val sorted = latencies.sorted()
        return Stats(
            mean = mean,
            std = std,
            p50 = sorted[sorted.size / 2],
            p95 = sorted[(sorted.size * 0.95).toInt()],
            p99 = sorted[(sorted.size * 0.99).toInt()],
            min = sorted.first(),
            max = sorted.last()
        )
    }

    data class Stats(
        val mean: Double,
        val std: Double,
        val p50: Double,
        val p95: Double,
        val p99: Double,
        val min: Double,
        val max: Double
    )

    // --- Phase C: Dual Backbone Parallel Inference ---

    data class DualBenchmarkResult(
        val mappingMs: Double,
        val maskingMs: Double,
        val wallClockMs: Double,
        val overlapRatio: Double
    )

    private data class SessionInputs(
        val inputMap: Map<String, OnnxTensor>,
        val tensors: List<OnnxTensor>
    )

    private fun createQnnHtpSessionOptions(useQdq: Boolean): OrtSession.SessionOptions? {
        if (!qnnAvailable) {
            Log.w(TAG, "QNN HTP not available")
            return null
        }
        val providerOptions = mapOf(
            "backend_path" to "libQnnHtp.so",
            "htp_performance_mode" to "burst",
            "htp_graph_finalization_optimization_mode" to "3",
            "enable_htp_fp16_precision" to if (useQdq) "0" else "1",
            "enable_htp_shared_memory_allocator" to "1",
            "vtcm_mb" to "8",
            "qnn_context_priority" to "high",
        )
        val sessionOptions = OrtSession.SessionOptions()
        try {
            sessionOptions.addQnn(providerOptions)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add QNN EP: ${e.message}")
            return null
        }
        return sessionOptions
    }

    private fun createSessionInputs(session: OrtSession): SessionInputs {
        val inputMap = mutableMapOf<String, OnnxTensor>()
        val tensors = mutableListOf<OnnxTensor>()
        for ((name, nodeInfo) in session.inputInfo) {
            val tensorInfo = nodeInfo.info as ai.onnxruntime.TensorInfo
            val shape = tensorInfo.shape
            val size = shape.reduce { a, b -> a * b }.toInt()
            val data = if (name.startsWith("state_")) {
                FloatArray(size)
            } else {
                FloatArray(size) { Random.nextFloat() }
            }
            val tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(data), shape)
            inputMap[name] = tensor
            tensors.add(tensor)
        }
        return SessionInputs(inputMap, tensors)
    }

    private fun reportDualBenchmarkResults(label: String, results: List<DualBenchmarkResult>) {
        val mappingStats = calculateStats(results.map { it.mappingMs })
        val maskingStats = calculateStats(results.map { it.maskingMs })
        val wallClockStats = calculateStats(results.map { it.wallClockMs })
        val overlapRatios = results.map { it.overlapRatio }
        val overlapMean = overlapRatios.average()
        val overlapSorted = overlapRatios.sorted()
        val overlapP95 = overlapSorted[(overlapSorted.size * 0.95).toInt()]
        val overlapP99 = overlapSorted[(overlapSorted.size * 0.99).toInt()]

        Log.i(TAG, "")
        Log.i(TAG, "=".repeat(70))
        Log.i(TAG, "DUAL BACKBONE RESULTS [$label]")
        Log.i(TAG, "=".repeat(70))
        Log.i(TAG, String.format("  Mapping:    Mean=%.1fms, P95=%.1fms, P99=%.1fms",
            mappingStats.mean, mappingStats.p95, mappingStats.p99))
        Log.i(TAG, String.format("  Masking:    Mean=%.1fms, P95=%.1fms, P99=%.1fms",
            maskingStats.mean, maskingStats.p95, maskingStats.p99))
        Log.i(TAG, String.format("  WallClock:  Mean=%.1fms, P95=%.1fms, P99=%.1fms",
            wallClockStats.mean, wallClockStats.p95, wallClockStats.p99))
        Log.i(TAG, String.format("  Sum of individuals: %.1fms",
            mappingStats.mean + maskingStats.mean))
        Log.i(TAG, String.format("  Overlap ratio: Mean=%.3f, P95=%.3f, P99=%.3f",
            overlapMean, overlapP95, overlapP99))
        Log.i(TAG, "")
        if (overlapMean > 0.3) {
            Log.i(TAG, "  → HTP achieves significant parallelism (overlap > 0.3)")
        } else if (overlapMean > 0.0) {
            Log.i(TAG, "  → Marginal parallelism (overlap 0.0-0.3)")
        } else {
            Log.i(TAG, "  → No parallelism or contention overhead (overlap ≤ 0)")
        }
        Log.i(TAG, "=".repeat(70))
    }

    @Test
    fun benchmarkDualBackboneSequential() {
        val sessionOptions1 = createQnnHtpSessionOptions(useQdq = false) ?: return
        val sessionOptions2 = createQnnHtpSessionOptions(useQdq = false) ?: return

        val modelBytes = try {
            context.assets.open(MODEL_FILE).readBytes()
        } catch (e: Exception) {
            Log.e(TAG, "Model file not found: $MODEL_FILE")
            return
        }

        val session1 = ortEnv.createSession(modelBytes, sessionOptions1)
        val session2 = try {
            ortEnv.createSession(modelBytes, sessionOptions2)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create second HTP session: ${e.message}")
            session1.close()
            return
        }

        val inputs1 = createSessionInputs(session1)
        val inputs2 = createSessionInputs(session2)

        Log.i(TAG, "=".repeat(70))
        Log.i(TAG, "DUAL BACKBONE: Sequential (FP16)")
        Log.i(TAG, "=".repeat(70))

        // Warmup
        repeat(WARMUP_ITERATIONS) {
            repeat(CHUNKS_PER_SESSION) {
                session1.run(inputs1.inputMap).close()
                session2.run(inputs2.inputMap).close()
            }
        }

        // Benchmark
        val results = mutableListOf<DualBenchmarkResult>()
        repeat(BENCHMARK_ITERATIONS) { sessionIdx ->
            repeat(CHUNKS_PER_SESSION) {
                val start1 = System.nanoTime()
                val result1 = session1.run(inputs1.inputMap)
                val mapping = (System.nanoTime() - start1) / 1_000_000.0
                result1.close()

                val start2 = System.nanoTime()
                val result2 = session2.run(inputs2.inputMap)
                val masking = (System.nanoTime() - start2) / 1_000_000.0
                result2.close()

                results.add(DualBenchmarkResult(mapping, masking, mapping + masking, 0.0))
            }
            if ((sessionIdx + 1) % 10 == 0) {
                Log.i(TAG, "Progress: ${sessionIdx + 1}/$BENCHMARK_ITERATIONS")
            }
        }

        reportDualBenchmarkResults("Sequential FP16", results)

        inputs1.tensors.forEach { it.close() }
        inputs2.tensors.forEach { it.close() }
        session1.close()
        session2.close()
    }

    @Test
    fun benchmarkDualBackboneConcurrent() {
        val sessionOptions1 = createQnnHtpSessionOptions(useQdq = false) ?: return
        val sessionOptions2 = createQnnHtpSessionOptions(useQdq = false) ?: return

        val modelBytes = try {
            context.assets.open(MODEL_FILE).readBytes()
        } catch (e: Exception) {
            Log.e(TAG, "Model file not found: $MODEL_FILE")
            return
        }

        val session1 = ortEnv.createSession(modelBytes, sessionOptions1)
        val session2 = try {
            ortEnv.createSession(modelBytes, sessionOptions2)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create second HTP session: ${e.message}")
            session1.close()
            return
        }

        val inputs1 = createSessionInputs(session1)
        val inputs2 = createSessionInputs(session2)
        val executor = Executors.newFixedThreadPool(2)

        Log.i(TAG, "=".repeat(70))
        Log.i(TAG, "DUAL BACKBONE: Concurrent (FP16)")
        Log.i(TAG, "=".repeat(70))

        // Warmup
        repeat(WARMUP_ITERATIONS) {
            repeat(CHUNKS_PER_SESSION) {
                session1.run(inputs1.inputMap).close()
                session2.run(inputs2.inputMap).close()
            }
        }

        // Benchmark
        val results = mutableListOf<DualBenchmarkResult>()
        repeat(BENCHMARK_ITERATIONS) { sessionIdx ->
            repeat(CHUNKS_PER_SESSION) {
                val wallClockStart = System.nanoTime()

                val future1 = executor.submit(Callable {
                    val start = System.nanoTime()
                    val result = session1.run(inputs1.inputMap)
                    val elapsed = (System.nanoTime() - start) / 1_000_000.0
                    result.close()
                    elapsed
                })
                val future2 = executor.submit(Callable {
                    val start = System.nanoTime()
                    val result = session2.run(inputs2.inputMap)
                    val elapsed = (System.nanoTime() - start) / 1_000_000.0
                    result.close()
                    elapsed
                })

                val mapping = future1.get()
                val masking = future2.get()
                val wallClock = (System.nanoTime() - wallClockStart) / 1_000_000.0
                val overlapRatio = (mapping + masking - wallClock) / wallClock

                results.add(DualBenchmarkResult(mapping, masking, wallClock, overlapRatio))
            }
            if ((sessionIdx + 1) % 10 == 0) {
                Log.i(TAG, "Progress: ${sessionIdx + 1}/$BENCHMARK_ITERATIONS")
            }
        }

        reportDualBenchmarkResults("Concurrent FP16", results)

        executor.shutdown()
        inputs1.tensors.forEach { it.close() }
        inputs2.tensors.forEach { it.close() }
        session1.close()
        session2.close()
    }

    @Test
    fun benchmarkDualBackboneConcurrentQdq() {
        val qdqModelFile = "model_qdq.onnx"
        val qdqExists = try {
            context.assets.open(qdqModelFile).close()
            true
        } catch (e: Exception) {
            false
        }
        if (!qdqExists) {
            Log.w(TAG, "QDQ model not found: $qdqModelFile, skipping")
            return
        }

        val sessionOptions1 = createQnnHtpSessionOptions(useQdq = true) ?: return
        val sessionOptions2 = createQnnHtpSessionOptions(useQdq = true) ?: return

        val modelBytes = context.assets.open(qdqModelFile).readBytes()

        val session1 = ortEnv.createSession(modelBytes, sessionOptions1)
        val session2 = try {
            ortEnv.createSession(modelBytes, sessionOptions2)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create second HTP session (QDQ): ${e.message}")
            session1.close()
            return
        }

        val inputs1 = createSessionInputs(session1)
        val inputs2 = createSessionInputs(session2)
        val executor = Executors.newFixedThreadPool(2)

        Log.i(TAG, "=".repeat(70))
        Log.i(TAG, "DUAL BACKBONE: Concurrent (QDQ INT8)")
        Log.i(TAG, "=".repeat(70))

        // Warmup
        repeat(WARMUP_ITERATIONS) {
            repeat(CHUNKS_PER_SESSION) {
                session1.run(inputs1.inputMap).close()
                session2.run(inputs2.inputMap).close()
            }
        }

        // Benchmark
        val results = mutableListOf<DualBenchmarkResult>()
        repeat(BENCHMARK_ITERATIONS) { sessionIdx ->
            repeat(CHUNKS_PER_SESSION) {
                val wallClockStart = System.nanoTime()

                val future1 = executor.submit(Callable {
                    val start = System.nanoTime()
                    val result = session1.run(inputs1.inputMap)
                    val elapsed = (System.nanoTime() - start) / 1_000_000.0
                    result.close()
                    elapsed
                })
                val future2 = executor.submit(Callable {
                    val start = System.nanoTime()
                    val result = session2.run(inputs2.inputMap)
                    val elapsed = (System.nanoTime() - start) / 1_000_000.0
                    result.close()
                    elapsed
                })

                val mapping = future1.get()
                val masking = future2.get()
                val wallClock = (System.nanoTime() - wallClockStart) / 1_000_000.0
                val overlapRatio = (mapping + masking - wallClock) / wallClock

                results.add(DualBenchmarkResult(mapping, masking, wallClock, overlapRatio))
            }
            if ((sessionIdx + 1) % 10 == 0) {
                Log.i(TAG, "Progress: ${sessionIdx + 1}/$BENCHMARK_ITERATIONS")
            }
        }

        reportDualBenchmarkResults("Concurrent QDQ INT8", results)

        executor.shutdown()
        inputs1.tensors.forEach { it.close() }
        inputs2.tensors.forEach { it.close() }
        session1.close()
        session2.close()
    }
}
