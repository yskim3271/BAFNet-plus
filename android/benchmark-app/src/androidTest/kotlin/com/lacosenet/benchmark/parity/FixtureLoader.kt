package com.lacosenet.benchmark.parity

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Loader for Python-generated golden fixtures under androidTest/assets/fixtures/.
 *
 * Fixture layout (matches scripts/make_streaming_golden.py):
 *   - assets/fixtures/manifest.json
 *   - assets/fixtures/input_audio.bin
 *   - assets/fixtures/chunk_000/input_samples.bin, stft_mag.bin, ...
 *
 * Binaries are raw little-endian float32, row-major. Shapes come from the
 * manifest, and the loader exposes them as flat FloatArrays. Reshaping is left
 * to the caller so the comparison path can keep whatever internal layout the
 * Kotlin code under test uses.
 */
class FixtureLoader(private val testContext: Context, private val fixtureRoot: String = "fixtures") {

    data class TensorRef(val file: String, val shape: IntArray, val bytes: Int) {
        val numElements: Int get() = shape.fold(1) { acc, dim -> acc * dim }
    }

    data class ChunkFixture(val idx: Int, val files: Map<String, TensorRef>)

    data class StateLayoutEntry(
        val name: String,
        val shape: IntArray,
        val offsetFloats: Int,
        val sizeFloats: Int,
    )

    data class Manifest(
        val version: Int,
        val numChunks: Int,
        val samplesPerChunk: Int,
        val outputSamplesPerChunk: Int,
        val totalFramesNeeded: Int,
        val olaTailSize: Int,
        val freqBins: Int,
        val chunkSizeFrames: Int,
        val hopSize: Int,
        val winSize: Int,
        val nFft: Int,
        val compressFactor: Float,
        val inputAudio: TensorRef,
        val stateLayout: List<StateLayoutEntry>,
        val stateOrder: List<String>,
        val chunks: List<ChunkFixture>,
    )

    val manifest: Manifest by lazy { loadManifest() }

    /** Load the manifest. Throws if missing or malformed. */
    private fun loadManifest(): Manifest {
        val text = testContext.assets.open("$fixtureRoot/manifest.json").bufferedReader().use { it.readText() }
        val json = JSONObject(text)
        val version = json.getInt("version")
        val derived = json.getJSONObject("derived")
        val stft = json.getJSONObject("stft_config")
        val streaming = json.getJSONObject("streaming_config")

        fun parseTensorRef(o: JSONObject): TensorRef {
            val shape = IntArray(o.getJSONArray("shape").length()) { o.getJSONArray("shape").getInt(it) }
            return TensorRef(
                file = o.getString("file"),
                shape = shape,
                bytes = o.getInt("bytes"),
            )
        }

        val input = parseTensorRef(json.getJSONObject("input_audio"))

        val layoutJson = json.getJSONArray("state_layout")
        val layout = (0 until layoutJson.length()).map { i ->
            val entry = layoutJson.getJSONObject(i)
            val shape = IntArray(entry.getJSONArray("shape").length()) { k ->
                entry.getJSONArray("shape").getInt(k)
            }
            StateLayoutEntry(
                name = entry.getString("name"),
                shape = shape,
                offsetFloats = entry.getInt("offset_floats"),
                sizeFloats = entry.getInt("size_floats"),
            )
        }
        val order = jsonArrayToStringList(json.getJSONArray("state_order"))

        val chunksJson = json.getJSONArray("chunks")
        val chunks = (0 until chunksJson.length()).map { i ->
            val c = chunksJson.getJSONObject(i)
            val files = mutableMapOf<String, TensorRef>()
            val filesJson = c.getJSONObject("files")
            for (key in filesJson.keys()) {
                files[key] = parseTensorRef(filesJson.getJSONObject(key))
            }
            ChunkFixture(idx = c.getInt("idx"), files = files)
        }

        return Manifest(
            version = version,
            numChunks = json.getInt("num_chunks"),
            samplesPerChunk = derived.getInt("samples_per_chunk"),
            outputSamplesPerChunk = derived.getInt("output_samples_per_chunk"),
            totalFramesNeeded = derived.getInt("total_frames_needed"),
            olaTailSize = derived.getInt("ola_tail_size"),
            freqBins = streaming.getInt("freq_bins"),
            chunkSizeFrames = streaming.getInt("chunk_size_frames"),
            hopSize = stft.getInt("hop_size"),
            winSize = stft.getInt("win_length"),
            nFft = stft.getInt("n_fft"),
            compressFactor = stft.getDouble("compress_factor").toFloat(),
            inputAudio = input,
            stateLayout = layout,
            stateOrder = order,
            chunks = chunks,
        )
    }

    /** Read a raw float32 little-endian binary file into a flat FloatArray. */
    fun readFloatArray(relativePath: String): FloatArray {
        testContext.assets.open("$fixtureRoot/$relativePath").use { stream ->
            return readAllFloats(stream)
        }
    }

    /** Read a tensor by its TensorRef. Verifies size matches manifest. */
    fun readTensor(ref: TensorRef): FloatArray {
        val arr = readFloatArray(ref.file)
        check(arr.size == ref.numElements) {
            "Fixture size mismatch at ${ref.file}: got ${arr.size} floats, manifest expected ${ref.numElements}"
        }
        return arr
    }

    private companion object {
        private fun readAllFloats(stream: InputStream): FloatArray {
            val bytes = stream.readBytes()
            require(bytes.size % 4 == 0) { "Binary length ${bytes.size} not multiple of 4" }
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val out = FloatArray(bytes.size / 4)
            bb.asFloatBuffer().get(out)
            return out
        }

        private fun jsonArrayToStringList(a: JSONArray): List<String> {
            return (0 until a.length()).map { a.getString(it) }
        }
    }
}

/**
 * Loader for BAFNetPlus Python-generated golden fixtures under
 * androidTest/assets/bafnetplus_fixtures/ (schema from
 * scripts/make_bafnetplus_streaming_golden.py).
 *
 * Schema differs from [FixtureLoader]:
 *  - Two input audio streams: input_audio_bcs + input_audio_acs.
 *  - No state_layout / state_order — Stage 3 builds the ONNX graph so the
 *    native state contract is defined at export time.
 *  - Per-chunk tensor keys mirror BAFNetPlus' dual-branch pipeline
 *    (input_samples_bcs, bcs_mag, bcs_est_mag, acs_mask, calibration_feat,
 *    common_log_gain, alpha_softmax, est_mag, istft_output, ...).
 */
class BafnetPlusFixtureLoader(
    private val testContext: Context,
    private val fixtureRoot: String = "bafnetplus_fixtures",
) {
    data class Manifest(
        val version: Int,
        val model: String,
        val ablationMode: String,
        val numChunks: Int,
        val samplesPerChunk: Int,
        val outputSamplesPerChunk: Int,
        val totalFramesNeeded: Int,
        val olaTailSize: Int,
        val freqBins: Int,
        val chunkSizeFrames: Int,
        val hopSize: Int,
        val winSize: Int,
        val nFft: Int,
        val compressFactor: Float,
        val inputAudioBcs: FixtureLoader.TensorRef,
        val inputAudioAcs: FixtureLoader.TensorRef,
        val chunks: List<FixtureLoader.ChunkFixture>,
        val calibrationDiagnostics: CalibrationDiagnostics,
    )

    data class CalibrationDiagnostics(
        val commonLogGainStd: Float,
        val relativeLogGainStd: Float,
        val alphaSoftmaxStd: Float,
        val exercised: Boolean,
    )

    val manifest: Manifest by lazy { loadManifest() }

    private fun loadManifest(): Manifest {
        val text = testContext.assets.open("$fixtureRoot/manifest.json").bufferedReader().use { it.readText() }
        val json = JSONObject(text)
        val version = json.getInt("version")
        val derived = json.getJSONObject("derived")
        val stft = json.getJSONObject("stft_config")
        val streaming = json.getJSONObject("streaming_config")

        fun parseTensorRef(o: JSONObject): FixtureLoader.TensorRef {
            val shape = IntArray(o.getJSONArray("shape").length()) { o.getJSONArray("shape").getInt(it) }
            return FixtureLoader.TensorRef(
                file = o.getString("file"),
                shape = shape,
                bytes = o.getInt("bytes"),
            )
        }

        val chunksJson = json.getJSONArray("chunks")
        val chunks = (0 until chunksJson.length()).map { i ->
            val c = chunksJson.getJSONObject(i)
            val files = mutableMapOf<String, FixtureLoader.TensorRef>()
            val filesJson = c.getJSONObject("files")
            for (key in filesJson.keys()) {
                files[key] = parseTensorRef(filesJson.getJSONObject(key))
            }
            FixtureLoader.ChunkFixture(idx = c.getInt("idx"), files = files)
        }

        val diag = json.getJSONObject("calibration_diagnostics")

        return Manifest(
            version = version,
            model = json.getString("model"),
            ablationMode = json.getString("ablation_mode"),
            numChunks = json.getInt("num_chunks"),
            samplesPerChunk = derived.getInt("samples_per_chunk"),
            outputSamplesPerChunk = derived.getInt("output_samples_per_chunk"),
            totalFramesNeeded = derived.getInt("total_frames_needed"),
            olaTailSize = derived.getInt("ola_tail_size"),
            freqBins = derived.getInt("freq_bins"),
            chunkSizeFrames = streaming.getInt("chunk_size_frames"),
            hopSize = stft.getInt("hop_size"),
            winSize = stft.getInt("win_length"),
            nFft = stft.getInt("n_fft"),
            compressFactor = stft.getDouble("compress_factor").toFloat(),
            inputAudioBcs = parseTensorRef(json.getJSONObject("input_audio_bcs")),
            inputAudioAcs = parseTensorRef(json.getJSONObject("input_audio_acs")),
            chunks = chunks,
            calibrationDiagnostics = CalibrationDiagnostics(
                commonLogGainStd = diag.getDouble("common_log_gain_overall_std").toFloat(),
                relativeLogGainStd = diag.getDouble("relative_log_gain_overall_std").toFloat(),
                alphaSoftmaxStd = diag.getDouble("alpha_softmax_overall_std").toFloat(),
                exercised = diag.getBoolean("exercised"),
            ),
        )
    }

    fun readFloatArray(relativePath: String): FloatArray {
        testContext.assets.open("$fixtureRoot/$relativePath").use { stream ->
            val bytes = stream.readBytes()
            require(bytes.size % 4 == 0) { "Binary length ${bytes.size} not multiple of 4" }
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val out = FloatArray(bytes.size / 4)
            bb.asFloatBuffer().get(out)
            return out
        }
    }

    fun readTensor(ref: FixtureLoader.TensorRef): FloatArray {
        val arr = readFloatArray(ref.file)
        check(arr.size == ref.numElements) {
            "Fixture size mismatch at ${ref.file}: got ${arr.size} floats, manifest expected ${ref.numElements}"
        }
        return arr
    }
}

/**
 * Max absolute difference between two FloatArrays (same size).
 */
fun maxAbsDiff(a: FloatArray, b: FloatArray): Float {
    require(a.size == b.size) { "Size mismatch ${a.size} vs ${b.size}" }
    var m = 0f
    for (i in a.indices) {
        val d = kotlin.math.abs(a[i] - b[i])
        if (d > m) m = d
    }
    return m
}

/**
 * Root-mean-square difference between two FloatArrays (same size).
 */
fun rmsDiff(a: FloatArray, b: FloatArray): Float {
    require(a.size == b.size)
    if (a.isEmpty()) return 0f
    var sumSq = 0.0
    for (i in a.indices) {
        val d = (a[i] - b[i]).toDouble()
        sumSq += d * d
    }
    return kotlin.math.sqrt(sumSq / a.size).toFloat()
}
