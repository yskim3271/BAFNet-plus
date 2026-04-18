/**
 * BackendSelector.kt
 *
 * Automatic selection of the best available execution backend.
 * Priority: QNN HTP > NNAPI > CPU
 */
package com.lacosenet.streaming.backend

import android.content.Context
import android.os.Build
import android.util.Log
import com.lacosenet.streaming.core.StreamingConfig

/**
 * Selects the optimal execution backend based on device capabilities.
 *
 * Selection priority:
 * 1. QNN HTP - ONNX Runtime backend for Qualcomm SoCs with INT8 ONNX model
 * 2. NNAPI - For non-Qualcomm devices or Android < 15
 * 3. CPU - Always available as fallback
 *
 * @param context Android context for device capability checks
 */
class BackendSelector(private val context: Context) {

    companion object {
        private const val TAG = "BackendSelector"

        // Known Qualcomm SoC identifiers
        private val QUALCOMM_IDENTIFIERS = listOf(
            "qcom", "qualcomm", "sm8", "sm7", "sm6",
            "sdm", "msm", "apq", "snapdragon"
        )

        // Android 15 API level where NNAPI is deprecated
        private const val NNAPI_DEPRECATED_API = 35

        // B6: cache libQnnHtp.so availability so selectBestBackend()/createBackendChain()/
        // getDeviceInfo() do not repeatedly invoke System.loadLibrary and its exception path.
        private val qnnAvailableCache: Boolean by lazy {
            try {
                System.loadLibrary("QnnHtp")
                true
            } catch (e: UnsatisfiedLinkError) {
                try {
                    System.loadLibrary("qnn_htp")
                    true
                } catch (e2: UnsatisfiedLinkError) {
                    false
                }
            }
        }
    }

    /**
     * Select the best available backend.
     *
     * @param config Optional streaming config to check model compatibility
     * @return Selected backend type
     */
    fun selectBestBackend(config: StreamingConfig? = null): BackendType {
        Log.d(TAG, "Selecting backend for device: ${Build.HARDWARE}")

        // Check ONNX Runtime QNN availability
        if (isQualcommDevice() && isQnnAvailable()) {
            // QNN HTP requires INT8 quantized model
            val modelCompatible = config?.modelInfo?.quantization?.contains("int8") ?: true
            if (modelCompatible) {
                Log.i(TAG, "Selected: QNN_HTP (ONNX Runtime QNN EP)")
                return BackendType.QNN_HTP
            } else {
                Log.w(TAG, "QNN available but model not INT8 quantized")
            }
        }

        // Check NNAPI availability (not deprecated)
        if (isNnapiAvailable() && !isNnapiDeprecated()) {
            Log.i(TAG, "Selected: NNAPI")
            return BackendType.NNAPI
        }

        // Fallback to CPU
        Log.i(TAG, "Selected: CPU (fallback)")
        return BackendType.CPU
    }

    /**
     * Create a backend instance for the selected type.
     */
    fun createBackend(type: BackendType): ExecutionBackend {
        return when (type) {
            BackendType.QNN_HTP -> QnnBackend(context)
            BackendType.NNAPI -> NnapiBackend(context)
            BackendType.CPU -> CpuBackend()
        }
    }

    /**
     * Create backends with automatic fallback chain.
     *
     * Returns a list of backends to try in order.
     */
    @Suppress("UNUSED_PARAMETER")
    fun createBackendChain(config: StreamingConfig? = null): List<ExecutionBackend> {
        val chain = mutableListOf<ExecutionBackend>()

        // Try QNN first if available
        if (isQualcommDevice() && isQnnAvailable()) {
            chain.add(QnnBackend(context))
        }

        // NNAPI as fallback (if not deprecated)
        if (isNnapiAvailable() && !isNnapiDeprecated()) {
            chain.add(NnapiBackend(context))
        }

        // CPU always available
        chain.add(CpuBackend())

        Log.d(TAG, "Backend chain: ${chain.map { it.type }}")
        return chain
    }

    /**
     * Check if this is a Qualcomm device.
     */
    fun isQualcommDevice(): Boolean {
        val hardware = Build.HARDWARE.lowercase()
        val board = Build.BOARD.lowercase()
        val socManufacturer = try {
            Build::class.java.getField("SOC_MANUFACTURER")
                .get(null)?.toString()?.lowercase() ?: ""
        } catch (e: Exception) {
            ""
        }

        val isQualcomm = QUALCOMM_IDENTIFIERS.any { id ->
            hardware.contains(id) || board.contains(id) || socManufacturer.contains(id)
        }

        Log.d(TAG, "Qualcomm check: hardware=$hardware, board=$board, isQualcomm=$isQualcomm")
        return isQualcomm
    }

    /**
     * Check if QNN libraries are available.
     * B6: first call loads libQnnHtp.so (or the lowercase fallback); subsequent calls
     * read the cached result without touching System.loadLibrary.
     */
    fun isQnnAvailable(): Boolean = qnnAvailableCache

    /**
     * Check if NNAPI is available.
     */
    fun isNnapiAvailable(): Boolean {
        // NNAPI available from Android 8.1 (API 27)
        return Build.VERSION.SDK_INT >= 27
    }

    /**
     * Check if NNAPI is deprecated (Android 15+).
     */
    fun isNnapiDeprecated(): Boolean {
        return Build.VERSION.SDK_INT >= NNAPI_DEPRECATED_API
    }

    /**
     * Get device information for debugging.
     */
    fun getDeviceInfo(): Map<String, String> {
        return mapOf(
            "manufacturer" to Build.MANUFACTURER,
            "model" to Build.MODEL,
            "hardware" to Build.HARDWARE,
            "board" to Build.BOARD,
            "sdk_int" to Build.VERSION.SDK_INT.toString(),
            "is_qualcomm" to isQualcommDevice().toString(),
            "qnn_available" to isQnnAvailable().toString(),
            "nnapi_available" to isNnapiAvailable().toString(),
            "nnapi_deprecated" to isNnapiDeprecated().toString()
        )
    }
}
