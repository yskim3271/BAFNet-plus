package com.lacosenet.benchmark.parity

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Stage 2 smoke: load bafnetplus_fixtures/ via [BafnetPlusFixtureLoader] and
 * verify manifest round-trip + one chunk's tensors deserialize to the declared
 * shape. This only validates the fixture pipeline — bit-level parity checks
 * against a Kotlin BAFNetPlus runtime are Stage 4 scope.
 */
@RunWith(AndroidJUnit4::class)
class BafnetPlusFixtureSmokeTest {
    private val loader: BafnetPlusFixtureLoader by lazy {
        BafnetPlusFixtureLoader(InstrumentationRegistry.getInstrumentation().context)
    }

    @Test
    fun loadsBafnetPlusManifest() {
        val m = loader.manifest
        assertEquals("BAFNetPlus", m.model)
        assertEquals("full", m.ablationMode)
        assertTrue("num_chunks > 0", m.numChunks > 0)
        assertEquals(m.numChunks, m.chunks.size)
        assertEquals(1200, m.samplesPerChunk)
        assertEquals(800, m.outputSamplesPerChunk)
        assertEquals(11, m.totalFramesNeeded)
        assertEquals(300, m.olaTailSize)
        assertEquals(201, m.freqBins)
        assertEquals(8, m.chunkSizeFrames)
        assertEquals(100, m.hopSize)
        assertEquals(400, m.winSize)
        assertEquals(400, m.nFft)
        assertEquals(0.3f, m.compressFactor, 1e-6f)
    }

    @Test
    fun loadsInputAudioStreams() {
        val m = loader.manifest
        val bcs = loader.readTensor(m.inputAudioBcs)
        val acs = loader.readTensor(m.inputAudioAcs)
        assertEquals(bcs.size, acs.size)
        assertEquals(m.inputAudioBcs.numElements, bcs.size)
    }

    @Test
    fun chunk000HasExpectedKeys() {
        val chunk = loader.manifest.chunks[0]
        val required = listOf(
            "input_samples_bcs", "input_samples_acs",
            "stft_context_bcs_in", "stft_context_acs_in",
            "stft_input_bcs", "stft_input_acs",
            "bcs_mag", "bcs_pha", "acs_mag", "acs_pha",
            "bcs_est_mag", "bcs_est_pha", "bcs_com_out",
            "acs_est_mag", "acs_est_pha", "acs_com_out", "acs_mask",
            "calibration_feat", "calibration_hidden",
            "common_log_gain", "relative_log_gain",
            "bcs_com_cal", "acs_com_cal",
            "alpha_softmax", "est_mag", "est_pha",
            "ola_buffer_in", "ola_norm_in", "ola_buffer_out", "ola_norm_out",
            "istft_output",
        )
        for (key in required) {
            assertTrue("missing chunk 0 tensor $key", chunk.files.containsKey(key))
            val ref = chunk.files.getValue(key)
            val arr = loader.readTensor(ref)
            assertEquals("$key size", ref.numElements, arr.size)
        }
    }

    @Test
    fun calibrationDiagnosticsExercised() {
        val diag = loader.manifest.calibrationDiagnostics
        // Plan R2-2: calibration path should be exercised. Actual common_log_gain
        // saturates near -0.5 with synthetic input, so exercise is read from
        // relative_log_gain OR alpha_softmax std crossing 0.01.
        assertTrue(
            "calibration exercised (common_std=${diag.commonLogGainStd}, " +
                "relative_std=${diag.relativeLogGainStd}, alpha_std=${diag.alphaSoftmaxStd})",
            diag.exercised,
        )
    }
}
