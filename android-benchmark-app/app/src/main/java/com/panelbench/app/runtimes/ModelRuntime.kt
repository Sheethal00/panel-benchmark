package com.panelbench.app.runtimes

import android.content.Context
import android.graphics.Bitmap
import com.panelbench.app.ModelConfig

/** Raw output of one inference call -- kept generic since detector vs OCR outputs differ. */
data class InferenceOutput(
    val rawScores: FloatArray? = null,
    val boxes: FloatArray? = null,
    val text: String? = null,
    val numDetections: Int = 0
)

data class RuntimeLoadResult(val loadTimeMs: Long, val modelSizeBytes: Long)

/**
 * Common interface every candidate runtime (TFLite, ONNX Runtime, ML Kit, ...) implements.
 * BenchmarkRunner only talks to this interface, so adding a new runtime/model candidate
 * never requires touching the timing/metrics/reporting code.
 */
interface ModelRuntime {
    fun load(context: Context, config: ModelConfig): RuntimeLoadResult
    fun runInference(input: Bitmap): InferenceOutput
    fun release()

    /** Human-readable note on what delegate actually executed the graph. Fill this even on
     * fallback, e.g. "requested=nnapi actual=cpu (fallback)" -- NNAPI silently falling back
     * to CPU is the single most common source of misleading benchmark numbers. */
    fun delegateInfo(): String
}
