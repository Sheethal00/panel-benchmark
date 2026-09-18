package com.panelbench.app.metrics

import android.content.Context
import android.os.Build
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Full result of one benchmark config run. Mirrors the schema report/generate_report.py expects. */
data class BenchmarkResult(
    val configName: String,
    val task: String,
    val runtime: String,
    val requestedDelegate: String,
    val actualDelegateInfo: String,
    val modelSizeBytes: Long,
    val loadTimeMs: Long,
    val latenciesMs: List<Double>,     // one entry per timed run
    val pssBaselineKb: Int,
    val pssAfterLoadKb: Int,
    val pssPeakDuringInferenceKb: Int,
    // RSS (from /proc/self/status) alongside PSS as a diagnostic, since PSS
    // via ActivityManager.getProcessMemoryInfo() showed baseline/after-load/
    // peak as byte-for-byte IDENTICAL in a real run even with multi-sample
    // delays added -- consistent with that cross-process Binder IPC query
    // being throttled/cached at the OS level on modern Android, rather than
    // genuinely reflecting no memory change. RSS is a same-process file read
    // (no IPC), so it shouldn't be subject to the same throttling -- compare
    // rss_*_kb against pss_*_kb in the next report to confirm before treating
    // RSS as the primary metric.
    val rssBaselineKb: Long = -1,
    val rssAfterLoadKb: Long = -1,
    val rssPeakDuringInferenceKb: Long = -1,
    val deviceModel: String = Build.MODEL,
    val androidSdkInt: Int = Build.VERSION.SDK_INT,
    val soc: String = Build.HARDWARE,
    val error: String? = null
) {
    fun percentile(p: Double): Double {
        if (latenciesMs.isEmpty()) return -1.0
        val sorted = latenciesMs.sorted()
        val idx = (p / 100.0 * (sorted.size - 1)).toInt().coerceIn(0, sorted.size - 1)
        return sorted[idx]
    }

    fun toJson(): JSONObject {
        val o = JSONObject()
        o.put("config_name", configName)
        o.put("task", task)
        o.put("runtime", runtime)
        o.put("requested_delegate", requestedDelegate)
        o.put("actual_delegate_info", actualDelegateInfo)
        o.put("model_size_bytes", modelSizeBytes)
        o.put("load_time_ms", loadTimeMs)
        o.put("latencies_ms", JSONArray(latenciesMs))
        o.put("latency_p50_ms", percentile(50.0))
        o.put("latency_p90_ms", percentile(90.0))
        o.put("latency_p99_ms", percentile(99.0))
        o.put("pss_baseline_kb", pssBaselineKb)
        o.put("pss_after_load_kb", pssAfterLoadKb)
        o.put("pss_peak_during_inference_kb", pssPeakDuringInferenceKb)
        o.put("rss_baseline_kb", rssBaselineKb)
        o.put("rss_after_load_kb", rssAfterLoadKb)
        o.put("rss_peak_during_inference_kb", rssPeakDuringInferenceKb)
        o.put("device_model", deviceModel)
        o.put("android_sdk_int", androidSdkInt)
        o.put("soc", soc)
        if (error != null) o.put("error", error)
        return o
    }
}

object ResultWriter {

    /** Writes to app-specific external files dir, e.g.
     * /sdcard/Android/data/com.panelbench.app/files/results/<name>.json
     * so `adb pull` (via scripts/run_benchmark_matrix.py) can retrieve it without root. */
    fun write(context: Context, result: BenchmarkResult) {
        val dir = File(context.getExternalFilesDir(null), "results")
        if (!dir.exists()) dir.mkdirs()
        val file = File(dir, "${result.configName}.json")
        file.writeText(result.toJson().toString(2))
    }

    fun writePipeline(context: Context, result: PipelineBenchmarkResult) {
        val dir = File(context.getExternalFilesDir(null), "results")
        if (!dir.exists()) dir.mkdirs()
        val file = File(dir, "pipeline_${result.pipelineName}.json")
        file.writeText(result.toJson().toString(2))
    }
}

/**
 * Result of a sequential detector -> OCR run within ONE process lifetime (no force-stop
 * between stages). The number that matters most here is pssPeakOverallKb -- the true
 * worst-case memory the device saw across the whole cycle, which can exceed either
 * stage's isolated peak if native memory isn't fully reclaimed between release() and
 * the next model's load().
 */
data class PipelineBenchmarkResult(
    val pipelineName: String,
    val detectorConfigName: String,
    val ocrConfigName: String,
    val iterations: Int,
    val detectorLatenciesMs: List<Double>,
    val ocrLatenciesMs: List<Double>,
    val pssBaselineKb: Int,               // before anything loaded
    val pssAfterDetectorLoadKb: Int,
    val pssPeakDuringDetectorKb: Int,
    val pssAfterDetectorReleaseKb: Int,   // key number: did release() actually free memory?
    val pssAfterOcrLoadKb: Int,
    val pssPeakDuringOcrKb: Int,
    val pssAfterOcrReleaseKb: Int,
    val pssPeakOverallKb: Int,            // the number to compare against the 200MB budget
    val deviceModel: String = Build.MODEL,
    val androidSdkInt: Int = Build.VERSION.SDK_INT,
    val soc: String = Build.HARDWARE,
    val error: String? = null
) {
    fun percentile(values: List<Double>, p: Double): Double {
        if (values.isEmpty()) return -1.0
        val sorted = values.sorted()
        val idx = (p / 100.0 * (sorted.size - 1)).toInt().coerceIn(0, sorted.size - 1)
        return sorted[idx]
    }

    fun toJson(): JSONObject {
        val o = JSONObject()
        o.put("pipeline_name", pipelineName)
        o.put("detector_config", detectorConfigName)
        o.put("ocr_config", ocrConfigName)
        o.put("iterations", iterations)
        o.put("detector_latencies_ms", JSONArray(detectorLatenciesMs))
        o.put("ocr_latencies_ms", JSONArray(ocrLatenciesMs))
        o.put("detector_latency_p50_ms", percentile(detectorLatenciesMs, 50.0))
        o.put("ocr_latency_p50_ms", percentile(ocrLatenciesMs, 50.0))
        o.put("end_to_end_p50_ms", percentile(detectorLatenciesMs, 50.0) + percentile(ocrLatenciesMs, 50.0))
        o.put("pss_baseline_kb", pssBaselineKb)
        o.put("pss_after_detector_load_kb", pssAfterDetectorLoadKb)
        o.put("pss_peak_during_detector_kb", pssPeakDuringDetectorKb)
        o.put("pss_after_detector_release_kb", pssAfterDetectorReleaseKb)
        o.put("pss_after_ocr_load_kb", pssAfterOcrLoadKb)
        o.put("pss_peak_during_ocr_kb", pssPeakDuringOcrKb)
        o.put("pss_after_ocr_release_kb", pssAfterOcrReleaseKb)
        o.put("pss_peak_overall_kb", pssPeakOverallKb)
        o.put("device_model", deviceModel)
        o.put("android_sdk_int", androidSdkInt)
        o.put("soc", soc)
        if (error != null) o.put("error", error)
        return o
    }
}