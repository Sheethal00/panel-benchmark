package com.panelbench.app

import org.json.JSONArray
import org.json.JSONObject

/**
 * One entry in benchmark_config.json == one (model, runtime, delegate) combination
 * to benchmark. The whole point of this app is to loop over a list of these
 * so you don't need a separate app build per candidate model.
 */
data class ModelConfig(
    val name: String,            // unique id, e.g. "yolov8n_int8_nnapi"
    val task: String,            // "detector" | "ocr"
    val modelPath: String,       // path relative to assets/models/, or absolute on-device path
    val runtime: String,         // "tflite" | "onnx" | "mlkit"
    val delegate: String,        // "cpu" | "nnapi" | "gpu" | "xnnpack"
    val inputWidth: Int,
    val inputHeight: Int,
    val warmupRuns: Int = 10,
    val timedRuns: Int = 50,
    val numThreads: Int = 4,
    val labelsPath: String? = null   // for detector configs, class label file
)

data class BenchmarkSuite(val models: List<ModelConfig>, val pipelines: List<PipelineConfig> = emptyList())

/**
 * A sequential detector -> OCR run within a single process lifetime (not force-stopped
 * between stages), to measure the realistic worst-case memory profile: detector loads,
 * runs, releases; OCR loads, runs, releases -- all without Android necessarily reclaiming
 * native/delegate memory back to the OS between the two. Peak PSS across the WHOLE
 * sequence is what matters here, not each stage's isolated number.
 */
data class PipelineConfig(
    val name: String,
    val detectorConfigName: String,  // must match a ModelConfig.name with task == "detector"
    val ocrConfigName: String,       // must match a ModelConfig.name with task == "ocr"
    val iterations: Int = 5          // number of full detector->OCR cycles per run
)

object ConfigLoader {

    fun parse(json: String): BenchmarkSuite {
        val root = JSONObject(json)
        val arr: JSONArray = root.getJSONArray("models")
        val list = mutableListOf<ModelConfig>()
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            list.add(
                ModelConfig(
                    name = o.getString("name"),
                    task = o.getString("task"),
                    modelPath = o.getString("model_path"),
                    runtime = o.getString("runtime"),
                    delegate = o.optString("delegate", "cpu"),
                    inputWidth = o.getInt("input_width"),
                    inputHeight = o.getInt("input_height"),
                    warmupRuns = o.optInt("warmup_runs", 10),
                    timedRuns = o.optInt("timed_runs", 50),
                    numThreads = o.optInt("num_threads", 4),
                    labelsPath = if (o.has("labels_path")) o.getString("labels_path") else null
                )
            )
        }

        val pipelines = mutableListOf<PipelineConfig>()
        if (root.has("pipelines")) {
            val pArr = root.getJSONArray("pipelines")
            for (i in 0 until pArr.length()) {
                val o = pArr.getJSONObject(i)
                pipelines.add(
                    PipelineConfig(
                        name = o.getString("name"),
                        detectorConfigName = o.getString("detector_config"),
                        ocrConfigName = o.getString("ocr_config"),
                        iterations = o.optInt("iterations", 5)
                    )
                )
            }
        }

        return BenchmarkSuite(list, pipelines)
    }
}
