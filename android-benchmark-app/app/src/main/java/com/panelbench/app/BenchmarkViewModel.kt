package com.panelbench.app

import android.app.Application
import android.graphics.Bitmap
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import com.panelbench.app.metrics.BenchmarkResult
import com.panelbench.app.metrics.PipelineBenchmarkResult
import kotlin.concurrent.thread

/** Either kind of result the interactive screen can show for a list row. */
sealed class RunOutcome {
    data class Single(val result: BenchmarkResult) : RunOutcome()
    data class Pipeline(val result: PipelineBenchmarkResult) : RunOutcome()
}

/**
 * Backs the interactive "pick a task and run it" screen. Runs stay sequential by
 * design -- this harness loads one model at a time on purpose (matches the real app's
 * behavior and avoids cross-contaminating memory readings between candidates), so the
 * UI disables every other Run action while one is in flight rather than allowing
 * concurrent runs.
 *
 * Uses plain Thread (kotlin.concurrent.thread), matching BenchmarkRunner/MainActivity's
 * existing auto-run path, rather than introducing coroutines as a second concurrency
 * style in the same app. Compose State writes from a background thread are safe --
 * only reads need to happen during composition, which they do here.
 */
class BenchmarkViewModel(application: Application) : AndroidViewModel(application) {

    val suite: BenchmarkSuite by lazy {
        val json = getApplication<Application>().assets.open("benchmark_config.json")
            .bufferedReader().readText()
        ConfigLoader.parse(json)
    }

    private val runner by lazy { BenchmarkRunner(getApplication()) }
    private var sampleImage: Bitmap? = null

    /** name (config or pipeline) -> its latest outcome. */
    val results = mutableStateMapOf<String, RunOutcome>()

    /** name of the config/pipeline currently running, or null if nothing is. */
    var runningName by mutableStateOf<String?>(null)
        private set

    var isRunningFullSuite by mutableStateOf(false)
        private set

    var fullSuiteStatus by mutableStateOf("")
        private set

    /** Non-null only while a run just failed to even start (e.g. missing pipeline config). */
    var lastStartupError by mutableStateOf<String?>(null)
        private set

    private fun ensureSampleImage(): Bitmap =
        sampleImage ?: loadSampleImage(getApplication()).also { sampleImage = it }

    val isBusy: Boolean get() = runningName != null || isRunningFullSuite

    fun runConfig(config: ModelConfig) {
        if (isBusy) return
        lastStartupError = null
        runningName = config.name
        thread {
            try {
                val img = ensureSampleImage()
                val result = runner.runSingle(config, img)
                results[config.name] = RunOutcome.Single(result)
            } catch (e: Exception) {
                android.util.Log.e("PanelBenchmark", "runConfig(${config.name}) failed", e)
            } finally {
                runningName = null
            }
        }
    }

    fun runPipeline(pipeline: PipelineConfig) {
        if (isBusy) return
        val detectorConfig = suite.models.firstOrNull { it.name == pipeline.detectorConfigName }
        val ocrConfig = suite.models.firstOrNull { it.name == pipeline.ocrConfigName }
        if (detectorConfig == null || ocrConfig == null) {
            lastStartupError = "Pipeline '${pipeline.name}' references a missing config " +
                "(detector='${pipeline.detectorConfigName}', ocr='${pipeline.ocrConfigName}')"
            return
        }
        lastStartupError = null
        runningName = pipeline.name
        thread {
            try {
                val img = ensureSampleImage()
                val result = runner.runPipeline(pipeline, detectorConfig, ocrConfig, img)
                results[pipeline.name] = RunOutcome.Pipeline(result)
            } catch (e: Exception) {
                android.util.Log.e("PanelBenchmark", "runPipeline(${pipeline.name}) failed", e)
            } finally {
                runningName = null
            }
        }
    }

    fun runFullSuite() {
        if (isBusy) return
        isRunningFullSuite = true
        fullSuiteStatus = "Starting..."
        thread {
            try {
                val img = ensureSampleImage()
                suite.models.forEachIndexed { i, config ->
                    fullSuiteStatus = "Running model ${i + 1}/${suite.models.size}: ${config.name}"
                    val result = runner.runSingle(config, img)
                    results[config.name] = RunOutcome.Single(result)
                }
                suite.pipelines.forEachIndexed { i, pipeline ->
                    fullSuiteStatus = "Running pipeline ${i + 1}/${suite.pipelines.size}: ${pipeline.name}"
                    val detectorConfig = suite.models.firstOrNull { it.name == pipeline.detectorConfigName }
                    val ocrConfig = suite.models.firstOrNull { it.name == pipeline.ocrConfigName }
                    if (detectorConfig != null && ocrConfig != null) {
                        val result = runner.runPipeline(pipeline, detectorConfig, ocrConfig, img)
                        results[pipeline.name] = RunOutcome.Pipeline(result)
                    }
                }
                fullSuiteStatus = "Done. Results in getExternalFilesDir/results/"
            } catch (e: Exception) {
                android.util.Log.e("PanelBenchmark", "runFullSuite failed", e)
                fullSuiteStatus = "FAILED: ${e.javaClass.simpleName}: ${e.message}"
            } finally {
                isRunningFullSuite = false
            }
        }
    }
}
