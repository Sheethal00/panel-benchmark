package com.panelbench.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.panelbench.app.metrics.BenchmarkResult
import com.panelbench.app.metrics.MemoryProfiler
import com.panelbench.app.metrics.PipelineBenchmarkResult
import com.panelbench.app.metrics.ResultWriter
import com.panelbench.app.runtimes.MlKitOcrRuntime
import com.panelbench.app.runtimes.ModelRuntime
import com.panelbench.app.runtimes.OnnxRuntime
import com.panelbench.app.runtimes.TFLiteRuntime

class BenchmarkRunner(private val context: Context) {

    /** Runs every config in the suite sequentially, writing one result JSON per config.
     * Loading/releasing per config (rather than keeping models resident) mirrors how the
     * real app will behave -- one model loaded at a time -- and avoids cross-contaminating
     * memory readings between candidates. */
    fun runSuite(suite: BenchmarkSuite, sampleImage: Bitmap) {
        for (config in suite.models) {
            runSingle(config, sampleImage)
        }
    }

    fun runSingle(config: ModelConfig, sampleImage: Bitmap): BenchmarkResult {
        val runtime: ModelRuntime = createRuntime(config.runtime)

        val baseline = MemoryProfiler.sample(context)

        val result = try {
            val loadResult = runtime.load(context, config)
            val afterLoad = sampleMemorySettled()

            // Warmup: JIT/delegate compilation, caches, first-run overhead -- excluded from
            // the timed numbers so they reflect steady-state field performance.
            repeat(config.warmupRuns) {
                runtime.runInference(sampleImage)
            }

            val latencies = mutableListOf<Double>()
            var peakPssKb = afterLoad.pssKb
            var peakRssKb = afterLoad.rssKb

            repeat(config.timedRuns) {
                val t0 = System.nanoTime()
                runtime.runInference(sampleImage)
                val t1 = System.nanoTime()
                latencies.add((t1 - t0) / 1_000_000.0)

                if (it % 10 == 0) {
                    val sample = MemoryProfiler.sample(context)
                    if (sample.pssKb > peakPssKb) peakPssKb = sample.pssKb
                    if (sample.rssKb > peakRssKb) peakRssKb = sample.rssKb
                }
            }

            BenchmarkResult(
                configName = config.name,
                task = config.task,
                runtime = config.runtime,
                requestedDelegate = config.delegate,
                actualDelegateInfo = runtime.delegateInfo(),
                modelSizeBytes = loadResult.modelSizeBytes,
                loadTimeMs = loadResult.loadTimeMs,
                latenciesMs = latencies,
                pssBaselineKb = baseline.pssKb,
                pssAfterLoadKb = afterLoad.pssKb,
                pssPeakDuringInferenceKb = peakPssKb,
                rssBaselineKb = baseline.rssKb,
                rssAfterLoadKb = afterLoad.rssKb,
                rssPeakDuringInferenceKb = peakRssKb
            )
        } catch (e: Exception) {
            BenchmarkResult(
                configName = config.name,
                task = config.task,
                runtime = config.runtime,
                requestedDelegate = config.delegate,
                actualDelegateInfo = "error",
                modelSizeBytes = 0,
                loadTimeMs = -1,
                latenciesMs = emptyList(),
                pssBaselineKb = baseline.pssKb,
                pssAfterLoadKb = -1,
                pssPeakDuringInferenceKb = -1,
                rssBaselineKb = baseline.rssKb,
                rssAfterLoadKb = -1,
                rssPeakDuringInferenceKb = -1,
                error = e.stackTraceToString()
            )
        } finally {
            runtime.release()
        }

        ResultWriter.write(context, result)
        return result
    }

    private fun createRuntime(name: String): ModelRuntime = when (name) {
        "tflite" -> TFLiteRuntime()
        "onnx" -> OnnxRuntime()
        "mlkit" -> MlKitOcrRuntime()
        else -> error("Unknown runtime: $name")
    }

    /**
     * Samples memory multiple times with a short delay between each, returning
     * the sample with the highest PSS seen. A single point-in-time snapshot
     * taken immediately after load() can miss a real memory increase --
     * observed in practice: several small models showed a ~0 delta between
     * baseline and after-load despite genuinely allocating a few MB, likely
     * because Android's PSS accounting hadn't caught up with newly-committed
     * pages yet at that exact instant. Spreading samples over a short window
     * makes it more likely to catch the settled, real value. Used only for
     * post-load sampling (where the goal is catching a settling INCREASE) --
     * not for post-release sampling, where taking a max could incorrectly
     * mask how much memory was actually reclaimed.
     */
    private fun sampleMemorySettled(samples: Int = 3, delayMs: Long = 50): MemoryProfiler.MemorySample {
        var best = MemoryProfiler.sample(context)
        repeat(samples - 1) {
            Thread.sleep(delayMs)
            val s = MemoryProfiler.sample(context)
            if (s.pssKb > best.pssKb) best = s
        }
        return best
    }

    /**
     * Runs detector -> OCR sequentially within ONE process lifetime (no force-stop between
     * stages, matching your real pipeline behavior). This is deliberately NOT just "run
     * config A's isolated numbers plus config B's isolated numbers" -- the point is to catch
     * the case where release() doesn't fully return native memory to the OS before the next
     * model loads, which force-stopping between isolated runs would hide.
     *
     * Requires detectorConfig.task == "detector" and ocrConfig.task == "ocr".
     */
    fun runPipeline(
        pipeline: PipelineConfig,
        detectorConfig: ModelConfig,
        ocrConfig: ModelConfig,
        sampleImage: Bitmap
    ): PipelineBenchmarkResult {
        require(detectorConfig.task == "detector") { "${detectorConfig.name} is not a detector config" }
        require(ocrConfig.task == "ocr") { "${ocrConfig.name} is not an ocr config" }

        val baseline = MemoryProfiler.sample(context)
        var peakOverallKb = baseline.pssKb

        fun trackPeak(kb: Int) {
            if (kb > peakOverallKb) peakOverallKb = kb
        }

        return try {
            // --- Detector stage ---
            val detectorRuntime = createRuntime(detectorConfig.runtime)
            detectorRuntime.load(context, detectorConfig)
            val afterDetectorLoad = sampleMemorySettled()
            trackPeak(afterDetectorLoad.pssKb)

            repeat(detectorConfig.warmupRuns) { detectorRuntime.runInference(sampleImage) }

            val detectorLatencies = mutableListOf<Double>()
            var peakDuringDetector = afterDetectorLoad.pssKb
            repeat(pipeline.iterations) {
                val t0 = System.nanoTime()
                detectorRuntime.runInference(sampleImage)
                val t1 = System.nanoTime()
                detectorLatencies.add((t1 - t0) / 1_000_000.0)
                val s = MemoryProfiler.sample(context)
                if (s.pssKb > peakDuringDetector) peakDuringDetector = s.pssKb
                trackPeak(s.pssKb)
            }

            detectorRuntime.release()
            val afterDetectorRelease = MemoryProfiler.sample(context)
            trackPeak(afterDetectorRelease.pssKb)

            // --- OCR stage (detector should be fully released by now) ---
            val ocrRuntime = createRuntime(ocrConfig.runtime)
            ocrRuntime.load(context, ocrConfig)
            val afterOcrLoad = sampleMemorySettled()
            trackPeak(afterOcrLoad.pssKb)

            repeat(ocrConfig.warmupRuns) { ocrRuntime.runInference(sampleImage) }

            val ocrLatencies = mutableListOf<Double>()
            var peakDuringOcr = afterOcrLoad.pssKb
            repeat(pipeline.iterations) {
                val t0 = System.nanoTime()
                ocrRuntime.runInference(sampleImage)
                val t1 = System.nanoTime()
                ocrLatencies.add((t1 - t0) / 1_000_000.0)
                val s = MemoryProfiler.sample(context)
                if (s.pssKb > peakDuringOcr) peakDuringOcr = s.pssKb
                trackPeak(s.pssKb)
            }

            ocrRuntime.release()
            val afterOcrRelease = MemoryProfiler.sample(context)
            trackPeak(afterOcrRelease.pssKb)

            val result = PipelineBenchmarkResult(
                pipelineName = pipeline.name,
                detectorConfigName = detectorConfig.name,
                ocrConfigName = ocrConfig.name,
                iterations = pipeline.iterations,
                detectorLatenciesMs = detectorLatencies,
                ocrLatenciesMs = ocrLatencies,
                pssBaselineKb = baseline.pssKb,
                pssAfterDetectorLoadKb = afterDetectorLoad.pssKb,
                pssPeakDuringDetectorKb = peakDuringDetector,
                pssAfterDetectorReleaseKb = afterDetectorRelease.pssKb,
                pssAfterOcrLoadKb = afterOcrLoad.pssKb,
                pssPeakDuringOcrKb = peakDuringOcr,
                pssAfterOcrReleaseKb = afterOcrRelease.pssKb,
                pssPeakOverallKb = peakOverallKb
            )
            ResultWriter.writePipeline(context, result)
            result
        } catch (e: Exception) {
            val result = PipelineBenchmarkResult(
                pipelineName = pipeline.name,
                detectorConfigName = detectorConfig.name,
                ocrConfigName = ocrConfig.name,
                iterations = pipeline.iterations,
                detectorLatenciesMs = emptyList(),
                ocrLatenciesMs = emptyList(),
                pssBaselineKb = baseline.pssKb,
                pssAfterDetectorLoadKb = -1,
                pssPeakDuringDetectorKb = -1,
                pssAfterDetectorReleaseKb = -1,
                pssAfterOcrLoadKb = -1,
                pssPeakDuringOcrKb = -1,
                pssAfterOcrReleaseKb = -1,
                pssPeakOverallKb = peakOverallKb,
                error = e.stackTraceToString()
            )
            ResultWriter.writePipeline(context, result)
            result
        }
    }
}

/** Loads the shared test image used for every timed run. Swap for a loop over
 * test_data/images (the *.jpg files there) if you want per-image latency variance too. */
fun loadSampleImage(context: Context, assetPath: String = "sample_panel.jpg"): Bitmap {
    context.assets.open(assetPath).use { return BitmapFactory.decodeStream(it) }
}