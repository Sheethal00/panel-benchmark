package com.panelbench.app

import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val statusView = TextView(this).apply { text = "Idle" }
        val runAllButton = Button(this).apply { text = "Run Full Suite" }
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(statusView)
            addView(runAllButton)
        }
        setContentView(layout)

        val configJson = assets.open("benchmark_config.json").bufferedReader().readText()
        val suite = ConfigLoader.parse(configJson)
        val runner = BenchmarkRunner(this)

        runAllButton.setOnClickListener {
            statusView.text = "Running ${suite.models.size} configs + ${suite.pipelines.size} pipelines..."
            thread {
                try {
                    val sampleImage = loadSampleImage(this)
                    suite.models.forEach { config ->
                        runOnUiThread { statusView.text = "Running: ${config.name}" }
                        runner.runSingle(config, sampleImage)
                    }
                    suite.pipelines.forEach { pipeline ->
                        runOnUiThread { statusView.text = "Running pipeline: ${pipeline.name}" }
                        val detectorConfig = suite.models.first { it.name == pipeline.detectorConfigName }
                        val ocrConfig = suite.models.first { it.name == pipeline.ocrConfigName }
                        runner.runPipeline(pipeline, detectorConfig, ocrConfig, sampleImage)
                    }
                    runOnUiThread { statusView.text = "Done. Results in getExternalFilesDir/results/" }
                } catch (e: Exception) {
                    // Surface the real cause on-screen instead of a silent crash -- e.g.
                    // a missing assets/sample_panel.jpg (FileNotFoundException) would
                    // otherwise kill the whole app with no readable message, since an
                    // uncaught exception on a raw Thread has no default handler here.
                    val message = "FAILED: ${e.javaClass.simpleName}: ${e.message}"
                    runOnUiThread { statusView.text = message }
                    android.util.Log.e("PanelBenchmark", "runAllButton failed", e)
                }
            }
        }

        // Automated single-config mode, driven by scripts/run_benchmark_matrix.py:
        //   adb shell am start -n com.panelbench.app/.MainActivity \
        //       --es config_name yolov8n_int8_nnapi --ez auto_run true
        // Automated pipeline mode:
        //   adb shell am start -n com.panelbench.app/.MainActivity \
        //       --es pipeline_name yolov8n_to_paddleocr --ez auto_run true
        val autoRun = intent.getBooleanExtra("auto_run", false)
        val configName = intent.getStringExtra("config_name")
        val pipelineName = intent.getStringExtra("pipeline_name")

        if (autoRun && configName != null) {
            val config = suite.models.firstOrNull { it.name == configName }
            if (config != null) {
                thread {
                    try {
                        val sampleImage = loadSampleImage(this)
                        runner.runSingle(config, sampleImage)
                        runOnUiThread { statusView.text = "Auto-run complete: $configName" }
                    } catch (e: Exception) {
                        android.util.Log.e("PanelBenchmark", "Auto-run failed: $configName", e)
                        runOnUiThread { statusView.text = "FAILED: ${e.javaClass.simpleName}: ${e.message}" }
                    } finally {
                        finish()
                    }
                }
            }
        } else if (autoRun && pipelineName != null) {
            val pipeline = suite.pipelines.firstOrNull { it.name == pipelineName }
            if (pipeline != null) {
                val detectorConfig = suite.models.firstOrNull { it.name == pipeline.detectorConfigName }
                val ocrConfig = suite.models.firstOrNull { it.name == pipeline.ocrConfigName }
                if (detectorConfig != null && ocrConfig != null) {
                    thread {
                        try {
                            val sampleImage = loadSampleImage(this)
                            runner.runPipeline(pipeline, detectorConfig, ocrConfig, sampleImage)
                            runOnUiThread { statusView.text = "Pipeline auto-run complete: $pipelineName" }
                        } catch (e: Exception) {
                            android.util.Log.e("PanelBenchmark", "Pipeline auto-run failed: $pipelineName", e)
                            runOnUiThread { statusView.text = "FAILED: ${e.javaClass.simpleName}: ${e.message}" }
                        } finally {
                            finish()
                        }
                    }
                }
            }
        }
    }
}