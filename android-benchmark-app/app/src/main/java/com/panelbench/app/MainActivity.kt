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
                    val sampleImage = loadSampleImage(this)
                    runner.runSingle(config, sampleImage)
                    runOnUiThread { statusView.text = "Auto-run complete: $configName" }
                    finish()
                }
            }
        } else if (autoRun && pipelineName != null) {
            val pipeline = suite.pipelines.firstOrNull { it.name == pipelineName }
            if (pipeline != null) {
                val detectorConfig = suite.models.firstOrNull { it.name == pipeline.detectorConfigName }
                val ocrConfig = suite.models.firstOrNull { it.name == pipeline.ocrConfigName }
                if (detectorConfig != null && ocrConfig != null) {
                    thread {
                        val sampleImage = loadSampleImage(this)
                        runner.runPipeline(pipeline, detectorConfig, ocrConfig, sampleImage)
                        runOnUiThread { statusView.text = "Pipeline auto-run complete: $pipelineName" }
                        finish()
                    }
                }
            }
        }
    }
}
