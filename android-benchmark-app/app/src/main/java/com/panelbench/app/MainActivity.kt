package com.panelbench.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import kotlin.concurrent.thread

class MainActivity : ComponentActivity() {

    private val viewModel: BenchmarkViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Interactive UI: pick a task from the list, tap to run it, see the result
        // inline. Harmless to set up even in auto-run mode below -- it's simply never
        // shown, since finish() is called as soon as the automated run completes.
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    BenchmarkScreen(viewModel)
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
        val suite = viewModel.suite
        val runner = BenchmarkRunner(this)

        if (autoRun && configName != null) {
            val config = suite.models.firstOrNull { it.name == configName }
            if (config != null) {
                thread {
                    try {
                        val sampleImage = loadSampleImage(this)
                        runner.runSingle(config, sampleImage)
                    } catch (e: Exception) {
                        android.util.Log.e("PanelBenchmark", "Auto-run failed: $configName", e)
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
                        } catch (e: Exception) {
                            android.util.Log.e("PanelBenchmark", "Pipeline auto-run failed: $pipelineName", e)
                        } finally {
                            finish()
                        }
                    }
                }
            }
        }
    }
}