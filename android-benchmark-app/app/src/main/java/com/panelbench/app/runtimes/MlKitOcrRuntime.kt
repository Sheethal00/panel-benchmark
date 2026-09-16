package com.panelbench.app.runtimes

import android.content.Context
import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.panelbench.app.ModelConfig
import java.util.concurrent.TimeUnit

/**
 * ML Kit's on-device text recognizer -- a zero-conversion-cost OCR baseline to compare
 * custom OCR candidates (PaddleOCR mobile, a CRNN, etc.) against. There is no model file
 * to load or size: ML Kit's recognition model ships bundled with the library and is
 * initialized lazily by the client rather than read from a path we control, so
 * `modelSizeBytes` is reported as 0 here (not a real "0 bytes", just "not applicable" --
 * don't read this as ML Kit being free on APK size or memory).
 *
 * ML Kit's process() call is async (returns a Play Services Task), but the rest of this
 * harness times synchronous calls -- Tasks.await() blocks the calling thread until the
 * Task completes, so runInference()'s timing still reflects real end-to-end OCR latency.
 */
class MlKitOcrRuntime : ModelRuntime {

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    override fun load(context: Context, config: ModelConfig): RuntimeLoadResult {
        val start = System.nanoTime()
        // No file to read -- instead, run one throwaway inference on a tiny blank bitmap
        // so "load time" captures the recognizer's real first-use initialization cost
        // (native library init, on-device model warm-up), matching what "load" means for
        // the file-based runtimes rather than reporting a meaningless ~0ms.
        val warmupBitmap = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888)
        val warmupImage = InputImage.fromBitmap(warmupBitmap, 0)
        Tasks.await(recognizer.process(warmupImage), 30, TimeUnit.SECONDS)
        val loadTimeMs = (System.nanoTime() - start) / 1_000_000
        return RuntimeLoadResult(loadTimeMs = loadTimeMs, modelSizeBytes = 0)
    }

    override fun runInference(input: Bitmap): InferenceOutput {
        val image = InputImage.fromBitmap(input, 0)
        val result: Text = Tasks.await(recognizer.process(image), 30, TimeUnit.SECONDS)
        return InferenceOutput(text = result.text, numDetections = result.textBlocks.size)
    }

    override fun release() {
        recognizer.close()
    }

    // ML Kit doesn't expose a CPU/GPU/NNAPI delegate choice the way TFLite/ONNX Runtime do --
    // it picks its own execution path internally. Reported here for consistency with the
    // other runtimes' delegateInfo(), not because there's a knob to turn.
    override fun delegateInfo(): String = "mlkit on-device (no user-configurable delegate)"
}
