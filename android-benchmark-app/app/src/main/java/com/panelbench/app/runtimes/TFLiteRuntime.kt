package com.panelbench.app.runtimes

import android.content.Context
import android.graphics.Bitmap
import com.panelbench.app.ModelConfig
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class TFLiteRuntime : ModelRuntime {

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null
    private var delegateNote: String = "unknown"
    private var inputW = 0
    private var inputH = 0

    override fun load(context: Context, config: ModelConfig): RuntimeLoadResult {
        val start = System.nanoTime()

        // Copy from assets to a plain file so we can mmap it and know its on-disk size.
        val modelFile = copyAssetToFile(context, config.modelPath)
        inputW = config.inputWidth
        inputH = config.inputHeight

        val options = Interpreter.Options().apply { setNumThreads(config.numThreads) }

        delegateNote = when (config.delegate) {
            "gpu" -> {
                val compatList = CompatibilityList()
                if (compatList.isDelegateSupportedOnThisDevice) {
                    gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                    options.addDelegate(gpuDelegate)
                    "requested=gpu actual=gpu"
                } else {
                    "requested=gpu actual=cpu (gpu delegate unsupported on this device)"
                }
            }
            "nnapi" -> {
                try {
                    nnApiDelegate = NnApiDelegate()
                    options.addDelegate(nnApiDelegate)
                    "requested=nnapi actual=nnapi (verify accelerator via NNAPI logs -- may still fall back per-op)"
                } catch (e: Exception) {
                    "requested=nnapi actual=cpu (delegate init failed: ${e.message})"
                }
            }
            "xnnpack" -> {
                options.setUseXNNPACK(true)
                "requested=xnnpack actual=xnnpack"
            }
            else -> "requested=cpu actual=cpu"
        }

        interpreter = Interpreter(modelFile, options)
        val loadTimeMs = (System.nanoTime() - start) / 1_000_000

        return RuntimeLoadResult(loadTimeMs = loadTimeMs, modelSizeBytes = modelFile.length())
    }

    override fun runInference(input: Bitmap): InferenceOutput {
        val interp = interpreter ?: error("TFLiteRuntime.load() must be called first")

        val resized = Bitmap.createScaledBitmap(input, inputW, inputH, true)

        // Most models here have exactly one input (the image) and one output.
        // Some (e.g. PP-PicoDet, exported with Paddle's NMS baked in) have a
        // second input like "scale_factor" feeding that post-processing.
        // Identify the image input by ELEMENT COUNT (it's always far larger
        // than any auxiliary input), not by shape rank -- onnx2tf can report
        // an auxiliary tensor's shape in a form that isn't cleanly 2D, which
        // made a rank-based check misidentify it and try to copy the full
        // image buffer into a tiny scale_factor slot (confirmed by a real
        // "Cannot copy ... 8 bytes from a Java Buffer with 1228800 bytes" error).
        val inputCount = interp.inputTensorCount
        val elementCounts = IntArray(inputCount) { i ->
            interp.getInputTensor(i).shape().fold(1) { acc, d -> acc * (if (d > 0) d else 1) }
        }
        val imageInputIndex = elementCounts.indices.maxByOrNull { elementCounts[it] } ?: 0

        // TEMPORARY diagnostic logging -- the element-count heuristic above has
        // now failed twice with the exact same "8 bytes vs 1228800 bytes"
        // mismatch even after being applied, meaning the actual runtime shapes
        // don't match what we assumed. Logging the real name/shape/element
        // count of every input tensor here so the next fix is based on ground
        // truth (check via `adb logcat -d | grep PanelBenchmarkDebug`) instead
        // of another guess. Remove once the real cause is found and fixed.
        for (i in 0 until inputCount) {
            val t = interp.getInputTensor(i)
            android.util.Log.d(
                "PanelBenchmarkDebug",
                "input[$i] name=${t.name()} shape=${t.shape().toList()} elementCount=${elementCounts[i]} " +
                    "chosenAsImage=${i == imageInputIndex}"
            )
        }

        val inputs = arrayOfNulls<Any>(inputCount)
        for (i in 0 until inputCount) {
            inputs[i] = if (i == imageInputIndex) {
                bitmapToByteBuffer(resized)
            } else {
                auxInputBuffer(interp.getInputTensor(i).shape())
            }
        }

        // NOTE: output CONTENTS are still model-specific -- this binds outputs
        // by shape so inference runs correctly, but decoding detections out of
        // them (NMS-free single tensor for YOLO26, raw grid + separate NMS for
        // YOLOX/NanoDet, Paddle's multiclass_nms3-style outputs for PicoDet)
        // is left as a placeholder per model family, same as before.
        val outputCount = interp.outputTensorCount
        val outputMap = HashMap<Int, Any>()
        for (i in 0 until outputCount) {
            val shape = interp.getOutputTensor(i).shape()
            outputMap[i] = ByteBuffer
                .allocateDirect(shape.fold(4) { acc, d -> acc * d })
                .order(ByteOrder.nativeOrder())
        }

        if (inputCount == 1 && outputCount == 1) {
            interp.run(inputs[0], outputMap[0])
        } else {
            interp.runForMultipleInputsOutputs(inputs, outputMap)
        }

        return InferenceOutput(numDetections = 0) // fill in post-processing per model family
    }

    override fun release() {
        interpreter?.close()
        gpuDelegate?.close()
        nnApiDelegate?.close()
        interpreter = null
    }

    override fun delegateInfo(): String = delegateNote

    private fun copyAssetToFile(context: Context, assetPath: String): File {
        val outFile = File(context.cacheDir, assetPath.substringAfterLast("/"))
        if (!outFile.exists()) {
            context.assets.open(assetPath).use { input ->
                outFile.outputStream().use { output -> input.copyTo(output) }
            }
        }
        return outFile
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * inputW * inputH * 3)
        buffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputW * inputH)
        bitmap.getPixels(pixels, 0, inputW, 0, 0, inputW, inputH)
        for (px in pixels) {
            buffer.putFloat(((px shr 16 and 0xFF) / 255.0f))
            buffer.putFloat(((px shr 8 and 0xFF) / 255.0f))
            buffer.putFloat(((px and 0xFF) / 255.0f))
        }
        buffer.rewind()
        return buffer
    }

    /**
     * Builds a value for a non-image input tensor on a multi-input model, e.g.
     * PP-PicoDet's "scale_factor" (shape [1, 2]) that feeds its baked-in NMS
     * post-processing. ASSUMPTION: every element is filled with 1.0f, meaning
     * "no rescaling" -- correct for scale_factor specifically, since the
     * bitmap is already resized to exactly the model's expected input
     * dimensions before inference. If a future multi-input model needs a
     * differently-valued auxiliary input, this will need to be extended
     * (e.g. by inspecting tensor name, not just shape) rather than assumed.
     */
    private fun auxInputBuffer(shape: IntArray): ByteBuffer {
        val count = shape.fold(1) { acc, d -> acc * (if (d > 0) d else 1) }
        val buffer = ByteBuffer.allocateDirect(4 * count).order(ByteOrder.nativeOrder())
        repeat(count) { buffer.putFloat(1.0f) }
        buffer.rewind()
        return buffer
    }
}

/** Helper retained for models that ship as raw FileChannel-mmap candidates (rarely needed
 * once copyAssetToFile is used, but kept for reference when loading directly from storage). */
internal fun mapModelFile(path: String): ByteBuffer {
    FileInputStream(path).use { fis ->
        val channel = fis.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
    }
}