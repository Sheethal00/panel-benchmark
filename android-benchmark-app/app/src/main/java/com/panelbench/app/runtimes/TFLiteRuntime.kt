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
        val inputBuffer = bitmapToByteBuffer(resized)

        // NOTE: output shape is model-specific. This is a placeholder single-output binding --
        // for real YOLO-style models you'll typically bind to a [1, N, 4+numClasses] tensor.
        // Replace with the actual output tensor shape of the candidate model under test.
        //
        // YOLO26 specifically ships as NMS-free end-to-end: the exported graph already
        // includes the final box selection, so decoding here is "parse boxes/scores directly
        // from the single output tensor" with no separate NMS loop needed -- simpler than
        // the anchor-decode + NMS post-processing required for YOLOv8/YOLO-NAS-style outputs.
        val outputShape = interp.getOutputTensor(0).shape()
        val outputBuffer = ByteBuffer
            .allocateDirect(outputShape.fold(4) { acc, d -> acc * d })
            .order(ByteOrder.nativeOrder())

        interp.run(inputBuffer, outputBuffer)

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
}

/** Helper retained for models that ship as raw FileChannel-mmap candidates (rarely needed
 * once copyAssetToFile is used, but kept for reference when loading directly from storage). */
internal fun mapModelFile(path: String): ByteBuffer {
    FileInputStream(path).use { fis ->
        val channel = fis.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
    }
}
