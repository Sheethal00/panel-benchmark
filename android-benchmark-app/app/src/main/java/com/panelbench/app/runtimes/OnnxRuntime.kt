package com.panelbench.app.runtimes

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import com.panelbench.app.ModelConfig
import java.io.File
import java.nio.FloatBuffer

class OnnxRuntime : ModelRuntime {

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var delegateNote = "unknown"
    private var inputW = 0
    private var inputH = 0

    override fun load(context: Context, config: ModelConfig): RuntimeLoadResult {
        val start = System.nanoTime()
        val modelFile = copyAssetToFile(context, config.modelPath)
        inputW = config.inputWidth
        inputH = config.inputHeight

        env = OrtEnvironment.getEnvironment()
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(config.numThreads)
        }

        delegateNote = when (config.delegate) {
            "nnapi" -> {
                try {
                    sessionOptions.addNnapi()
                    "requested=nnapi actual=nnapi"
                } catch (e: Exception) {
                    "requested=nnapi actual=cpu (NNAPI EP unavailable: ${e.message})"
                }
            }
            else -> "requested=cpu actual=cpu"
        }

        session = env!!.createSession(modelFile.absolutePath, sessionOptions)

        val loadTimeMs = (System.nanoTime() - start) / 1_000_000
        return RuntimeLoadResult(loadTimeMs = loadTimeMs, modelSizeBytes = modelFile.length())
    }

    override fun runInference(input: Bitmap): InferenceOutput {
        val s = session ?: error("OnnxRuntime.load() must be called first")
        val e = env ?: error("OnnxRuntime.load() must be called first")

        val resized = Bitmap.createScaledBitmap(input, inputW, inputH, true)
        val floatData = bitmapToCHWFloatArray(resized)
        val imageShape = longArrayOf(1, 3, inputH.toLong(), inputW.toLong())
        val imageElementCount = floatData.size

        // Most models here have exactly one input (the image). Some (e.g.
        // PP-PicoDet, exported with Paddle's NMS baked in) have a second
        // input like "scale_factor" feeding that post-processing.
        //
        // Iterate session.inputNames (the ground truth for required feeds)
        // rather than session.inputInfo -- inputInfo was observed to NOT
        // reliably include "scale_factor" for this export even though the
        // graph requires it at run time (confirmed by a real
        // "Missing Input: scale_factor" error when only inputInfo's entries
        // were used to build the feed map). For each name, prefer real shape
        // info when available (largest element count = the image, same
        // reasoning as TFLiteRuntime); if shape info is unavailable for a
        // name, fall back to name-based detection ("scale" in the name) --
        // the only currently-known case needing this fallback is exactly
        // PicoDet's scale_factor, shape [1, 2], value 1.0 ("no rescaling").
        val inputTensors = mutableMapOf<String, OnnxTensor>()
        try {
            if (s.inputNames.size == 1) {
                val name = s.inputNames.first()
                inputTensors[name] = OnnxTensor.createTensor(e, FloatBuffer.wrap(floatData), imageShape)
            } else {
                for (name in s.inputNames) {
                    val tensorInfo = (s.inputInfo[name]?.info) as? TensorInfo
                    val shape = tensorInfo?.shape
                    val elementCount = shape?.fold(1L) { acc, d -> acc * (if (d > 0) d else 1) }?.toInt()

                    inputTensors[name] = when {
                        elementCount != null && elementCount >= imageElementCount ->
                            OnnxTensor.createTensor(e, FloatBuffer.wrap(floatData), imageShape)
                        elementCount != null ->
                            OnnxTensor.createTensor(e, FloatBuffer.wrap(FloatArray(elementCount) { 1.0f }), shape!!)
                        name.contains("scale", ignoreCase = true) ->
                            OnnxTensor.createTensor(e, FloatBuffer.wrap(floatArrayOf(1.0f, 1.0f)), longArrayOf(1, 2))
                        else ->
                            OnnxTensor.createTensor(e, FloatBuffer.wrap(floatData), imageShape)
                    }
                }
            }

            s.run(inputTensors).use { results ->
                // Output parsing is model-specific -- bind to the real output name/shape
                // for the detector/OCR head under test.
                return InferenceOutput(numDetections = 0)
            }
        } finally {
            inputTensors.values.forEach { it.close() }
        }
    }

    override fun release() {
        session?.close()
        env?.close()
        session = null
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

    private fun bitmapToCHWFloatArray(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(inputW * inputH)
        bitmap.getPixels(pixels, 0, inputW, 0, 0, inputW, inputH)
        val out = FloatArray(3 * inputW * inputH)
        val plane = inputW * inputH
        for (i in pixels.indices) {
            val px = pixels[i]
            out[i] = ((px shr 16 and 0xFF) / 255.0f)               // R plane
            out[plane + i] = ((px shr 8 and 0xFF) / 255.0f)        // G plane
            out[2 * plane + i] = ((px and 0xFF) / 255.0f)          // B plane
        }
        return out
    }
}