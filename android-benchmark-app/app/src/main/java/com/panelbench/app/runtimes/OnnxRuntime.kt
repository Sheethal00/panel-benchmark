package com.panelbench.app.runtimes

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtException
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

        data class InputSpec(val name: String, val shape: LongArray?, val elementCountHint: Long)

        val specs = s.inputNames.map { name ->
            val shape = (s.inputInfo[name]?.info as? TensorInfo)?.shape
            val hint = shape?.fold(1L) { acc, d -> acc * (if (d > 0) d else 1) } ?: 0L
            InputSpec(name, shape, hint)
        }

        // The image input is whichever has the largest declared element
        // count -- always far bigger than an auxiliary input like
        // scale_factor. Ties or fully-dynamic shapes default to the first.
        val imageSpec = specs.maxByOrNull { it.elementCountHint } ?: specs.first()

        // Determine the image tensor's ACTUAL channels/height/width from its
        // own declared shape (NCHW assumed) where static, falling back to
        // this config's values for any dynamic (-1) dims. Different OCR
        // models genuinely disagree here -- confirmed real via two separate
        // crashes: PaddleOCR's recognizer wants 3-channel RGB at height 48;
        // MMOCR's CRNN wants 1-channel grayscale at height 32 ("Got invalid
        // dimensions ... index 1 Got: 3 Expected: 1 ... index 2 Got: 48
        // Expected: 32"). Trusting each model's own declared shape instead
        // of assuming one fixed convention avoids hardcoding something that
        // just breaks again on the next model that disagrees with it.
        val imgShape = imageSpec.shape
        val channels = if (imgShape != null && imgShape.size == 4 && imgShape[1] > 0) imgShape[1].toInt() else 3
        val height = if (imgShape != null && imgShape.size == 4 && imgShape[2] > 0) imgShape[2].toInt() else inputH
        val width = if (imgShape != null && imgShape.size == 4 && imgShape[3] > 0) imgShape[3].toInt() else inputW

        val resized = Bitmap.createScaledBitmap(input, width, height, true)
        val floatData = bitmapToCHWFloatArray(resized, width, height, channels)
        val imageTensorShape = longArrayOf(1, channels.toLong(), height.toLong(), width.toLong())

        // Most models here have exactly one input (the image). Some (e.g.
        // PP-PicoDet, exported with Paddle's NMS baked in) have a second
        // input like "scale_factor" feeding that post-processing.
        //
        // session.inputNames/inputInfo were both observed to NOT reliably
        // list "scale_factor" as a required input for this export, even
        // though the compiled graph's execution genuinely requires it
        // (confirmed by a persistent real "Missing Input: scale_factor"
        // OrtException even after building the feed map from both of those
        // APIs). Rather than keep guessing at why ORT's Java metadata is
        // incomplete for this graph, handle it directly: build the feed map
        // from whatever inputNames DOES report, attempt the run, and if ORT
        // itself reports a specific missing input by name, add a fallback
        // tensor for exactly that name and retry once. This is correct
        // regardless of the underlying metadata quirk, and inert for every
        // single-input model where inputNames is already complete.
        val inputTensors = mutableMapOf<String, OnnxTensor>()
        try {
            for (spec in specs) {
                inputTensors[spec.name] = if (spec.name == imageSpec.name) {
                    OnnxTensor.createTensor(e, FloatBuffer.wrap(floatData), imageTensorShape)
                } else {
                    val shape = spec.shape
                    if (shape != null) {
                        val elementCount = shape.fold(1L) { acc, d -> acc * (if (d > 0) d else 1) }.toInt()
                        OnnxTensor.createTensor(e, FloatBuffer.wrap(FloatArray(elementCount) { 1.0f }), shape)
                    } else {
                        // Unknown shape for a name inputNames DID report -- default to
                        // the image tensor; extremely unlikely path given the logic above.
                        OnnxTensor.createTensor(e, FloatBuffer.wrap(floatData), imageTensorShape)
                    }
                }
            }

            return runWithMissingInputRetry(s, inputTensors)
        } finally {
            inputTensors.values.forEach { it.close() }
        }
    }

    /**
     * Runs the session with the given feeds. If ORT reports a specific named
     * input as missing (a real observed case: session.inputNames/inputInfo
     * don't list "scale_factor" for PP-PicoDet's export even though the graph
     * requires it), adds a fallback tensor for exactly that name and retries
     * once. The fallback value (1.0 in every element, shape [1, 2] unless the
     * name itself suggests otherwise) matches the one currently-known real
     * case -- extend this if a different missing input shows up in practice.
     */
    private fun runWithMissingInputRetry(
        s: OrtSession,
        inputTensors: MutableMap<String, OnnxTensor>
    ): InferenceOutput {
        val env = this.env ?: error("OnnxRuntime.load() must be called first")
        try {
            s.run(inputTensors).use { results ->
                return InferenceOutput(numDetections = 0)
            }
        } catch (ex: OrtException) {
            val missingName = Regex("Missing Input: (\\S+)")
                .find(ex.message ?: "")
                ?.groupValues?.get(1)
                ?: throw ex

            if (inputTensors.containsKey(missingName)) throw ex // already supplied, some other cause

            // Added to the same map the caller's `finally` already closes every
            // value of -- do NOT also close it here, or it gets double-closed.
            inputTensors[missingName] = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(floatArrayOf(1.0f, 1.0f)), longArrayOf(1, 2)
            )
            s.run(inputTensors).use { results ->
                return InferenceOutput(numDetections = 0)
            }
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

    private fun bitmapToCHWFloatArray(bitmap: Bitmap, width: Int, height: Int, channels: Int): FloatArray {
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        return when (channels) {
            1 -> FloatArray(width * height) { i ->
                val px = pixels[i]
                val r = (px shr 16 and 0xFF)
                val g = (px shr 8 and 0xFF)
                val b = (px and 0xFF)
                // Standard luminance weighting for RGB -> grayscale.
                (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f
            }
            else -> {
                val out = FloatArray(channels * width * height)
                val plane = width * height
                for (i in pixels.indices) {
                    val px = pixels[i]
                    out[i] = ((px shr 16 and 0xFF) / 255.0f)                    // R plane
                    if (channels > 1) out[plane + i] = ((px shr 8 and 0xFF) / 255.0f)      // G plane
                    if (channels > 2) out[2 * plane + i] = ((px and 0xFF) / 255.0f)        // B plane
                }
                out
            }
        }
    }
}