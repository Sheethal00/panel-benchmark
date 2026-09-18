# HANDOFF: Wire up YOLOX output decoding + NMS

## PROJECT CONTEXT
PanelBenchmark: Android app for site engineers -- photograph an electrical
panel, detect devices, OCR their ratings, output a CSV. Weeks of benchmarking
across 6 detector architectures + 4 OCR candidates settled on:
- **Detector: YOLOX-Nano** (Apache-2.0; won on license + latency, ~36.7ms p50
  ONNX Runtime CPU, verified stable across repeat runs)
- **OCR: ML Kit** (~145ms combined, far ahead of every open-source
  alternative on both latency and the 200MB memory budget)

A separate conversation is fine-tuning YOLOX on ~15 annotated panel photos
(Roboflow-exported labels, MLflow tracking). This task is independent of
that and can proceed against the COCO-pretrained weights already exported.

## THE GAP
Every detector in this repo -- across all 6 architectures benchmarked --
currently returns `InferenceOutput(numDetections = 0)` from
`runInference()`. The whole benchmarking effort only ever measured raw
model *inference speed*, never decoded real bounding boxes. This is THE
hard blocker: nothing downstream (accuracy validation, the real app
pipeline) can proceed until this exists for YOLOX specifically.

## WHAT YOLOX'S OUTPUT ACTUALLY LOOKS LIKE
- YOLOX is **NOT NMS-free** (unlike YOLO26, which was dropped anyway for
  AGPL licensing). Raw output needs separate NMS post-processing.
- Export used `--decode_in_inference` (see
  `conversion/export_yolox.sh`), meaning box decoding (converting raw
  grid-cell predictions to actual x/y/w/h) already happens INSIDE the
  exported model graph -- what comes out is a `[1, N, 5+num_classes]`
  tensor (or similar; verify exact shape against the actual exported
  model) with per-anchor `[x, y, w, h, objectness, class_scores...]`,
  NOT raw grid logits needing manual decode math.
- What's still needed on the Android side: **NMS** (filter overlapping
  boxes above a confidence threshold, keep highest-scoring per cluster) --
  this is NOT baked into the export.
- Fixed input resolution: 416x416 (confirm against whichever exact config
  was benchmarked -- check `benchmark_config.json` for
  `yolox_nano_onnx_cpu`'s `input_width`/`input_height`).

## RELEVANT FILES
- `android-benchmark-app/app/src/main/java/com/panelbench/app/runtimes/OnnxRuntime.kt`
  -- `runInference()` currently returns the hardcoded empty
  `InferenceOutput`. This is where real decode + NMS needs to go, OR
  factor it into a new `YoloxPostProcessor.kt` that `OnnxRuntime.kt` calls
  into (recommended -- keeps runtime-agnostic code separate from
  model-specific decode logic, since other architectures will need their
  own decoders eventually too).
- `models/detector/yolox_nano_labels.txt` -- currently placeholder COCO-80
  classes. Real panel-device class names will come from the fine-tuning
  effort in the other conversation; until then this can stay as-is for
  structural/testing purposes.
- `conversion/export_yolox.sh` -- the export script; re-read its
  `--decode_in_inference` note before assuming raw output shape.

## SUGGESTED APPROACH
1. Dump the actual output tensor from a real inference run (log shape +
   raw values for a handful of anchors) to confirm the true output format
   empirically -- don't assume from memory, verify against this specific
   export.
2. Implement confidence thresholding (drop low-objectness anchors) +
   standard NMS (IoU threshold, greedy suppression) in Kotlin.
3. Update `InferenceOutput` (check its current definition -- likely in
   `ModelRuntime.kt` or wherever `InferenceOutput` is declared) to
   actually carry box coordinates + class + confidence, not just a count.
4. Test against `sample_panel.jpg` (or better, a real annotated panel photo
   if the fine-tuning conversation has produced one) and sanity-check the
   boxes visually make sense (e.g. dump to a debug overlay image, or log
   coordinates and manually verify against the image dimensions).

## WHAT "DONE" LOOKS LIKE
`runInference()` for the YOLOX ONNX config returns real, NMS-filtered
bounding boxes with class + confidence, verified sane against at least one
real test image -- not just "compiles and returns a count."
