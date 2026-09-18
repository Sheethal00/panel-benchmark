# HANDOFF: End-to-end pipeline + CSV export (the actual product)

## PROJECT CONTEXT
PanelBenchmark: Android app for site engineers. The ORIGINAL goal, from
week one: **photo of electrical panel -> detector finds devices -> OCR
reads each device's rating -> CSV output**. Everything built so far
(6 detector architectures benchmarked, 4 OCR candidates benchmarked,
YOLOX-Nano + ML Kit chosen as winners) has been FEASIBILITY work --
proving the pieces can run fast enough and fit in memory. This task is
building the actual thing: real orchestration logic connecting detection
output to OCR input to a structured file.

## DEPENDENCIES -- READ FIRST
This task depends on BOTH of these being done first (separate handoffs):
- `01-detector-decode-nms.md` -- need real bounding boxes from YOLOX,
  not the current `numDetections = 0` placeholder.
- `03-camera-capture-flow.md` -- need a real captured image, not just
  the static `sample_panel.jpg` (though this task CAN be built/tested
  against the static image first, then swapped for real camera input
  once that's ready -- doesn't strictly have to wait).

Accuracy validation (`02-accuracy-validation.md`) is not a hard
dependency but its results should inform this work -- if detector/OCR
accuracy turns out poor, the orchestration logic built here (e.g.
confidence thresholds, how aggressively to crop, whether to allow manual
correction) will need adjusting anyway.

## WHAT'S NEEDED
1. **Per-detection cropping.** For each bounding box YOLOX outputs, crop
   that region out of the full panel photo (with some padding margin --
   OCR generally does better with a bit of context around tight text,
   not a pixel-perfect crop to the box edges).
2. **Per-crop OCR.** Run ML Kit's `MlKitOcrRuntime` against each cropped
   region individually -- NOT the whole original photo. This is
   different from how benchmarking worked (one static image straight
   into OCR) -- now it's N detections -> N separate OCR calls.
3. **Result association.** Each OCR result needs to stay linked to which
   detected device it came from (device class/type from YOLOX,
   confidence, bounding box location) -- not just a flat list of
   recognized strings.
4. **CSV structure.** Design the actual output schema -- likely something
   like: device_id, device_class, confidence, bounding_box_coords,
   recognized_text, ocr_confidence, timestamp. Get clarity on exactly
   what fields the field engineers actually need downstream before
   finalizing columns.
5. **Error/edge-case handling.** What happens when: zero detections found
   (empty panel photo, bad angle)? OCR returns empty string for a
   detected region (blurry, obscured label)? Multiple devices overlap in
   the frame? These need sane, non-crashing behavior, not just a happy
   path.
6. **File output + sharing.** Where does the CSV get written
   (`getExternalFilesDir`, matching the existing results-writing pattern
   in `ResultWriter.kt`)? Does the user need a way to view/export/share
   it from the app (email, save to Drive, etc.), or is on-device storage
   enough for now?

## RELEVANT FILES
- `runtimes/OnnxRuntime.kt`, `runtimes/MlKitOcrRuntime.kt` -- the actual
  inference calls this orchestration will chain together.
- `metrics/ResultWriter.kt` -- existing pattern for writing structured
  output to `getExternalFilesDir`; CSV writing should probably live
  alongside this, following the same conventions (file location, naming).
- `BenchmarkRunner.kt` -- shows the existing load -> infer -> release
  lifecycle pattern for both detector and OCR runtimes; this task's
  orchestration will look structurally similar but with real
  crop-and-chain logic instead of a flat timing loop.

## WHAT "DONE" LOOKS LIKE
Take a real (or realistic test) panel photo all the way through: capture
(or load) -> YOLOX detects devices -> each device gets cropped and OCR'd
-> a real CSV file is written with device+text data, associated
correctly -- verified by actually opening the CSV and checking it makes
sense against the source photo.
