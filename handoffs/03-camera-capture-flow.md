# HANDOFF: Real camera capture flow

## PROJECT CONTEXT
PanelBenchmark: Android app for site engineers -- photograph an electrical
panel, detect devices, OCR their ratings, output a CSV. Weeks of
benchmarking settled on YOLOX-Nano (detector) + ML Kit (OCR).

## THE GAP
The entire app, throughout all of benchmarking, has only ever used ONE
static test image (`assets/sample_panel.jpg`, loaded via
`loadSampleImage()` in `BenchmarkRunner.kt`). There is no real camera
capture flow at all. `androidx.camera:camera-core` /
`camera-camera2` / `camera-lifecycle` (CameraX) are already declared as
dependencies in `app/build.gradle.kts` (comment there says "only needed
if you benchmark end-to-end with live camera capture") but never actually
wired into any UI or capture logic.

## CONTEXT ON CURRENT APP STRUCTURE
The app just got a real interactive UI (Jetpack Compose, Material3) for
picking and running individual benchmark configs -- see
`MainActivity.kt`, `BenchmarkViewModel.kt`, `BenchmarkScreen.kt`. This is
still fundamentally a BENCHMARKING harness (pick a model config, run it,
see latency/memory), not the real field app. This task is about adding
actual camera capture as a genuinely new capability, most likely as
either:
(a) a new screen/flow in the same app, separate from the benchmark
    picker UI, or
(b) the start of a genuinely separate "real app" module/build target,
    reusing the runtime/model-loading code but with a different UI
    entirely.
Worth deciding which approach fits before diving in -- (a) is faster to
get something working, (b) is cleaner long-term if the benchmark harness
and the real field app are meant to diverge significantly.

## WHAT'S NEEDED
1. Camera permission handling (runtime permission request, not just a
   manifest entry).
2. A capture screen: live preview (CameraX `Preview` use case) + a
   capture button, or a simpler "pick from gallery" fallback if live
   preview isn't the priority yet.
3. Getting the captured image into the same `Bitmap` format the existing
   `ModelRuntime.runInference(bitmap: Bitmap)` interface expects (check
   `ModelRuntime.kt` for the exact interface) -- CameraX's `ImageCapture`
   use case with `takePicture()` typically gives a JPEG file or an
   `ImageProxy`; needs converting to `Bitmap` the same way
   `loadSampleImage()` does for the static test asset.
4. Image orientation handling -- a real-world consideration
   `sample_panel.jpg` never had to deal with (phones report EXIF
   orientation; captured images often need rotation correction before
   feeding to the detector, since panel photos won't always be taken
   perfectly upright).
5. Reasonable image sizing/compression before feeding to inference --
   full-resolution phone camera photos (often 12MP+) are much larger than
   any of the fixed input sizes (320x320 to 640x640) used throughout
   benchmarking; decide whether resizing happens before or as part of
   the existing `bitmapToCHWFloatArray`-style preprocessing already in
   `OnnxRuntime.kt`/`TFLiteRuntime.kt`.

## RELEVANT FILES
- `app/build.gradle.kts` -- CameraX deps already present (version
  `1.3.4` at last check -- verify still current when picking this up).
- `BenchmarkRunner.kt` -- `loadSampleImage()` shows the existing
  Bitmap-loading pattern to match for consistency.
- `runtimes/ModelRuntime.kt` -- the interface every runtime implements;
  confirms the exact `Bitmap` contract capture needs to satisfy.

## WHAT "DONE" LOOKS LIKE
A real screen where the user can point the camera at something, capture a
photo, and get a `Bitmap` that successfully feeds into
`runInference()` -- end to end, on a real device, not just the static
test asset.
