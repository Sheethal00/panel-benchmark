# panel-benchmark

Feasibility harness for benchmarking electrical-panel detector + OCR models on Android,
before committing to a model for the SE field-capture app.

Single Android app, config-driven: add a new candidate model by adding one entry to
`benchmark_config.json`, not by writing a new app.

## Target device profile

All benchmark decisions (which delegates to test, the memory budget, minSdk) are pinned
to this spec:

| Constraint | Value |
|---|---|
| Minimum OS | Android 9.0 (API 28) |
| Target device era | 2018 onwards (Snapdragon 845-equivalent) |
| Minimum RAM | 3 GB total system memory |
| AI hardware | CPU or GPU only -- no NPU/NNAPI dependency |
| **Total app memory budget** | **200 MB** (device becomes crash-prone above this on 3GB RAM) |

Practical implications baked into this repo:
- `minSdk = 28` in `app/build.gradle.kts`.
- `benchmark_config.json` defaults to `cpu` (XNNPACK) and `gpu` delegate variants per
  model, not NNAPI -- 2018-era NNAPI drivers are inconsistent (often silently fall back to
  CPU per-op) and the target spec doesn't require NPU acceleration anyway. The `ModelRuntime`
  interface still supports NNAPI if you want to test it as a bonus data point, but it's not
  in the default matrix.
- `report/generate_report.py` treats peak PSS during inference as a hard PASS/FAIL gate
  against the 200MB budget (`--budget-mb` to override), not just another metric in the table.
  **Note**: 200MB is the *total app* budget, not per-model -- if your production flow loads
  detector and OCR sequentially (not concurrently), each can approach the full 200MB as long
  as one is released before the next loads; if they're ever resident together, budget each
  against roughly half.
- Test on an actual Snapdragon 845-class device (e.g. a Pixel 3, Galaxy S9, or similar
  2018 device), not a current flagship or an emulator -- GPU delegate op coverage and
  thermal behavior on that chipset differ meaningfully from what you'd see on modern hardware.

## Repo layout

```
android-benchmark-app/   Android app that loads each model, times it, samples memory
conversion/              Scripts to export/quantize candidate models (TFLite / ONNX)
models/                  Converted model files (.tflite / .onnx), gitignored by default
test_data/               Sample panel images + ground-truth annotations for accuracy checks
scripts/                 Host-side ADB driver to run the full matrix unattended
report/                  Aggregates pulled results into one HTML/CSV comparison report
results/                 Raw JSON pulled from device, one folder per run
```

## Model files: hosted on Google Drive, not in this repo

The app's bundled model assets (`android-benchmark-app/app/src/main/assets/models/`)
are **not** tracked in git -- they're large binaries (dozens of `.tflite`/`.onnx` files
across 6+ architectures) that would otherwise bloat repo size and clone time.
`.gitignore` deliberately excludes both this folder and the top-level `models/`
conversion working directory.

**Before building the app for the first time (or after a fresh clone):**

1. Download the models from Google Drive: `[ADD SHARED DRIVE LINK HERE]`
2. Place them into `android-benchmark-app/app/src/main/assets/models/`, matching the
   existing `detector/` and `ocr/` subfolder structure (check `benchmark_config.json`'s
   `model_path` values if unsure where a specific file belongs).
3. Then proceed with the normal build: `cd android-benchmark-app && ./gradlew installDebug`.

If a model is missing from the Drive folder (e.g. after adding a new candidate
architecture), regenerate it with the matching `conversion/export_*.sh` script rather
than assuming it's just a stale copy -- see the Workflow section below for each script's
usage. Whoever regenerates a model this way should re-upload it to the shared Drive
folder so the rest of the team stays in sync, since there's no automated way (git or
otherwise) to detect that a model file is out of date here.

## Two-phase plan: feasibility first, fine-tune the winner

Every detector candidate benchmarked this week runs on stock COCO-pretrained weights, not
fine-tuned on panel photos. That's intentional, not a gap to fix immediately:

- **Phase 1 (this week, this repo's focus): feasibility only.** Latency, memory, and model
  size don't depend on what the model was trained to detect -- a COCO-pretrained
  YOLO26n/YOLO11n/YOLOv8n gives a legitimate signal for "can this architecture hit the
  200MB / real-time bar on Snapdragon-845-class hardware" even though it can't yet
  recognize MCBs or contactors. Treat the `report.html` latency/memory tables as the
  real output of this phase; treat detection accuracy as not-yet-meaningful.
- **Phase 2 (after a candidate is chosen): fine-tune, then re-check accuracy.** Once the
  report narrows the field to 1-2 architectures that clear the memory budget and hit
  acceptable latency, fine-tune only those on a labeled panel dataset, re-export with
  `conversion/export_yolo.sh <your-finetuned>.pt`, and re-run the isolated + pipeline
  benchmarks so the final numbers (latency, memory, *and* accuracy) all reflect the
  same real checkpoint you'd ship. `test_data/annotations.json` is the ground-truth
  schema to use for that accuracy pass -- it's premature to score mAP against it while
  every candidate still only knows COCO's 80 classes.

This keeps fine-tuning cost proportional to how many candidates actually survive the
cheap filter (latency + memory), instead of paying for labeled data and training runs
on architectures that were never going to fit the device budget anyway.

## Workflow

### 1. Get candidate models onto disk
Put/export your candidate detector and OCR models into `models/detector/` and `models/ocr/`.
See `conversion/export_yolov8.py` and `conversion/quantize_int8.py` for examples of
producing fp32 vs int8 variants -- quantization level is one of the main levers you'll
want to compare, so export both where possible.

`conversion/export_yolo.sh` exports any Ultralytics-compatible YOLO checkpoint (YOLO26n,
YOLOv8n, YOLO11n, or your own fine-tuned weights) into the 3 variants the default
`benchmark_config.json` expects: `yolo_416_dynamic.tflite`, `yolo_320_dynamic.tflite`,
`yolo_416_int8.tflite`, plus a `labels.txt`. Output filenames are architecture-agnostic,
so pointing it at a different checkpoint and re-running overwrites the same config slots
rather than requiring new entries -- handy for A/B'ing YOLO generations without editing
the config each time. Run it from inside `conversion/` with a venv active:
```
cd conversion
python3 -m venv panel-detector-venv && source panel-detector-venv/bin/activate
./export_yolo.sh yolo11n.pt        # or yolo26n.pt (default), or your own best.pt
```
A few things to know before trusting the output:
- `labels.txt` reflects whatever `$WEIGHTS` you pointed the script at -- COCO's 80 classes
  for any stock pretrained checkpoint, or your own panel device types if you export a
  fine-tuned checkpoint. A stock pretrained checkpoint's labels won't match panel device
  types (MCB, contactor, relay, etc.), so treat the resulting accuracy numbers as
  placeholders until you fine-tune.
- INT8 calibration uses Ultralytics' bundled 8-image `coco8` set as a placeholder -- swap
  in ~100-200 real panel crops once that data exists, for calibration that reflects your
  actual deployment images.
- If the ONNX export step fails on an unsupported op, bump `OPSET` inside the script from
  12 to 17+ and retry -- newer YOLO generations (e.g. YOLO26's end-to-end head) sometimes
  need ops later than opset 12 supports.
- Copy the outputs into the app's assets before building: `cp models/detector/*.tflite
  models/detector/labels.txt android-benchmark-app/app/src/main/assets/models/detector/`.
  `benchmark_config.json` already has matching entries (`yolo_416_dynamic_cpu`,
  `yolo_416_dynamic_gpu`, `yolo_320_dynamic_cpu`, `yolo_416_int8_cpu`, `yolo_416_int8_gpu`)
  plus 3 pipeline entries chaining them into your OCR candidates.

### 2. Register each candidate in the config
Edit `android-benchmark-app/app/src/main/assets/benchmark_config.json`. Each entry is one
(model, runtime, delegate) combination:

```json
{
  "name": "yolov8n_int8_nnapi",
  "task": "detector",
  "model_path": "models/detector/yolov8n_int8.tflite",
  "runtime": "tflite",
  "delegate": "nnapi",
  "input_width": 640,
  "input_height": 640,
  "warmup_runs": 10,
  "timed_runs": 50
}
```

Copy the actual model files into `android-benchmark-app/app/src/main/assets/models/...`
(matching `model_path`) before building -- they need to ship in the APK assets for this
skeleton. For larger models you'd rather not bundle, swap `copyAssetToFile` in the runtime
classes for a plain file read from `/sdcard/panel_benchmark_models/`.

### 3. Build and install
```
cd android-benchmark-app
./gradlew installDebug
```

### 4. Run the matrix
Either tap "Run Full Suite" in the app UI, or drive it unattended from your dev machine
(recommended once you have more than a couple of candidates):

```
cd scripts
python run_benchmark_matrix.py \
  --config ../android-benchmark-app/app/src/main/assets/benchmark_config.json \
  --out ../results/run_2026-09-16
```

This runs every isolated `models` config (force-stopping between each for a clean memory
baseline), then every `pipelines` entry (detector → OCR run within one continuous process
lifetime, deliberately *not* force-stopped between stages -- see below).

### 4b. Pipeline mode (sequential detector → OCR, real-world memory profile)
Since your production flow runs detector and OCR sequentially in the same process, an
isolated per-model benchmark can miss a real failure mode: `release()` on the detector's
native/delegate memory doesn't always return it to the OS immediately, so the OCR model
can load on top of memory that *looks* freed in Kotlin but isn't yet reclaimed at the OS
level. Two isolated "PASS" numbers can still add up to a pipeline that blows the 200MB
budget.

Add entries to the `pipelines` array in `benchmark_config.json`, referencing `models`
entries by name:

```json
{
  "name": "yolov8n_int8_to_paddleocr",
  "detector_config": "yolov8n_int8_cpu_xnnpack",
  "ocr_config": "paddleocr_mobile_cpu",
  "iterations": 5
}
```

Each pipeline run: loads the detector, warms up, runs `iterations` timed inferences,
releases it, samples memory, loads the OCR model, warms up, runs `iterations` timed
inferences, releases it -- all in one process lifetime. The result reports:
- `pss_peak_overall_mb` — the true worst case across the whole cycle; this is what to
  check against the 200MB budget, not either stage's isolated peak.
- `detector_mem_reclaimed_mb` — how much memory actually came back after the detector's
  `release()`, before the OCR model loaded. A small or negative value flags that native
  memory isn't being freed promptly, worth investigating even if the pipeline still passes.

### 5. Generate the report
```
cd report
pip install -r requirements.txt --break-system-packages
python generate_report.py --results-dir ../results/run_2026-09-16
```

Produces `report.html` with three sections: isolated detector models, isolated OCR models,
and sequential pipeline runs -- each isolated config and each pipeline gets a PASS/FAIL
badge against the 200MB memory budget (override with `--budget-mb`). Also writes
`report.csv` (isolated configs) and `report_pipelines.csv` (pipeline runs) for pivoting
in Excel/Sheets if you want to slice differently.

## What's measured per model
- Model file size on disk
- Cold load/init time
- Inference latency: p50 / p90 / p99 over N timed runs (after a warmup phase excluded from timing)
- Memory: PSS baseline, PSS after model load, peak PSS during inference
- Actual delegate used vs requested (NNAPI/GPU silently fall back to CPU on unsupported
  devices/ops -- always check `actual_delegate_info` in results, not just what you configured)
- Device model / SoC / Android version the run happened on

## What this skeleton does NOT do yet (fill in per your models)
- **Output post-processing**: `TFLiteRuntime.runInference` / `OnnxRuntime.runInference`
  bind inputs and run the graph, but detection-box decoding (NMS, anchor decoding) and
  OCR sequence decoding (CTC/attention) are model-family-specific -- wire those up once
  you've picked real candidate architectures.
- **Accuracy scoring**: `test_data/annotations.json` gives you a ground-truth schema;
  add a scoring pass (mAP for detector, CER for OCR) once post-processing is in place,
  so the report can show accuracy alongside latency/memory rather than just performance.
- **ML Kit runtime binding**: config schema and runtime factory already have a slot for
  `"runtime": "mlkit"` as a zero-integration-cost OCR baseline; implement `MlKitOcrRuntime`
  in `runtimes/` following the same `ModelRuntime` interface.

## Notes
- Benchmark on physical devices, not emulators -- NNAPI/GPU delegate behavior and available
  accelerators differ meaningfully from what an emulator reports.
- Run each config after a force-stop (the driver script does this) so one candidate's
  leftover memory/thermal state doesn't bias the next one's numbers.
- If field devices run hot (SEs working outdoors), consider a sustained-load variant
  (loop timed_runs to 200+) to see thermal throttling effects on later iterations, not
  just cold/warm best-case latency.