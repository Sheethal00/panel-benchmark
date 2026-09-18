# HANDOFF: Accuracy validation (detector + OCR)

## PROJECT CONTEXT
PanelBenchmark: Android app for site engineers -- photograph an electrical
panel, detect devices, OCR their ratings, output a CSV. Chosen candidates:
- **Detector: YOLOX-Nano** (Apache-2.0, COCO-pretrained; being fine-tuned on
  ~15 annotated panel photos in a separate conversation)
- **OCR: ML Kit** (~145ms combined latency, clear winner over PaddleOCR,
  MMOCR, and RapidOCR on both speed and the 200MB memory budget)

## THE GAP
**Everything measured in this whole project so far has been latency and
memory feasibility -- never actual correctness.** No mAP number exists for
YOLOX on panel-device detection. No OCR accuracy number exists for reading
actual panel labels (current ratings, voltage, breaker numbers, etc.). We
do not actually know whether this detector+OCR combination WORKS, only
that it's fast and fits the memory budget. This is arguably the most
important gap before treating any of this as validated.

Depends on: detector decode/NMS being wired up first (separate handoff,
`01-detector-decode-nms.md`) -- can't measure detection accuracy without
real bounding box output.

## TWO SEPARATE ACCURACY QUESTIONS

### 1. Detector accuracy (YOLOX on panel devices)
- COCO-pretrained YOLOX has never seen an electrical panel -- its
  COCO-80 classes are irrelevant to this domain. Accuracy against real
  panel photos will likely be poor until the fine-tuning (separate
  conversation) completes.
- Once fine-tuned weights exist: needs a held-out validation set (the
  ~15-image dataset is tiny -- consider whether fine-tuning used all 15
  for training, meaning a genuinely separate validation set may not
  exist yet and needs collecting).
- Standard metric: mAP@0.5 (or similar) via a real eval script -- COCO-style
  eval tooling exists in most detection frameworks; may be easiest to
  export detections in COCO JSON format and reuse a standard evaluator
  rather than hand-rolling mAP calculation.

### 2. OCR accuracy (ML Kit on panel label text)
- ML Kit's accuracy on printed panel labels (often stamped/engraved metal
  or printed stickers, sometimes at odd angles or partially obscured) is
  unknown -- benchmarked only for timing, using one static test image
  the whole time (`sample_panel.jpg`), never checked for whether the
  recognized text was actually CORRECT.
- Needs: a set of real panel photos with ground-truth transcriptions of
  what the labels actually say (current ratings like "32A", voltage like
  "415V", breaker/circuit numbers, etc.), then character/word-level
  accuracy comparison against ML Kit's actual output.
- Real-world panel labels are a genuinely different domain from ML Kit's
  general-purpose training data (arbitrary photographed text) -- don't
  assume good performance carries over; this needs real measurement.

## RELEVANT FILES / CONTEXT
- `test_data/` directory exists in the repo layout (per README) for
  "sample panel images + ground-truth annotations for accuracy checks" --
  check whether anything real is already there or if it's still a
  placeholder.
- `android-benchmark-app/app/src/main/java/com/panelbench/app/runtimes/MlKitOcrRuntime.kt`
  -- confirm this actually surfaces recognized TEXT (not just timing) from
  ML Kit's API; the benchmarking harness may only be reading latency and
  discarding the actual recognition result.
- Fine-tuning conversation (YOLOX) will produce the trained weights this
  work depends on for detector accuracy specifically.

## SUGGESTED APPROACH
1. Collect/confirm a real validation image set with ground truth (both
   bounding boxes for devices AND transcribed text for labels) --
   probably needs actual site photos, not just the one sample image used
   throughout benchmarking.
2. Detector: run fine-tuned YOLOX against validation images, compute
   mAP or at minimum a simpler precision/recall count if full mAP
   tooling is overkill for a dataset this size.
3. OCR: run ML Kit against cropped label regions (once detector output
   feeds real crops -- see pipeline handoff `04-...`), compare recognized
   text against ground truth transcriptions.
4. Document results plainly -- if either number is bad, that changes
   next steps (more fine-tuning data, a different OCR post-processing
   strategy, etc.) before investing further in productization.

## WHAT "DONE" LOOKS LIKE
A real, documented accuracy number (not just latency) for both the
detector and OCR stages, against real panel photos with ground truth --
even if the number itself isn't great yet, having it measured is the goal.
