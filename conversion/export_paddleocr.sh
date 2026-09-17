#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Exports PaddleOCR's English mobile models (PaddlePaddle/PaddleOCR,
# Apache-2.0) to ONNX: detection (en_PP-OCRv3_det, finds text regions) and
# recognition (en_PP-OCRv4_rec, reads characters from a cropped text line).
# Real OCR is two models chained together, not one -- unlike ML Kit, which
# bundles both internally as a black box.
#
# Usage:
#   ./export_paddleocr.sh
#
# This script creates and activates its OWN dedicated venv
# (panel-paddleocr-venv), same reasoning as export_picodet.sh: keeps this
# genuinely different, version-sensitive dependency stack isolated.
#
# Much simpler than export_picodet.sh/export_rtmdet.sh: PaddleOCR's model
# zoo distributes these as ALREADY-EXPORTED inference tars (inference.pdmodel
# + inference.pdiparams), so there's no export_model.py step and no
# PaddleOCR repo clone needed -- straight from download to paddle2onnx.
#
# ONNX ONLY -- NO TFLITE OUTPUT. This was a real, multi-round debugging
# effort, not an oversight:
# - Detection's dynamic-range TFLite hit a genuine TFLite kernel gap
#   (TRANSPOSE_CONV doesn't support mixed int8-weight/float32-activation).
# - Switching to onnx2tf's own plain float32.tflite output didn't help --
#   confirmed via direct tensor inspection that the file had zero int8
#   tensors, yet the app still failed identically, meaning the issue wasn't
#   in the file at all.
# - Recognition hit a SEPARATE "FULLY_CONNECTED version 12" op mismatch
#   between the Python tensorflow converter and the app's TFLite AAR, even
#   at matching version tags on both sides.
# - Bumping the app's TFLite AAR past 2.16.1 to chase op compatibility
#   triggered a real build-breaking duplicate-class conflict: Maven
#   relocates org.tensorflow:tensorflow-lite -> com.google.ai.edge.litert
#   at 2.17.0, colliding with ML Kit's own internally-bundled
#   tensorflow-lite-api:2.13.0.
# - Pinning the Python converter down to 2.13.0 (matching that floor) then
#   broke on an unrelated ml_dtypes/onnx2tf incompatibility -- TF 2.13.0 is
#   old enough that the modern onnx/onnx2tf/ml_dtypes stack no longer
#   cleanly supports it.
# ONNX Runtime has none of this fragility (already confirmed working:
# paddleocr_det_onnx_cpu ran cleanly with real numbers on the first try),
# so it's the only path here now. If TFLite is wanted again later, it needs
# a genuinely different approach -- not another version-pin guess.
#
# IMPORTANT, read before wiring up real inference (not just benchmarking):
# - Detection model output is a probability MAP (DB algorithm), not boxes
#   directly -- needs thresholding + contour-finding post-processing to get
#   actual text region bounding boxes. Different from every detector in this
#   repo, which output boxes (or box-like tensors) directly.
# - Recognition model output is a CTC character-probability sequence over
#   time steps -- needs CTC decoding with the character dictionary (saved as
#   en_dict.txt alongside the models) to get actual text. Not usable as
#   readable text without that decode step implemented.
# - Both models take DYNAMIC input sizes in real deployment (detection scales
#   to the input image's aspect ratio; recognition crops get resized per
#   detected box). Fixed sizes are used below (640x640 for detection, a
#   320-wide crop for recognition) purely for consistent benchmarking --
#   real accuracy will differ from whatever a production pipeline's adaptive
#   resizing would produce.
# ---------------------------------------------------------------------------

OUT_DIR="../models/ocr"
mkdir -p "$OUT_DIR"

VENV_DIR="./panel-paddleocr-venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "== Creating dedicated venv for PaddleOCR export: $VENV_DIR =="
    python3 -m venv "$VENV_DIR"
fi
echo "== Activating $VENV_DIR =="
source "$VENV_DIR/bin/activate"

echo "== Installing dependencies =="
pip install "paddlepaddle==2.6.2"
# paddle2onnx pinned to 1.3.1 -- confirmed working version against
# paddlepaddle 2.6.2 from the PicoDet export (2.1.0, whatever pip resolves
# by default, is incompatible).
# NOTE: no tensorflow/onnx2tf/tf_keras here -- this script is ONNX-only now
# (see the top-of-file note on why the TFLite conversion path was removed
# entirely after a real, multi-round dead end chasing TFLite AAR/converter
# version compatibility). onnxruntime is kept for validating the ONNX
# output loads correctly; psutil is needed by paddle2onnx/paddle itself.
pip install "paddle2onnx==1.3.1" onnx onnxruntime psutil

echo "== Downloading PaddleOCR English mobile models (official release) =="
DET_URL="https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar"
REC_URL="https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar"
DICT_URL="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/en_dict.txt"

if [ ! -f en_PP-OCRv3_det_infer.tar ]; then
    curl -fL -o en_PP-OCRv3_det_infer.tar "$DET_URL"
fi
if [ ! -f en_PP-OCRv4_rec_infer.tar ]; then
    curl -fL -o en_PP-OCRv4_rec_infer.tar "$REC_URL"
fi
if [ ! -s en_PP-OCRv3_det_infer.tar ] || [ ! -s en_PP-OCRv4_rec_infer.tar ]; then
    echo "ERROR: one or both model tars missing/empty after download. Check the URL/network and retry." >&2
    exit 1
fi

tar xf en_PP-OCRv3_det_infer.tar
tar xf en_PP-OCRv4_rec_infer.tar

curl -fL -o "$OUT_DIR/en_dict.txt" "$DICT_URL" || \
    echo "WARNING: could not fetch en_dict.txt -- needed later for CTC decode, not for this export itself."

# Detection and recognition tars extract to differently-named directories
# depending on PaddleOCR release -- find them robustly rather than hardcode.
DET_DIR=$(find . -maxdepth 1 -type d -iname "*det_infer*" | head -n 1)
REC_DIR=$(find . -maxdepth 1 -type d -iname "*rec_infer*" | head -n 1)
if [ -z "$DET_DIR" ] || [ -z "$REC_DIR" ]; then
    echo "ERROR: could not locate extracted det/rec directories. Contents of CWD:" >&2
    ls -la >&2
    exit 1
fi
echo "Detection dir: $DET_DIR"
echo "Recognition dir: $REC_DIR"

# ---------------------------------------------------------------------------
# Detection model
# ---------------------------------------------------------------------------
echo "== Converting detection model: Paddle inference -> ONNX =="
paddle2onnx --model_dir "$DET_DIR" \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --opset_version 11 \
    --save_file paddleocr_det.onnx

if [ ! -s paddleocr_det.onnx ]; then
    echo "ERROR: paddle2onnx did not produce paddleocr_det.onnx -- scroll up for its actual output/error." >&2
    exit 1
fi
cp paddleocr_det.onnx "$OUT_DIR/paddleocr_det.onnx"
echo "wrote $OUT_DIR/paddleocr_det.onnx (for ONNX Runtime benchmarking -- see top-of-file note on why there's no TFLite output)"

# ---------------------------------------------------------------------------
# Recognition model
# ---------------------------------------------------------------------------
echo "== Converting recognition model: Paddle inference -> ONNX =="
paddle2onnx --model_dir "$REC_DIR" \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --opset_version 11 \
    --save_file paddleocr_rec.onnx

if [ ! -s paddleocr_rec.onnx ]; then
    echo "ERROR: paddle2onnx did not produce paddleocr_rec.onnx -- scroll up for its actual output/error." >&2
    exit 1
fi
cp paddleocr_rec.onnx "$OUT_DIR/paddleocr_rec.onnx"
echo "wrote $OUT_DIR/paddleocr_rec.onnx (for ONNX Runtime benchmarking)"

echo ""
echo "== Done. Exported: =="
ls -la "$OUT_DIR"/paddleocr_*.onnx "$OUT_DIR"/en_dict.txt 2>/dev/null

echo ""
echo "Next:"
echo "  1. Copy into the Android app's assets:"
echo "     cp $OUT_DIR/paddleocr_det.onnx $OUT_DIR/paddleocr_rec.onnx $OUT_DIR/en_dict.txt \\"
echo "        ../android-benchmark-app/app/src/main/assets/models/ocr/"
echo "  2. benchmark_config.json entries: paddleocr_det_onnx_cpu, paddleocr_rec_onnx_cpu"
echo "     (runtime: onnx) -- TFLite variants were removed after a real, multi-round"
echo "     debugging dead end, see the note at the top of this script."
echo "  3. Remember: detection outputs a probability MAP (needs threshold +"
echo "     contour-finding for boxes), recognition outputs CTC sequences"
echo "     (needs en_dict.txt-based decode) -- neither produces readable"
echo "     output without that post-processing, same placeholder status as"
echo "     every detector's output parsing in this repo."