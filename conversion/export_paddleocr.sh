#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Exports PaddleOCR's English mobile models (PaddlePaddle/PaddleOCR,
# Apache-2.0) to TFLite + ONNX: detection (en_PP-OCRv3_det, finds text
# regions) and recognition (en_PP-OCRv4_rec, reads characters from a cropped
# text line). Real OCR is two models chained together, not one -- unlike
# ML Kit, which bundles both internally as a black box.
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
# - INT8 quantization is skipped here for the first pass, matching the
#   caution learned from PicoDet/RTMDet: models with non-trivial output
#   post-processing have shown real INT8 kernel/hang issues on this device.
#   Add it later if the numbers below look promising enough to justify the
#   extra validation effort.
# - The DETECTION model's "*_dynamic.tflite" is actually plain float32, not
#   truly dynamic-range quantized -- confirmed real incompatibility between
#   dynamic-range's mixed int8-weight/float32-activation scheme and
#   TFLite's TRANSPOSE_CONV kernel (used by this model's upsampling
#   decoder). The recognition model's dynamic-range export is unaffected
#   (different architecture, no transpose convs in its path).
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
# tensorflow pinned to 2.17.0 -- MUST match the Android app's
# org.tensorflow:tensorflow-lite AAR version (app/build.gradle.kts). Started
# at 2.16.1 (matching the app at the time), but that pin alone didn't
# resolve a real "Didn't find op for builtin opcode 'FULLY_CONNECTED'
# version '12'" crash even with both sides confirmed at 2.16.1 -- the
# Python tensorflow package and the Android AAR are apparently built from
# slightly different points in TF's release branches even at matching
# version tags. Bumped both sides to 2.17.0 (the last release before
# org.tensorflow:tensorflow-lite was renamed/relocated to
# com.google.ai.edge.litert) to try the other direction. If this still
# doesn't resolve it, the AAR may need migrating to the new litert artifact
# instead of chasing version numbers within the deprecated one further.
pip install "paddle2onnx==1.3.1" onnx onnx_graphsurgeon sng4onnx onnx2tf "tensorflow==2.17.0" tf_keras onnxruntime psutil
# Pin protobuf from the start this time -- no MMDeploy-style tool here with
# a conflicting older-protobuf requirement (unlike export_rtmdet.sh, where
# this pin has to be delayed until after MMDeploy's own install).
pip install --upgrade "protobuf>=5.28,<6"

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
echo "wrote $OUT_DIR/paddleocr_det.onnx (for direct ONNX Runtime benchmarking)"

echo "== Converting detection model: ONNX -> SavedModel =="
python3 -c "
import numpy as np
arr = np.random.rand(20, 128, 128, 3).astype(np.float32)
np.save('calibration_image_sample_data_20x128x128x3_float32.npy', arr)
"
# -osd/--disable_group_convolution: same fixes learned from PicoDet --
# signature defs for TFLiteConverter, and MobileNetV3-backbone depthwise
# convs hitting the SavedModel/GroupConvolution incompatibility.
# -ois x:1,3,640,640: this model's ONNX input is fully dynamic (not just
# batch), confirmed via a real onnx2tf failure -- -b 1 alone only fixes the
# batch dimension, not height/width. -ois (--overwrite_input_shape) pins the
# whole shape explicitly; "x" is this model's actual input tensor name.
onnx2tf -i paddleocr_det.onnx -o "$OUT_DIR/paddleocr_det_sm" \
    -osd -ois x:1,3,640,640 --disable_group_convolution

if [ ! -f "$OUT_DIR/paddleocr_det_sm/saved_model.pb" ]; then
    echo "ERROR: onnx2tf did not produce a SavedModel for the detection model -- scroll up for its actual output/error." >&2
    exit 1
fi

echo "== Using onnx2tf's own directly-emitted float32.tflite (not our own re-conversion) =="
# First attempt re-converted the SavedModel ourselves with no optimizations,
# expecting pure float32 -- but hit the EXACT SAME "weights->type !=
# input->type (INT8 != FLOAT32)" error again, meaning the mixed-precision
# weights aren't coming from our TFLiteConverter step at all. onnx2tf
# itself auto-generates its own float32.tflite directly as part of its
# default output (confirmed in earlier debugging: "Float32 tflite output
# complete!" in its own log) -- using that file directly, instead of
# running a second TFLiteConverter pass on top of the SavedModel, avoids
# whatever onnx2tf-internal step was introducing the mixed types.
DET_FLOAT32_SRC=$(find "$OUT_DIR/paddleocr_det_sm" -iname "*float32.tflite" | head -n 1)
if [ -z "$DET_FLOAT32_SRC" ]; then
    echo "ERROR: onnx2tf did not emit a float32.tflite for the detection model. Contents of $OUT_DIR/paddleocr_det_sm:" >&2
    ls -la "$OUT_DIR/paddleocr_det_sm" >&2
    exit 1
fi
cp "$DET_FLOAT32_SRC" "$OUT_DIR/paddleocr_det_dynamic.tflite"
echo "wrote $OUT_DIR/paddleocr_det_dynamic.tflite (from $DET_FLOAT32_SRC -- plain float32, not actually dynamic-range quantized)"

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
echo "wrote $OUT_DIR/paddleocr_rec.onnx (for direct ONNX Runtime benchmarking)"

echo "== Converting recognition model: ONNX -> SavedModel =="
# -ois: same fix as the detection model above -- fully dynamic input,
# pinned explicitly (height 48, width 320, matching this model's fixed
# benchmarking input size noted at the top of this script).
onnx2tf -i paddleocr_rec.onnx -o "$OUT_DIR/paddleocr_rec_sm" \
    -osd -ois x:1,3,48,320 --disable_group_convolution

if [ ! -f "$OUT_DIR/paddleocr_rec_sm/saved_model.pb" ]; then
    echo "ERROR: onnx2tf did not produce a SavedModel for the recognition model -- scroll up for its actual output/error." >&2
    exit 1
fi

echo "== Quantizing recognition model to TFLite dynamic range =="
python3 <<PYEOF
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("$OUT_DIR/paddleocr_rec_sm")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("$OUT_DIR/paddleocr_rec_dynamic.tflite", "wb") as f:
    f.write(tflite_model)
print("wrote $OUT_DIR/paddleocr_rec_dynamic.tflite")
PYEOF

echo ""
echo "== Done. Exported: =="
ls -la "$OUT_DIR"/paddleocr_*.tflite "$OUT_DIR"/paddleocr_*.onnx "$OUT_DIR"/en_dict.txt 2>/dev/null

echo ""
echo "Next:"
echo "  1. Copy into the Android app's assets:"
echo "     cp $OUT_DIR/paddleocr_det_dynamic.tflite $OUT_DIR/paddleocr_rec_dynamic.tflite \\"
echo "        $OUT_DIR/paddleocr_det.onnx $OUT_DIR/paddleocr_rec.onnx $OUT_DIR/en_dict.txt \\"
echo "        ../android-benchmark-app/app/src/main/assets/models/ocr/"
echo "  2. Add benchmark_config.json entries -- TWO configs (det + rec), since"
echo "     this is a two-stage pipeline unlike ML Kit's single black-box call."
echo "     Detection: input_width/height 640 (fixed for benchmarking; real"
echo "     deployment resizes per input image)."
echo "     Recognition: input a fixed-width crop (e.g. 320x48) -- real"
echo "     deployment resizes per detected text box's aspect ratio."
echo "  3. Remember: detection outputs a probability MAP (needs threshold +"
echo "     contour-finding for boxes), recognition outputs CTC sequences"
echo "     (needs en_dict.txt-based decode) -- neither produces readable"
echo "     output without that post-processing, same placeholder status as"
echo "     every detector's output parsing in this repo."