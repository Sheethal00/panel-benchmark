#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Exports NanoDet-Plus-m (RangiLyu/nanodet, Apache-2.0) at 320 and 416 to
# TFLite: a dynamic-range variant and an int8 variant, for each resolution.
#
# Unlike export_yolo.sh / export_yolox.sh, this does NOT need PyTorch, a
# cloned repo, or a checkpoint export step -- RangiLyu's own GitHub release
# already ships pre-exported ONNX files directly, so we start from those.
# Much less fragile: no torch/torchvision version dance, no legacy export
# API compatibility shims.
#
# Usage:
#   ./export_nanodet.sh              # nanodet-plus-m (1.0x width)
#   ./export_nanodet.sh 1.5x         # nanodet-plus-m-1.5x (wider, more accurate, bigger)
#
# Run this INSIDE an activated venv:
#   python3 -m venv panel-detector-venv && source panel-detector-venv/bin/activate
#   ./export_nanodet.sh
#
# IMPORTANT -- read before wiring up real inference (not just benchmarking):
# NanoDet is NOT NMS-free (like YOLOX, unlike YOLO26) -- a separate NMS pass
# is required in app code after parsing this model's output.
#
# NanoDet's own preprocessing uses per-channel mean/std normalization
# (approximately mean=[103.53,116.28,123.675], std=[57.375,57.12,58.395] on
# a 0-255 input), NOT a simple divide-by-255 like the YOLO family. The
# TFLiteRuntime.kt bitmapToByteBuffer() in this repo currently does /255
# normalization for every model -- fine for latency/memory numbers this
# week (the model still runs real compute either way), but WRONG for actual
# detection accuracy once you wire up real output decoding. Match NanoDet's
# specific mean/std before trusting any detections from it.
# ---------------------------------------------------------------------------

VARIANT="${1:-}"   # "" (1.0x, default) or "1.5x"
TAG="v1.0.0-alpha-1"
OUT_DIR="../models/detector"
mkdir -p "$OUT_DIR"

if [ -z "$VARIANT" ]; then
    MODEL_PREFIX="nanodet-plus-m"
    OUT_PREFIX="nanodet_plus"
else
    MODEL_PREFIX="nanodet-plus-m-${VARIANT}"
    OUT_PREFIX="nanodet_plus_${VARIANT}"
fi

echo "== Exporting: $MODEL_PREFIX (320 and 416) =="

echo "== Installing dependencies =="
# No torch/pytorch-lightning needed -- we start from RangiLyu's own
# pre-exported ONNX files, skipping the PyTorch export step entirely.
pip install onnx onnxsim onnx_graphsurgeon sng4onnx onnx2tf tensorflow tf_keras

for SIZE in 320 416; do
    ONNX_FILE="${MODEL_PREFIX}_${SIZE}.onnx"

    echo "== Downloading pre-exported ONNX: $ONNX_FILE (RangiLyu's official release) =="
    if [ ! -f "$ONNX_FILE" ]; then
        curl -fL -o "$ONNX_FILE" \
            "https://github.com/RangiLyu/nanodet/releases/download/${TAG}/${ONNX_FILE}"
    fi
    if [ ! -s "$ONNX_FILE" ]; then
        echo "ERROR: $ONNX_FILE is missing or empty after download. Check the URL/network and retry." >&2
        exit 1
    fi

    echo "== Converting ONNX -> SavedModel ($SIZE) =="
    # -osd / --output_signaturedefs: onnx2tf doesn't embed a signature_def by
    # default, which makes TFLiteConverter.from_saved_model() below fail with
    # "Only support at least one signature key." Required, not optional.
    onnx2tf -i "$ONNX_FILE" -o "$OUT_DIR/${OUT_PREFIX}_${SIZE}_sm" -osd

    # onnx2tf can fail on an unsupported op or internal error without
    # necessarily returning a nonzero exit code the shell's `set -e` catches
    # (same class of issue as export_yolox.sh's export_onnx.py step) --
    # verify the actual SavedModel file exists before trusting this step
    # succeeded, or the quantization steps below fail with a confusing,
    # unrelated-looking "SavedModel file does not exist" error instead.
    if [ ! -f "$OUT_DIR/${OUT_PREFIX}_${SIZE}_sm/saved_model.pb" ]; then
        echo "ERROR: onnx2tf did not produce a SavedModel for size $SIZE -- scroll up for its actual output/error." >&2
        exit 1
    fi

    echo "== Quantizing to TFLite dynamic range ($SIZE) =="
    python3 <<PYEOF
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("$OUT_DIR/${OUT_PREFIX}_${SIZE}_sm")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("$OUT_DIR/${OUT_PREFIX}_${SIZE}_dynamic.tflite", "wb") as f:
    f.write(tflite_model)
print("wrote $OUT_DIR/${OUT_PREFIX}_${SIZE}_dynamic.tflite")
PYEOF

    echo "== Exporting INT8 ($SIZE, calibrated on synthetic placeholder images) =="
    python3 <<PYEOF
import numpy as np
import tensorflow as tf

IMG_SIZE = $SIZE

# Synthetic random images as a placeholder -- same caveat as export_yolox.sh:
# gives a working int8 export for latency/memory feasibility numbers this
# week, but swap in ~100-200 real panel crops once that data exists.
def representative_dataset():
    rng = np.random.default_rng(seed=0)
    for _ in range(20):
        img = rng.random((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        yield [img]

converter = tf.lite.TFLiteConverter.from_saved_model("$OUT_DIR/${OUT_PREFIX}_${SIZE}_sm")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()

with open("$OUT_DIR/${OUT_PREFIX}_${SIZE}_int8.tflite", "wb") as f:
    f.write(tflite_model)
print("wrote $OUT_DIR/${OUT_PREFIX}_${SIZE}_int8.tflite")
PYEOF
done

echo "== Writing labels.txt (COCO-80 -- NanoDet-Plus-m is COCO-pretrained) =="
python3 -c "
coco_classes = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush',
]
with open('$OUT_DIR/${OUT_PREFIX}_labels.txt', 'w') as f:
    for name in coco_classes:
        f.write(name + '\n')
print('wrote $OUT_DIR/${OUT_PREFIX}_labels.txt (' + str(len(coco_classes)) + ' COCO classes -- placeholder until fine-tuned)')
"

echo ""
echo "== Done. Exported: =="
ls -la "$OUT_DIR/${OUT_PREFIX}"*.tflite "$OUT_DIR/${OUT_PREFIX}_labels.txt"

echo ""
echo "Next:"
echo "  1. Copy into the Android app's assets:"
echo "     cp $OUT_DIR/${OUT_PREFIX}_320_dynamic.tflite $OUT_DIR/${OUT_PREFIX}_320_int8.tflite \\"
echo "        $OUT_DIR/${OUT_PREFIX}_416_dynamic.tflite $OUT_DIR/${OUT_PREFIX}_416_int8.tflite \\"
echo "        $OUT_DIR/${OUT_PREFIX}_labels.txt \\"
echo "        ../android-benchmark-app/app/src/main/assets/models/detector/"
echo "  2. Add benchmark_config.json entries with model_path values like"
echo "     models/detector/${OUT_PREFIX}_320_dynamic.tflite, input_width/height"
echo "     matching each resolution (320 or 416)."
echo "  3. Remember the mean/std preprocessing caveat at the top of this script"
echo "     before wiring up real output decoding for this model."