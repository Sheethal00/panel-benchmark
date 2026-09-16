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

    echo "== Converting ONNX -> TFLite directly via onnx2tf ($SIZE) =="
    # This model's ShuffleNetV2 backbone uses GroupConvolution, which SavedModel
    # export doesn't support -- onnx2tf skips SavedModel entirely for it and
    # writes TFLite files directly instead (confirmed from actual run output:
    # "WARNING: ... GroupConvolution ... saved_model does not support
    # GroupConvolution" followed by "Float32 tflite output complete!"). So
    # unlike export_yolo.sh/export_yolox.sh, we use onnx2tf's own direct
    # quantization flags here rather than routing through SavedModel +
    # TFLiteConverter, which would never produce a SavedModel for this model.
    #
    # -odrqt: dynamic-range quantized TFLite, no calibration data needed.
    # -oiqt: full int8 quantized TFLite. Calibration is onnx2tf's own internal
    # random-data default here (no --custom_input_op_name_np_data_path given)
    # -- fine for latency/memory feasibility numbers this week, same synthetic-
    # calibration caveat as export_yolox.sh; swap in real panel photos later.
    #
    # The calibration_image_sample_data npy below is a SEPARATE thing: it's
    # for onnx2tf's own internal float-vs-quantized sanity check, which
    # otherwise tries to download a small dummy file from a GitHub release tag
    # that can go stale and 404 -- same workaround as export_yolo.sh.
    python3 -c "
import numpy as np
arr = np.random.rand(20, 128, 128, 3).astype(np.float32)
np.save('calibration_image_sample_data_20x128x128x3_float32.npy', arr)
"
    onnx2tf -i "$ONNX_FILE" -o "$OUT_DIR/${OUT_PREFIX}_${SIZE}_out" -odrqt -oiqt

    # Don't hardcode onnx2tf's exact output filenames -- find whatever it
    # actually produced by pattern, and fail with a directory listing if a
    # pattern doesn't match, rather than guess wrong and cascade into another
    # confusing downstream error.
    DYNAMIC_SRC=$(find "$OUT_DIR/${OUT_PREFIX}_${SIZE}_out" -iname "*dynamic_range_quant*.tflite" | head -n 1)
    INT8_SRC=$(find "$OUT_DIR/${OUT_PREFIX}_${SIZE}_out" -iname "*full_integer_quant*.tflite" | head -n 1)

    if [ -z "$DYNAMIC_SRC" ] || [ -z "$INT8_SRC" ]; then
        echo "ERROR: expected onnx2tf outputs not found for size $SIZE. Actual contents of $OUT_DIR/${OUT_PREFIX}_${SIZE}_out:" >&2
        ls -la "$OUT_DIR/${OUT_PREFIX}_${SIZE}_out" >&2
        exit 1
    fi

    cp "$DYNAMIC_SRC" "$OUT_DIR/${OUT_PREFIX}_${SIZE}_dynamic.tflite"
    cp "$INT8_SRC" "$OUT_DIR/${OUT_PREFIX}_${SIZE}_int8.tflite"
    echo "wrote $OUT_DIR/${OUT_PREFIX}_${SIZE}_dynamic.tflite (from $DYNAMIC_SRC)"
    echo "wrote $OUT_DIR/${OUT_PREFIX}_${SIZE}_int8.tflite (from $INT8_SRC)"
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