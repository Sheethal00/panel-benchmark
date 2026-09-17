#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Exports PP-PicoDet-S (PaddlePaddle/PaddleDetection, Apache-2.0) at 320 and
# 416 to TFLite: a dynamic-range variant and an int8 variant, plus the raw
# ONNX for direct ONNX Runtime benchmarking -- same pattern as the other
# export scripts in this repo.
#
# Usage:
#   ./export_picodet.sh          # PicoDet-S (LCNet backbone), 320 and 416
#
# This script creates and activates its OWN dedicated venv (panel-picodet-venv,
# alongside this script) automatically -- no need to pre-create/activate one
# yourself. That's deliberate, not just convenience: PaddlePaddle is a
# genuinely different framework stack from torch/tensorflow used elsewhere in
# this repo, and its package ecosystem can be picky about coexisting with
# other ML framework versions in one environment. Re-running this script
# reuses the same venv if it already exists.
#
# IMPORTANT, read before wiring up real inference (not just benchmarking):
# - Pipeline is PyTorch/TF-free until the final onnx2tf step: PaddlePaddle
#   (Baidu's own framework) is a genuinely different dependency stack from
#   every other script in this repo. Give it its own venv if you're
#   iterating on multiple exports -- Paddle's package ecosystem can be
#   picky about coexisting with torch/tensorflow versions in one environment.
# - Unlike YOLOX/NanoDet, PP-PicoDet's export commonly keeps NMS baked into
#   the graph by default (confirmed via PaddleDetection's own GitHub issues:
#   attempts to disable it via export_post_process=false/export_nms=false
#   don't fully remove it in recent releases) -- so this export is
#   effectively end-to-end like YOLO26, not raw-output-needs-NMS like
#   YOLOX/NanoDet. Worth confirming when you get to real output parsing.
# - LCNet (PicoDet's backbone) uses depthwise-separable convolutions, which
#   ONNX represents as GroupConvolution -- the same onnx2tf/SavedModel
#   incompatibility hit with NanoDet's ShuffleNetV2. --disable_group_convolution
#   is included from the start here to avoid repeating that whole debugging
#   cycle.
# ---------------------------------------------------------------------------

PICODET_SRC_DIR="./.paddledetection-src"
OUT_DIR="../models/detector"
mkdir -p "$OUT_DIR"

VENV_DIR="./panel-picodet-venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "== Creating dedicated venv for PicoDet export: $VENV_DIR =="
    python3 -m venv "$VENV_DIR"
fi
echo "== Activating $VENV_DIR =="
source "$VENV_DIR/bin/activate"

echo "== Installing dependencies =="
pip install "paddlepaddle==2.6.2"
pip install paddle2onnx onnxsim onnx onnx2tf tensorflow tf_keras

echo "== Cloning PaddleDetection (Apache-2.0) if not already present =="
if [ ! -d "$PICODET_SRC_DIR" ]; then
    git clone --depth 1 https://github.com/PaddlePaddle/PaddleDetection.git "$PICODET_SRC_DIR"
fi
cd "$PICODET_SRC_DIR"
pip install -r requirements.txt -q

for SIZE in 320 416; do
    CONFIG="configs/picodet/picodet_s_${SIZE}_coco_lcnet.yml"
    WEIGHTS_URL="https://paddledet.bj.bcebos.com/models/picodet_s_${SIZE}_coco_lcnet.pdparams"
    INFER_DIR="output_inference/picodet_s_${SIZE}_coco_lcnet"

    echo "== Exporting PicoDet-S ($SIZE): Paddle -> inference model =="
    python tools/export_model.py -c "$CONFIG" \
        -o weights="$WEIGHTS_URL" \
        --output_dir=output_inference

    if [ ! -f "$INFER_DIR/model.pdmodel" ]; then
        echo "ERROR: export_model.py did not produce $INFER_DIR/model.pdmodel -- scroll up for its actual output/error." >&2
        exit 1
    fi

    echo "== Converting Paddle inference model -> ONNX ($SIZE) =="
    paddle2onnx --model_dir "$INFER_DIR" \
        --model_filename model.pdmodel \
        --params_filename model.pdiparams \
        --opset_version 11 \
        --save_file "picodet_s_${SIZE}.onnx"

    if [ ! -s "picodet_s_${SIZE}.onnx" ]; then
        echo "ERROR: paddle2onnx did not produce picodet_s_${SIZE}.onnx -- scroll up for its actual output/error." >&2
        exit 1
    fi

    echo "== Simplifying ONNX ($SIZE) =="
    python -m onnxsim "picodet_s_${SIZE}.onnx" "picodet_s_${SIZE}_sim.onnx"

    # Keep the raw (simplified) ONNX for direct ONNX Runtime benchmarking,
    # same as every other model in this repo.
    cp "picodet_s_${SIZE}_sim.onnx" "$OUT_DIR/picodet_s_${SIZE}.onnx"
    echo "wrote $OUT_DIR/picodet_s_${SIZE}.onnx (for direct ONNX Runtime benchmarking)"

    echo "== onnx2tf's internal calibration sanity-check workaround =="
    python3 -c "
import numpy as np
arr = np.random.rand(20, 128, 128, 3).astype(np.float32)
np.save('calibration_image_sample_data_20x128x128x3_float32.npy', arr)
"

    echo "== Converting ONNX -> SavedModel ($SIZE) =="
    # -osd: required for TFLiteConverter.from_saved_model() below, same as
    # every other script. --disable_group_convolution: required because
    # LCNet's depthwise convs hit the same GroupConvolution/SavedModel
    # incompatibility as NanoDet's ShuffleNetV2 -- see note at top of file.
    onnx2tf -i "picodet_s_${SIZE}_sim.onnx" -o "$OUT_DIR/picodet_s_${SIZE}_sm" \
        -osd --disable_group_convolution

    if [ ! -f "$OUT_DIR/picodet_s_${SIZE}_sm/saved_model.pb" ]; then
        echo "ERROR: onnx2tf did not produce a SavedModel for size $SIZE -- scroll up for its actual output/error." >&2
        exit 1
    fi

    echo "== Quantizing to TFLite dynamic range ($SIZE) =="
    python3 <<PYEOF
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("$OUT_DIR/picodet_s_${SIZE}_sm")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("$OUT_DIR/picodet_s_${SIZE}_dynamic.tflite", "wb") as f:
    f.write(tflite_model)
print("wrote $OUT_DIR/picodet_s_${SIZE}_dynamic.tflite")
PYEOF

    echo "== Exporting INT8 ($SIZE, calibrated on synthetic placeholder images) =="
    python3 <<PYEOF
import numpy as np
import tensorflow as tf

IMG_SIZE = $SIZE

def representative_dataset():
    rng = np.random.default_rng(seed=0)
    for _ in range(20):
        img = rng.random((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        yield [img]

converter = tf.lite.TFLiteConverter.from_saved_model("$OUT_DIR/picodet_s_${SIZE}_sm")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()

with open("$OUT_DIR/picodet_s_${SIZE}_int8.tflite", "wb") as f:
    f.write(tflite_model)
print("wrote $OUT_DIR/picodet_s_${SIZE}_int8.tflite")
PYEOF
done

echo "== Writing labels.txt (COCO-80 -- PicoDet-S-coco_lcnet is COCO-pretrained) =="
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
with open('$OUT_DIR/picodet_s_labels.txt', 'w') as f:
    for name in coco_classes:
        f.write(name + '\n')
print('wrote $OUT_DIR/picodet_s_labels.txt (' + str(len(coco_classes)) + ' COCO classes -- placeholder until fine-tuned)')
"

cd ..
echo ""
echo "== Done. Exported: =="
ls -la "$OUT_DIR/picodet_s"*.tflite "$OUT_DIR/picodet_s"*.onnx "$OUT_DIR/picodet_s_labels.txt"

echo ""
echo "Next:"
echo "  1. Copy into the Android app's assets:"
echo "     cp $OUT_DIR/picodet_s_320_dynamic.tflite $OUT_DIR/picodet_s_320_int8.tflite \\"
echo "        $OUT_DIR/picodet_s_416_dynamic.tflite $OUT_DIR/picodet_s_416_int8.tflite \\"
echo "        $OUT_DIR/picodet_s_320.onnx $OUT_DIR/picodet_s_416.onnx \\"
echo "        $OUT_DIR/picodet_s_labels.txt \\"
echo "        ../android-benchmark-app/app/src/main/assets/models/detector/"
echo "  2. Add benchmark_config.json entries with model_path values like"
echo "     models/detector/picodet_s_320_dynamic.tflite, input_width/height"
echo "     matching each resolution (320 or 416)."