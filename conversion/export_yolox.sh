#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Exports YOLOX-Nano or YOLOX-Tiny (Megvii-BaseDetection, Apache-2.0 license --
# the reason this replaces the Ultralytics YOLO family, which is AGPL-3.0 and
# requires open-sourcing your app or an Enterprise license for commercial use)
# to TFLite: a dynamic-range variant and an int8 variant.
#
# Usage:
#   ./export_yolox.sh                # defaults to yolox_nano
#   ./export_yolox.sh yolox_tiny      # or yolox_nano
#
# Run this INSIDE an activated venv:
#   python3 -m venv panel-detector-venv && source panel-detector-venv/bin/activate
#   ./export_yolox.sh yolox_nano
#
# IMPORTANT DIFFERENCES FROM export_yolo.sh (Ultralytics-family export):
# - Resolution is FIXED at 416x416 for both yolox_nano and yolox_tiny -- it's
#   baked into their exp config files, not a simple --imgsz-style CLI flag.
#   So this script only produces one resolution, not a 416+320 pair.
# - YOLOX is NOT NMS-free like YOLO26. --decode_in_inference below makes the
#   exported graph output decoded box coordinates directly, but a separate
#   NMS pass is still required in your app code afterward -- unlike YOLO26's
#   fully end-to-end output. Budget for that when wiring up real post-processing.
# - No bundled COCO8-style calibration set the way Ultralytics ships one, so
#   int8 calibration below uses synthetic random images as a placeholder --
#   swap in real panel photos once you have them, same caveat as export_yolo.sh.
# ---------------------------------------------------------------------------

MODEL_NAME="${1:-yolox_nano}"   # yolox_nano or yolox_tiny
OUT_DIR="../models/detector"
YOLOX_SRC_DIR="./.yolox-src"
mkdir -p "$OUT_DIR"

echo "== Exporting: $MODEL_NAME (fixed 416x416) =="

echo "== Installing dependencies =="
# Install torch AND torchvision TOGETHER from the same CPU wheel index. If
# torchvision is left to pip's default resolution (e.g. pulled in later as a
# YOLOX dependency from plain PyPI), it can land a build with a different ABI
# than this torch build, breaking C++ extension ops it registers (e.g.
# torchvision::nms) at import time. Installing both from one index call
# avoids that mismatch.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install \
onnx \
onnxsim \
onnx_graphsurgeon \
sng4onnx \
onnx2tf \
tensorflow \
tf_keras \
loguru \
thop \
ninja \
tabulate \
psutil \
tqdm \
pycocotools

echo "== Cloning YOLOX (Apache-2.0) if not already present =="
if [ ! -d "$YOLOX_SRC_DIR" ]; then
    git clone --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git "$YOLOX_SRC_DIR"
fi
cd "$YOLOX_SRC_DIR"
pip install -v -e . -q

echo "== Downloading pretrained weights (Megvii's official release, tag 0.1.1rc0) =="
if [ ! -f "${MODEL_NAME}.pth" ]; then
    # -f: treat HTTP error responses (404 etc.) as a real failure instead of
    # silently saving the error page as if it were the weights file.
    curl -fL -o "${MODEL_NAME}.pth" \
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/${MODEL_NAME}.pth"
fi
if [ ! -s "${MODEL_NAME}.pth" ]; then
    echo "ERROR: ${MODEL_NAME}.pth is missing or empty after download. Check the URL/network and retry." >&2
    exit 1
fi

echo "== Exporting PyTorch -> ONNX =="
# --decode_in_inference: bake box decoding into the exported graph (still
# needs a separate NMS pass afterward -- see note above). opset 12 to match
# export_yolo.sh's default; bump if onnx2tf/TFLite conversion below complains
# about an unsupported op.
#
# YOLOX's own tools/export_onnx.py calls torch.onnx._export(), a private API
# removed from modern PyTorch (2.x) in favor of the public torch.onnx.export().
# That's an age mismatch in YOLOX's own tooling, not a real environment
# problem -- rather than pin an old torch (which risks reintroducing the
# torch/torchvision ABI mismatch fixed earlier), alias _export to the public
# export() here. dynamo=False pins the stable TorchScript-based exporter path
# the original code was written against, in case a future torch version
# changes that default.
python3 -c "
import sys
import torch

def _export_shim(*args, **kwargs):
    kwargs.setdefault('dynamo', False)
    return torch.onnx.export(*args, **kwargs)

if not hasattr(torch.onnx, '_export'):
    torch.onnx._export = _export_shim

sys.argv = [
    'export_onnx.py',
    '-n', '$MODEL_NAME',
    '-c', '${MODEL_NAME}.pth',
    '--output-name', '${MODEL_NAME}.onnx',
    '--decode_in_inference',
    '-o', '12',
]
sys.path.insert(0, 'tools')
import export_onnx
export_onnx.main()
"

# export_onnx.py wraps main() in loguru's @logger.catch, which logs an
# exception's full traceback but does NOT re-raise it -- the process exits 0
# even after a hard failure, so `set -e` above can't catch it. Check the
# actual output file exists before trusting this step succeeded, or every
# later step silently operates on stale/missing data and fails with a
# confusing, unrelated-looking error instead.
if [ ! -s "${MODEL_NAME}.onnx" ]; then
    echo "ERROR: ${MODEL_NAME}.onnx was not created -- the export_onnx.py step above failed" >&2
    echo "(scroll up for its traceback; loguru's @logger.catch hides the failure from this script's exit code)." >&2
    exit 1
fi

# Keep a copy of the raw ONNX file itself alongside the TFLite outputs below,
# so it can be benchmarked directly through the ONNX Runtime path (runtime:
# "onnx" in benchmark_config.json) as well as via onnx2tf -> TFLite -- same
# weights, two runtimes, directly comparable.
cp "${MODEL_NAME}.onnx" "$OUT_DIR/${MODEL_NAME}.onnx"
echo "wrote $OUT_DIR/${MODEL_NAME}.onnx (for direct ONNX Runtime benchmarking)"

echo "== Converting ONNX -> SavedModel (float, no quantization) =="
# Same onnx2tf calibration-file workaround as export_yolo.sh: create the
# dummy npy onnx2tf's internal sanity check expects, so it skips a fetch
# from a GitHub release tag that can go stale and 404.
python3 -c "
import numpy as np
arr = np.random.rand(20, 128, 128, 3).astype(np.float32)
np.save('calibration_image_sample_data_20x128x128x3_float32.npy', arr)
"
onnx2tf -i "${MODEL_NAME}.onnx" -o "../$OUT_DIR/${MODEL_NAME}_sm" -osd
# -osd / --output_signaturedefs above: onnx2tf doesn't embed a signature_def
# by default, which makes TFLiteConverter.from_saved_model() below fail with
# "Only support at least one signature key." -- required, not optional.

echo "== Quantizing to TFLite dynamic range (weights int8, activations float) =="
python3 <<PYEOF
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("../$OUT_DIR/${MODEL_NAME}_sm")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("../$OUT_DIR/${MODEL_NAME}_dynamic.tflite", "wb") as f:
    f.write(tflite_model)
print("wrote ../$OUT_DIR/${MODEL_NAME}_dynamic.tflite")
PYEOF

echo "== Exporting INT8 (calibrated on synthetic placeholder images) =="
python3 <<PYEOF
import numpy as np
import tensorflow as tf

IMG_SIZE = 416

# No bundled calibration set for YOLOX the way Ultralytics ships coco8 --
# synthetic random images are a placeholder ONLY, to get a working int8
# export for latency/memory feasibility numbers this week. Swap in ~100-200
# real panel crops once that data exists; calibrating on noise gives a
# working model but not activation ranges representative of real inputs.
def representative_dataset():
    rng = np.random.default_rng(seed=0)
    for _ in range(20):
        img = rng.random((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        yield [img]

converter = tf.lite.TFLiteConverter.from_saved_model("../$OUT_DIR/${MODEL_NAME}_sm")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()

with open("../$OUT_DIR/${MODEL_NAME}_int8.tflite", "wb") as f:
    f.write(tflite_model)
print("wrote ../$OUT_DIR/${MODEL_NAME}_int8.tflite")
PYEOF

echo "== Writing labels.txt (COCO-80 -- YOLOX-Nano/Tiny are COCO-pretrained) =="
python3 -c "
from yolox.data.datasets import COCO_CLASSES
with open('../$OUT_DIR/${MODEL_NAME}_labels.txt', 'w') as f:
    for name in COCO_CLASSES:
        f.write(name + '\n')
print('wrote ../$OUT_DIR/${MODEL_NAME}_labels.txt (' + str(len(COCO_CLASSES)) + ' COCO classes -- placeholder until fine-tuned)')
"

cd ..
echo ""
echo "== Done. Exported: =="
ls -la "$OUT_DIR/${MODEL_NAME}"*.tflite "$OUT_DIR/${MODEL_NAME}.onnx" "$OUT_DIR/${MODEL_NAME}_labels.txt"

echo ""
echo "Next:"
echo "  1. Copy into the Android app's assets:"
echo "     cp $OUT_DIR/${MODEL_NAME}_dynamic.tflite $OUT_DIR/${MODEL_NAME}_int8.tflite \\"
echo "        $OUT_DIR/${MODEL_NAME}.onnx $OUT_DIR/${MODEL_NAME}_labels.txt \\"
echo "        ../android-benchmark-app/app/src/main/assets/models/detector/"
echo "  2. Add benchmark_config.json entries with model_path values like"
echo "     models/detector/${MODEL_NAME}_dynamic.tflite (runtime: tflite) and"
echo "     models/detector/${MODEL_NAME}.onnx (runtime: onnx) -- input_width/height"
echo "     416 for all of them (fixed resolution, see note at top of this script)."