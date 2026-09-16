#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Exports a YOLO detector (YOLO26n, YOLOv8n, YOLO11n, or any weights file
# Ultralytics' YOLO() class accepts) into the 3 variants benchmark_config.json's
# "models" list expects: yolo_416_dynamic, yolo_320_dynamic, yolo_416_int8.
#
# Usage:
#   ./export_yolo.sh                          # yolo26n.pt, generic output names
#   ./export_yolo.sh yolo11n.pt yolo11n       # yolo11n.pt, TAGGED output names
#                                              # (yolo11n_416_dynamic.tflite, etc.) --
#                                              # sits ALONGSIDE previous exports instead
#                                              # of overwriting them
#   ./export_yolo.sh runs/train/exp/weights/best.pt panel-finetuned
#
# Run this INSIDE an activated venv:
#   python3 -m venv panel-detector-venv && source panel-detector-venv/bin/activate
#   ./export_yolo.sh yolo11n.pt yolo11n
#
# IMPORTANT: if you're comparing multiple YOLO generations (e.g. YOLO26n vs YOLO11n),
# always pass a TAG on every run after the first. Without one, output filenames are
# fixed generic names (yolo_416_dynamic.tflite etc.) and a second run will silently
# overwrite the first's exports.
#
# Output filenames are generic (yolo_416_dynamic.tflite etc.) regardless of
# which architecture you point this at -- swap WEIGHTS and re-run to compare
# a different YOLO generation without touching benchmark_config.json at all.
# If you want to benchmark two YOLO generations side by side instead of
# overwriting one with the other, run this once per generation into separate
# OUT_DIRs (edit OUT_DIR below) and give each its own config entries.
#
# Adapted from a working export pipeline -- the calibration-file and
# litert_torch workarounds below are load-bearing, not optional cleanup.
# See inline notes before removing either.
# ---------------------------------------------------------------------------

WEIGHTS="${1:-yolo26n.pt}"
TAG="${2:-}"
PREFIX="${TAG:+${TAG}_}"   # e.g. TAG=yolo11n -> PREFIX="yolo11n_"; TAG unset -> PREFIX=""
OUT_DIR="../models/detector"
mkdir -p "$OUT_DIR"

echo "== Exporting from weights: $WEIGHTS (output prefix: '${PREFIX:-<none>}') =="
echo "== Installing dependencies =="
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NOTE: onnx2tf is only used here for float ONNX -> SavedModel conversion.
# We deliberately never pass its quantization flags (-oiqt / -odrqt), because
# that code path calls onnx2tf's internal download_test_image_data(), which
# fetches a calibration sample .npy from a hardcoded GitHub release tag
# baked into onnx2tf/utils/common_functions.py. That tag can go stale
# relative to newer onnx2tf releases and 404 -- if you hit this, confirm with:
#   curl -L -o cal.npy '<url from common_functions.py>'
# Doing dynamic-range quantization ourselves via tf.lite.TFLiteConverter
# avoids that code path entirely and produces the identical quant scheme.
pip install \
ultralytics \
onnx \
onnxsim \
onnx_graphsurgeon \
sng4onnx \
onnx2tf \
tensorflow \
tf_keras

# opset: YOLO26 is a from-scratch NMS-free end-to-end head (native end-to-end
# inference, no separate NMS post-processing needed in the exported graph).
# Some of the ops in that head are newer than opset 12 in other YOLO26
# export pipelines people have reported -- if the ONNX export below fails
# on an unsupported-op error, bump OPSET to 17+ and retry before digging
# further; this is a common first failure point, not a sign of a broken
# environment.
OPSET=12

echo "== Exporting ONNX (416 and 320, opset $OPSET) =="
python3 <<PYEOF
import shutil
from pathlib import Path
from ultralytics import YOLO

model = YOLO("$WEIGHTS")
weights_stem = Path("$WEIGHTS").stem  # e.g. "yolo26n", "yolo11n", "best"

# Ultralytics always names ONNX output after the weights file's stem
# (e.g. yolo26n.onnx, yolo11n.onnx) regardless of imgsz -- the second
# export() call below would silently overwrite the first one. Rename
# immediately after each export using a fixed, architecture-agnostic name.
model.export(format="onnx", imgsz=416, opset=$OPSET)
shutil.move(f"{weights_stem}.onnx", "yolo_416.onnx")

model.export(format="onnx", imgsz=320, opset=$OPSET)
shutil.move(f"{weights_stem}.onnx", "yolo_320.onnx")
PYEOF

echo "== Synthesizing onnx2tf calibration test file =="
# onnx2tf's convert() unconditionally calls download_test_image_data(),
# which fetches a small dummy .npy from a GitHub release tag hardcoded in
# onnx2tf/utils/common_functions.py -- see the note above the pip install.
# download_test_image_data() skips the fetch entirely if the file already
# exists at os.path.join(os.getcwd(), FILE_NAME), so we create it ourselves.
# It's only used for onnx2tf's internal ONNX-vs-TF accuracy sanity check --
# not for calibrating quantization -- so random values are fine. Must be run
# from the same directory onnx2tf is invoked from (CWD, not $OUT_DIR).
python3 -c "
import numpy as np
arr = np.random.rand(20, 128, 128, 3).astype(np.float32)
np.save('calibration_image_sample_data_20x128x128x3_float32.npy', arr)
"

echo "== Converting ONNX -> SavedModel (float, no quantization) =="
# -osd / --output_signaturedefs: onnx2tf does NOT embed a signature_def in
# the SavedModel by default, which makes TFLiteConverter.from_saved_model()
# below fail with "Only support at least one signature key." This flag is
# what actually gives the SavedModel a usable signature -- required, not
# optional, for the quantization steps that follow.
onnx2tf -i yolo_416.onnx -o "$OUT_DIR/${PREFIX}sm_416" -osd
onnx2tf -i yolo_320.onnx -o "$OUT_DIR/${PREFIX}sm_320" -osd

echo "== Quantizing to TFLite dynamic range (weights int8, activations float) =="
python3 <<PYEOF
import tensorflow as tf

pairs = [
    ("$OUT_DIR/${PREFIX}sm_416", "$OUT_DIR/${PREFIX}yolo_416_dynamic.tflite"),
    ("$OUT_DIR/${PREFIX}sm_320", "$OUT_DIR/${PREFIX}yolo_320_dynamic.tflite"),
]
for saved_model_dir, out_path in pairs:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic range quant
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"wrote {out_path}")
PYEOF

echo "== Exporting 416 INT8 (calibrated on bundled coco8 set) =="
# NOTE: we deliberately do NOT use Ultralytics' native
# model.export(format="tflite", int8=True, ...) here. On some Ultralytics
# versions that routes through a "litert_torch" backend (import chain:
# export_litert -> litert_torch -> torch._dynamo -> torch.distributed.tensor
# ._ops._math_ops), which references aten.cholesky during import. On some
# torch builds that op isn't registered under the aten namespace at import
# time, so it crashes with:
#   AttributeError: '_OpNamespace' 'aten' object has no attribute 'cholesky'
# before any actual export happens -- a litert_torch/torch version
# incompatibility unrelated to the model. We quantize ourselves instead,
# using the SavedModel onnx2tf already produced (sm_416) plus a
# representative dataset built from the same coco8 images, via
# tf.lite.TFLiteConverter -- same mechanism as the dynamic-range step above,
# just with a representative_dataset added for full-integer calibration.
python3 <<PYEOF
import glob
import numpy as np
import tensorflow as tf
from ultralytics.data.utils import check_det_dataset

# Downloads (if needed) and locates the tiny 8-image coco8 set that
# Ultralytics normally uses internally for this same calibration purpose.
# This calibrates activation ranges only -- it is NOT a substitute for
# calibrating on real panel photos once you have a labeled panel dataset.
# Swap img_dir below for a folder of ~100-200 representative panel crops
# as soon as that data exists; coco8 is a placeholder to get a feasibility
# number this week, not the calibration set to ship with.
data = check_det_dataset("coco8.yaml")
img_dir = data["train"]  # coco8's train split IS the 8 sample images
img_paths = sorted(glob.glob(f"{img_dir}/*.jpg"))
if not img_paths:
    raise RuntimeError(f"No calibration images found under {img_dir}")

IMG_SIZE = 416

def load_image(path):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def representative_dataset():
    for p in img_paths:
        img = load_image(p)
        yield [tf.expand_dims(img, axis=0)]

converter = tf.lite.TFLiteConverter.from_saved_model("$OUT_DIR/${PREFIX}sm_416")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Keep float in/out -- change both to tf.int8 or tf.uint8 if your Android
# inference code expects quantized tensors at the model boundary instead.
tflite_model = converter.convert()

with open("$OUT_DIR/${PREFIX}yolo_416_int8.tflite", "wb") as f:
    f.write(tflite_model)
print("wrote $OUT_DIR/${PREFIX}yolo_416_int8.tflite")
PYEOF

echo "== Writing labels.txt (from model's own class names) =="
# Writes whatever class names this checkpoint reports -- COCO-80 for a stock
# pretrained checkpoint (yolo26n.pt, yolo11n.pt, yolov8n.pt, ...), or your
# own panel-device classes if $WEIGHTS is a fine-tuned checkpoint. Either way
# this reflects the ACTUAL weights you pointed the script at, so no separate
# "placeholder" caveat is needed once you're exporting a fine-tuned model --
# just be aware a stock pretrained checkpoint's labels won't match panel
# device types (MCB, contactor, relay, etc).
python3 -c "
from ultralytics import YOLO
model = YOLO('$WEIGHTS')
names = model.names
with open('$OUT_DIR/${PREFIX}labels.txt', 'w') as f:
    for i in sorted(names.keys()):
        f.write(names[i] + '\n')
print('wrote $OUT_DIR/${PREFIX}labels.txt (' + str(len(names)) + ' classes from $WEIGHTS)')
"

echo ""
echo "== Done. Exported models: =="
ls -la "$OUT_DIR"/${PREFIX}*.tflite "$OUT_DIR/${PREFIX}labels.txt"

echo ""
echo "Next:"
echo "  1. Copy these into the Android app's assets so they're bundled in the APK:"
echo "     cp $OUT_DIR/${PREFIX}*.tflite $OUT_DIR/${PREFIX}labels.txt \\"
echo "        ../android-benchmark-app/app/src/main/assets/models/detector/"
echo "  2. Add matching benchmark_config.json entries with model_path values like"
echo "     models/detector/${PREFIX}yolo_416_dynamic.tflite -- see README for the"
echo "     pattern used by the existing (untagged) YOLO26n entries."