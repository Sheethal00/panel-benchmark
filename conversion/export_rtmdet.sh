#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Exports RTMDet-tiny (OpenMMLab/mmdetection, Apache-2.0) to ONNX via
# MMDeploy (the officially documented export path for RTMDet -- mmdetection's
# own README points here, not a plain torch.onnx.export script), then to
# TFLite via the same onnx2tf pipeline used for every other model here.
#
# Usage:
#   ./export_rtmdet.sh
#
# This script creates and activates its OWN dedicated venv
# (panel-rtmdet-venv, alongside this script), same reasoning as
# export_picodet.sh: this is a genuinely different, notoriously
# version-sensitive dependency stack (mmcv/mmdet/mmdeploy) that should not
# share an environment with torch/tensorflow used elsewhere in this repo.
#
# HONEST RISK NOTE: this is the highest-risk export in this repo. mmcv/mmdet
# version pinning is a well-documented source of friction across the
# OpenMMLab ecosystem -- more so than anything hit with YOLOX/NanoDet/PicoDet.
# `mim` (OpenMMLab's own dependency resolver, used below) exists specifically
# to reduce that pain, but this may still need a debugging round if versions
# have drifted since this script was written.
#
# IMPORTANT, read before wiring up real inference (not just benchmarking):
# - RTMDet is NOT NMS-free, like YOLOX/NanoDet -- MMDeploy's static ONNX
#   export for detection models typically DOES bundle NMS via TensorRT's/
#   ONNX Runtime's NMS op in the deployment config used below
#   (detection_onnxruntime-static), so this export is likely closer to
#   YOLO26/PicoDet (end-to-end-ish) than to YOLOX/NanoDet -- verify once you
#   get to real output parsing, the same way PicoDet's NMS-baked-in status
#   was confirmed empirically rather than assumed.
# - RTMDet-tiny's native resolution is 640x640 (its training pipeline is
#   fixed at that size) -- larger than every other candidate in this repo
#   (320/416). Expect it to be slower and heavier purely from that size
#   difference, independent of architecture quality.
# ---------------------------------------------------------------------------

OUT_DIR="../models/detector"
MMDET_SRC_DIR="./.mmdetection-src"
MMDEPLOY_SRC_DIR="./.mmdeploy-src"
mkdir -p "$OUT_DIR"

VENV_DIR="./panel-rtmdet-venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "== Creating dedicated venv for RTMDet export: $VENV_DIR =="
    python3 -m venv "$VENV_DIR"
fi
echo "== Activating $VENV_DIR =="
source "$VENV_DIR/bin/activate"

echo "== Installing PyTorch (CPU) =="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "== Installing OpenMMLab stack via mim (their own dependency resolver) =="
# mim exists specifically to avoid hand-pinning mmcv wheel URLs per
# torch/cuda combination -- the single most common source of OpenMMLab
# environment breakage. Let it resolve versions rather than pinning by hand.
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"

echo "== Installing ONNX/TFLite conversion toolchain =="
pip install onnx onnxsim onnx_graphsurgeon sng4onnx onnx2tf tensorflow tf_keras onnxruntime psutil
# NOTE: the protobuf pin needed for onnx2tf/TensorFlow is applied LATER,
# right before the onnx2tf call -- not here. MMDeploy's own install (below)
# pulls in an older protobuf as a dependency and would silently undo a pin
# placed here before it even runs (confirmed: this was the actual cause of a
# segfault that persisted even with this exact pin present earlier in the
# script). MMDeploy's export step may itself expect that older protobuf, so
# upgrading too early risks breaking the ONNX export instead of fixing
# onnx2tf -- do it only once MMDeploy's own work is done.

echo "== Cloning mmdetection (for configs) if not already present =="
if [ ! -d "$MMDET_SRC_DIR" ]; then
    git clone --depth 1 --branch main https://github.com/open-mmlab/mmdetection.git "$MMDET_SRC_DIR"
fi

echo "== Cloning MMDeploy (for the export tool + deployment configs) if not already present =="
if [ ! -d "$MMDEPLOY_SRC_DIR" ]; then
    git clone --depth 1 https://github.com/open-mmlab/mmdeploy.git "$MMDEPLOY_SRC_DIR"
fi
pip install -e "$MMDEPLOY_SRC_DIR"

echo "== Downloading RTMDet-tiny checkpoint (official OpenMMLab release) =="
CKPT_URL="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
CKPT_FILE="rtmdet_tiny_8xb32-300e_coco.pth"
if [ ! -f "$CKPT_FILE" ]; then
    curl -fL -o "$CKPT_FILE" "$CKPT_URL"
fi
if [ ! -s "$CKPT_FILE" ]; then
    echo "ERROR: $CKPT_FILE is missing or empty after download. Check the URL/network and retry." >&2
    exit 1
fi

echo "== Fetching a sample image for MMDeploy's tracing step =="
# tools/deploy.py needs a real image to trace through during export -- any
# real photo works; content doesn't affect the exported model's weights.
SAMPLE_IMG="sample.jpg"
if [ ! -f "$SAMPLE_IMG" ]; then
    curl -fL -o "$SAMPLE_IMG" \
        "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/demo/demo.jpg"
fi

echo "== Exporting to ONNX via MMDeploy (ONNX Runtime backend, static shape) =="
# Config filename confirmed via MMDeploy's own GitHub discussions (#2096) --
# unlike the TensorRT configs, the ONNX Runtime static config has no
# resolution suffix in its filename. RTMDet-tiny's own training pipeline is
# already fixed at 640x640 (see note at top of file), so this config's
# default static shape matches without needing an override.
DEPLOY_CFG="$MMDEPLOY_SRC_DIR/configs/mmdet/detection/detection_onnxruntime_static.py"

if [ ! -f "$DEPLOY_CFG" ]; then
    echo "ERROR: $DEPLOY_CFG not found. MMDeploy's config directory layout may" >&2
    echo "have changed since this script was written. Check:" >&2
    echo "  ls $MMDEPLOY_SRC_DIR/configs/mmdet/detection/ | grep onnxruntime" >&2
    echo "and update DEPLOY_CFG above to whatever exists (the *_dynamic.py variant" >&2
    echo "is a documented fallback, but needs a fixed-batch pass in onnx2tf, -b 1," >&2
    echo "the same way export_picodet.sh handles a dynamic-batch ONNX export)." >&2
    exit 1
fi
MMDET_CFG="$MMDET_SRC_DIR/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py"
WORK_DIR="./rtmdet_work_dir"
mkdir -p "$WORK_DIR"

# PyTorch 2.6+ defaults torch.load() to weights_only=True, which rejects the
# OpenMMLab checkpoint's pickled objects (e.g. HistoryBuffer) with a
# "Weights only load failed" / "not allowlisted" error -- confirmed via a
# real failure here. This checkpoint is from OpenMMLab's own official
# release (trusted source), so forcing the legacy full-unpickling load
# behavior is appropriate.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python "$MMDEPLOY_SRC_DIR/tools/deploy.py" \
    "$DEPLOY_CFG" \
    "$MMDET_CFG" \
    "$CKPT_FILE" \
    "$SAMPLE_IMG" \
    --work-dir "$WORK_DIR" \
    --device cpu

if [ ! -s "$WORK_DIR/end2end.onnx" ]; then
    echo "ERROR: MMDeploy did not produce $WORK_DIR/end2end.onnx -- scroll up for its actual output/error." >&2
    exit 1
fi

cp "$WORK_DIR/end2end.onnx" "$OUT_DIR/rtmdet_tiny.onnx"
echo "wrote $OUT_DIR/rtmdet_tiny.onnx (for direct ONNX Runtime benchmarking)"

echo "== Pinning protobuf for onnx2tf/TensorFlow (MMDeploy's export is done now) =="
# TensorFlow 2.20+ segfaults against an old protobuf (3.20.x) that MMDeploy's
# own install pulls in as a dependency -- confirmed via a real, reproducible
# segfault during ONNX -> SavedModel conversion below. Applying this pin any
# earlier gets silently undone by MMDeploy's install step (also confirmed).
# MMDeploy's own export work above is finished by this point, so its
# possible preference for an older protobuf no longer matters.
pip install --upgrade "protobuf>=5.28,<6"

echo "== onnx2tf's internal calibration sanity-check workaround =="
python3 -c "
import numpy as np
arr = np.random.rand(20, 128, 128, 3).astype(np.float32)
np.save('calibration_image_sample_data_20x128x128x3_float32.npy', arr)
"

echo "== Converting ONNX -> SavedModel =="
# -osd: required for TFLiteConverter.from_saved_model() below, same as every
# other script. -b 1: fixes the batch dimension -- if this export bakes in
# NMS (see note at top of file), that op typically requires a fixed batch
# size, same as PicoDet's export needed. --disable_group_convolution:
# RTMDet's CSPNeXt backbone also uses depthwise-separable convs in places --
# included preemptively given NanoDet and PicoDet both hit this exact
# incompatibility; harmless if this model doesn't actually need it.
onnx2tf -i "$OUT_DIR/rtmdet_tiny.onnx" -o "$OUT_DIR/rtmdet_tiny_sm" \
    -osd -b 1 --disable_group_convolution

if [ ! -f "$OUT_DIR/rtmdet_tiny_sm/saved_model.pb" ]; then
    echo "ERROR: onnx2tf did not produce a SavedModel -- scroll up for its actual output/error." >&2
    exit 1
fi

echo "== Quantizing to TFLite dynamic range =="
python3 <<PYEOF
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("$OUT_DIR/rtmdet_tiny_sm")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("$OUT_DIR/rtmdet_tiny_dynamic.tflite", "wb") as f:
    f.write(tflite_model)
print("wrote $OUT_DIR/rtmdet_tiny_dynamic.tflite")
PYEOF

echo "== Exporting INT8 (calibrated on synthetic placeholder images) =="
# NOTE: if this model also bakes in NMS/box-rescaling the way PicoDet does,
# it may hit the same TFLite DIV-kernel-doesn't-support-INT8 gap seen there.
# If int8 export fails or crashes at inference with a similar "Div only
# supports FLOAT32, INT32 and quantized UINT8" error, drop this variant the
# same way PicoDet's int8 was dropped -- it's a real TFLite kernel
# limitation, not something fixable by adjusting representative_dataset.
python3 <<PYEOF
import numpy as np
import tensorflow as tf

IMG_SIZE = 640

def representative_dataset():
    rng = np.random.default_rng(seed=0)
    for _ in range(20):
        img = rng.random((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        yield [img]

converter = tf.lite.TFLiteConverter.from_saved_model("$OUT_DIR/rtmdet_tiny_sm")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,  # float fallback for ops without an INT8 kernel
]
tflite_model = converter.convert()

with open("$OUT_DIR/rtmdet_tiny_int8.tflite", "wb") as f:
    f.write(tflite_model)
print("wrote $OUT_DIR/rtmdet_tiny_int8.tflite")
PYEOF

echo "== Writing labels.txt (COCO-80 -- RTMDet-tiny is COCO-pretrained) =="
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
with open('$OUT_DIR/rtmdet_tiny_labels.txt', 'w') as f:
    for name in coco_classes:
        f.write(name + '\n')
print('wrote $OUT_DIR/rtmdet_tiny_labels.txt (' + str(len(coco_classes)) + ' COCO classes -- placeholder until fine-tuned)')
"

echo ""
echo "== Done. Exported: =="
ls -la "$OUT_DIR"/rtmdet_tiny*

echo ""
echo "Next:"
echo "  1. Copy into the Android app's assets:"
echo "     cp $OUT_DIR/rtmdet_tiny_dynamic.tflite $OUT_DIR/rtmdet_tiny_int8.tflite \\"
echo "        $OUT_DIR/rtmdet_tiny.onnx $OUT_DIR/rtmdet_tiny_labels.txt \\"
echo "        ../android-benchmark-app/app/src/main/assets/models/detector/"
echo "  2. Add benchmark_config.json entries with model_path values like"
echo "     models/detector/rtmdet_tiny_dynamic.tflite, input_width/height 640"
echo "     (RTMDet-tiny's fixed native resolution -- larger than other candidates)."
echo "  3. If int8 crashes at inference with a Div-kernel error, drop it the"
echo "     same way PicoDet's int8 was dropped -- see note above the int8 step."