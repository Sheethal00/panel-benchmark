#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Exports MMOCR's lightweight detection+recognition pair (OpenMMLab/mmocr,
# Apache-2.0) to ONNX via MMDeploy: DBNet-ResNet18 (text detection) and
# CRNN (text recognition, MobileNetV3-free "mini-vgg" backbone). This is a
# genuinely different architecture family from PaddleOCR, not a repackaging
# of the same weights -- the point of adding it as a third OCR candidate.
#
# Usage:
#   ./export_mmocr.sh
#
# This script creates and activates its OWN dedicated venv
# (panel-mmocr-venv), same reasoning as export_picodet.sh/export_rtmdet.sh:
# keeps this version-sensitive dependency stack isolated.
#
# ONNX ONLY -- NO TFLITE OUTPUT, deliberately, from the start. PaddleOCR's
# export took many rounds to end up here anyway (real TFLite kernel gaps,
# an op-version skew between the Python tensorflow converter and the app's
# TFLite AAR even at matching version tags, and a build-breaking duplicate-
# class conflict from trying to bump the AAR to chase compatibility -- see
# export_paddleocr.sh's header for the full story). Going ONNX-only from
# the outset here skips all of that; ONNX Runtime has none of this fragility.
#
# Much of this script's structure mirrors export_rtmdet.sh (same MMDeploy
# toolchain), with two known fixes applied from the start instead of
# rediscovered:
# - PyTorch 2.6+ defaults torch.load() to weights_only=True, which rejects
#   OpenMMLab checkpoints' pickled objects -- TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
#   is set before the checkpoint-loading step.
# - torch + torchvision installed TOGETHER from the same CPU wheel index,
#   to avoid an ABI mismatch that breaks torchvision::nms at import time.
#
# IMPORTANT, read before wiring up real inference (not just benchmarking):
# - DBNet's output is a probability MAP (like PaddleOCR's detection model),
#   not boxes directly -- needs threshold + contour-finding post-processing.
# - CRNN's output is a CTC character-probability sequence -- needs CTC
#   decoding with a character dictionary to get actual text. This script
#   does not currently save that dictionary (MMOCR's CRNN default dict is
#   bundled in its config/checkpoint metadata rather than a standalone
#   text file the way PaddleOCR ships en_dict.txt) -- add that separately
#   before wiring up real decode.
# - Both models use DYNAMIC input shapes in this export (matching MMDeploy's
#   documented onnxruntime_dynamic configs) -- ONNX Runtime handles dynamic
#   shapes natively, so this isn't a problem the way it was for the
#   TFLite/onnx2tf path attempted for PaddleOCR.
# ---------------------------------------------------------------------------

OUT_DIR="../models/ocr"
MMDEPLOY_SRC_DIR="./.mmdeploy-src"
mkdir -p "$OUT_DIR"

VENV_DIR="./panel-mmocr-venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "== Creating dedicated venv for MMOCR export: $VENV_DIR =="
    python3 -m venv "$VENV_DIR"
fi
echo "== Activating $VENV_DIR =="
source "$VENV_DIR/bin/activate"

# PIP_CONSTRAINT applies to pip's ISOLATED BUILD ENVIRONMENTS too, unlike a
# plain `pip install "setuptools<81"` into this venv -- which does NOT
# affect them at all. Confirmed the hard way: multiple different packages
# needing to build from source (mmdeploy via -e, then mmcv when no
# matching prebuilt wheel exists for the installed torch version) each hit
# their own fresh "ModuleNotFoundError: No module named 'pkg_resources'"
# inside their own independently-resolved isolated setuptools, despite
# this venv's own setuptools being correctly pinned. This constraint file
# is the actual general fix, not another one-off per-package pin.
cat > /tmp/panel_mmocr_pip_constraints.txt << 'CONSTRAINTS'
setuptools>=64,<81
CONSTRAINTS
export PIP_CONSTRAINT=/tmp/panel_mmocr_pip_constraints.txt

echo "== Installing PyTorch (CPU) =="
# torch + torchvision installed TOGETHER from the same index -- see header note.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "== Installing OpenMMLab stack via mim =="
pip install openmim
mim install mmengine
# mmcv range constrained by mmdet (which is itself constrained by mmocr,
# see below) -- confirmed via a real "MMCV==2.1.0 is used but incompatible"
# assertion when mmcv resolved past what mmdet 3.1.x actually supports.
# MMCV_WITH_OPS=1: mmcv 2.0.x predates official support for recent torch
# releases, so no matching prebuilt wheel exists here and it builds from
# source -- without this flag, that source build silently skips compiling
# the C++ ops extension, confirmed via a real "No module named 'mmcv._ext'"
# failure the first time this ran (the rest of the package installed fine,
# just missing the compiled ops NMS/RoIAlign/etc. depend on).
export MMCV_WITH_OPS=1
mim install "mmcv>=2.0.0,<2.1.0"
# mmdet range constrained by MMOCR itself -- confirmed via a real
# "MMDetection 3.3.0 is incompatible with MMOCR 1.0.1" assertion. An
# unpinned/latest mmdet resolves past what MMOCR 1.0.1 actually supports.
mim install "mmdet>=3.0.0,<3.2.0"
# Pinned to the exact version the mmdet<3.2.0 constraint above was verified
# against -- a later mmocr release could have a different compatible mmdet
# range, silently reintroducing this same class of conflict.
mim install "mmocr==1.0.1"

echo "== Installing ONNX Runtime (validation) + misc =="
pip install onnx onnxruntime psutil

echo "== Cloning MMDeploy (for the export tool + deployment configs) if not already present =="
if [ ! -d "$MMDEPLOY_SRC_DIR" ]; then
    git clone --depth 1 https://github.com/open-mmlab/mmdeploy.git "$MMDEPLOY_SRC_DIR"
fi
# MMDeploy's own setup requires setuptools<81, but the editable install below
# (pip install -e) needs setuptools>=64 for PEP 660's build_editable hook --
# pinning "<81" alone let pip pick something older than that and hit a real
# "build backend is missing the 'build_editable' hook" failure. Narrow range
# satisfies both constraints.
pip install "setuptools>=64,<81" wheel
# --no-build-isolation: pip's DEFAULT build isolation creates a SEPARATE
# temporary environment for the actual build step, with its own freshly
# resolved setuptools -- completely ignoring the pin above, confirmed via a
# real "ModuleNotFoundError: No module named 'pkg_resources'" failure
# inside that isolated build env (a newer setuptools that doesn't bundle
# pkg_resources by default). Disabling isolation forces the build to use
# our already-correctly-pinned venv setuptools instead. Requires every
# build-time dependency to already be installed in this venv (torch, mmcv,
# mmdet, mmocr, etc. above already cover MMDeploy's requirements).
pip install --no-build-isolation -e "$MMDEPLOY_SRC_DIR"

# PyTorch 2.6+ weights_only default rejects OpenMMLab checkpoints' pickled
# objects (e.g. HistoryBuffer) with "Weights only load failed" -- these
# checkpoints are from OpenMMLab's own official release (trusted source),
# so forcing the legacy full-unpickling load behavior is appropriate.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd "$MMDEPLOY_SRC_DIR"

# ---------------------------------------------------------------------------
# Detection: DBNet-ResNet18
# ---------------------------------------------------------------------------
echo "== Downloading DBNet config + checkpoint (mim resolves the current filenames) =="
mim download mmocr --config dbnet_resnet18_fpnc_1200e_icdar2015 --dest .

DET_CFG=$(find . -maxdepth 1 -iname "dbnet_resnet18_fpnc_1200e_icdar2015*.py" | head -n 1)
DET_CKPT=$(find . -maxdepth 1 -iname "dbnet_resnet18_fpnc_1200e_icdar2015*.pth" | head -n 1)
if [ -z "$DET_CFG" ] || [ -z "$DET_CKPT" ]; then
    echo "ERROR: mim download did not produce the expected DBNet config/checkpoint. Contents of CWD:" >&2
    ls -la >&2
    exit 1
fi
echo "Detection config: $DET_CFG, checkpoint: $DET_CKPT"

echo "== Exporting DBNet to ONNX via MMDeploy (ONNX Runtime, dynamic shape) =="
python tools/deploy.py \
    configs/mmocr/text-detection/text-detection_onnxruntime_dynamic.py \
    "$DET_CFG" \
    "$DET_CKPT" \
    demo/resources/text_det.jpg \
    --work-dir mmocr_det_work_dir \
    --device cpu

if [ ! -s mmocr_det_work_dir/end2end.onnx ]; then
    echo "ERROR: MMDeploy did not produce mmocr_det_work_dir/end2end.onnx -- scroll up for its actual output/error." >&2
    exit 1
fi
cp mmocr_det_work_dir/end2end.onnx "../$OUT_DIR/mmocr_dbnet_det.onnx"
echo "wrote $OUT_DIR/mmocr_dbnet_det.onnx"

# ---------------------------------------------------------------------------
# Recognition: CRNN
# ---------------------------------------------------------------------------
echo "== Downloading CRNN config + checkpoint (mim resolves the current filenames) =="
mim download mmocr --config crnn_mini-vgg_5e_mj --dest .

REC_CFG=$(find . -maxdepth 1 -iname "crnn_mini-vgg_5e_mj*.py" | head -n 1)
REC_CKPT=$(find . -maxdepth 1 -iname "crnn_mini-vgg_5e_mj*.pth" | head -n 1)
if [ -z "$REC_CFG" ] || [ -z "$REC_CKPT" ]; then
    echo "ERROR: mim download did not produce the expected CRNN config/checkpoint. Contents of CWD:" >&2
    ls -la >&2
    exit 1
fi
echo "Recognition config: $REC_CFG, checkpoint: $REC_CKPT"

echo "== Exporting CRNN to ONNX via MMDeploy (ONNX Runtime, dynamic shape) =="
python tools/deploy.py \
    configs/mmocr/text-recognition/text-recognition_onnxruntime_dynamic.py \
    "$REC_CFG" \
    "$REC_CKPT" \
    demo/resources/text_recog.jpg \
    --work-dir mmocr_rec_work_dir \
    --device cpu

if [ ! -s mmocr_rec_work_dir/end2end.onnx ]; then
    echo "ERROR: MMDeploy did not produce mmocr_rec_work_dir/end2end.onnx -- scroll up for its actual output/error." >&2
    exit 1
fi
cp mmocr_rec_work_dir/end2end.onnx "../$OUT_DIR/mmocr_crnn_rec.onnx"
echo "wrote $OUT_DIR/mmocr_crnn_rec.onnx"

cd ..
echo ""
echo "== Done. Exported: =="
ls -la "$OUT_DIR"/mmocr_*.onnx

echo ""
echo "Next:"
echo "  1. Copy into the Android app's assets:"
echo "     cp $OUT_DIR/mmocr_dbnet_det.onnx $OUT_DIR/mmocr_crnn_rec.onnx \\"
echo "        ../android-benchmark-app/app/src/main/assets/models/ocr/"
echo "  2. Add benchmark_config.json entries -- TWO configs (runtime: onnx),"
echo "     input sizes matching whatever DBNet/CRNN's dynamic export accepts"
echo "     (start with the same 640x640 / 320x48 pattern used for PaddleOCR"
echo "     and adjust if ONNX Runtime complains about the shape)."
echo "  3. CTC dictionary for CRNN decode is not saved by this script --"
echo "     see the note at the top of this file before wiring up real"
echo "     output parsing."