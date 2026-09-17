#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Exports MMOCR's DBNet-ResNet18 (text detection) and CRNN (text
# recognition) -- OpenMMLab/mmocr, Apache-2.0 -- to ONNX via the OFFICIAL
# MMDeploy toolchain (not a third-party pre-exported copy).
#
# Usage:
#   ./export_mmocr.sh
#
# This script creates and activates its OWN dedicated venv
# (panel-mmocr-venv), same reasoning as export_picodet.sh/export_rtmdet.sh.
#
# THIS IS THE SECOND ATTEMPT at the full toolchain. The first hit SEVEN
# distinct real environment failures in a row before finally identifying
# the actual root cause of the last one (mmcv's compiled C++ ops extension
# not building even with the documented MMCV_WITH_OPS=1 flag):
#
#   1. setuptools<81 needed (MMDeploy's own requirement)
#   2. ...but pip install -e needs setuptools>=64 for the build_editable
#      hook -- narrowed to a range satisfying both
#   3. pip's build ISOLATION creates a separate environment for the actual
#      build step with its own independently-resolved setuptools, ignoring
#      the pin above entirely -- fixed with --no-build-isolation for the
#      editable install specifically
#   4. mmcv, needing to build from source (no prebuilt wheel matches recent
#      torch releases), hit the SAME isolated-setuptools problem via
#      `mim install` (which doesn't expose --no-build-isolation) --
#      generalized the fix with PIP_CONSTRAINT, which pip DOES apply to
#      isolated build environments
#   5. mmdet's installed version (latest) is incompatible with mmocr 1.0.1
#      -- pinned mmdet<3.2.0
#   6. mmdet 3.1.x's compatible mmcv range is narrower still -- pinned
#      mmcv<2.1.0
#   7. mmcv (still installed via `mim install`, i.e. still pip under the
#      hood) built successfully but WITHOUT its compiled ops extension
#      (`ModuleNotFoundError: No module named 'mmcv._ext'`) even with
#      MMCV_WITH_OPS=1 exported -- because arbitrary env vars, unlike
#      PIP_CONSTRAINT, are not guaranteed to reach pip's isolated build
#      subprocess the way the mmcv build script expects to read them.
#
# THE ACTUAL FIX (this version): stop routing mmcv through `pip`/`mim`
# entirely. Build it the way mmcv's own documentation describes for exactly
# this situation (https://mmcv.readthedocs.io/en/latest/get_started/build.html):
# clone the source and run `python setup.py install` directly. This runs in
# the current interpreter with no isolated subprocess at all, so
# MMCV_WITH_OPS=1 is guaranteed to be read correctly.
#
# IMPORTANT, read before wiring up real inference (not just benchmarking):
# - DBNet's output is a probability MAP, not boxes directly -- needs
#   threshold + contour-finding post-processing.
# - CRNN's output is a CTC character-probability sequence -- needs CTC
#   decoding with a character dictionary. This script does not currently
#   save that dictionary as a standalone file (MMOCR bundles it in
#   config/checkpoint metadata) -- add that separately before wiring up
#   real decode.
# - Both models use DYNAMIC input shapes in this export (MMDeploy's
#   documented onnxruntime_dynamic configs) -- ONNX Runtime handles this
#   natively.
# ---------------------------------------------------------------------------

OUT_DIR="../models/ocr"
MMDEPLOY_SRC_DIR="./.mmdeploy-src"
MMCV_SRC_DIR="./.mmcv-src"
mkdir -p "$OUT_DIR"

VENV_DIR="./panel-mmocr-venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "== Creating dedicated venv for MMOCR export: $VENV_DIR =="
    python3 -m venv "$VENV_DIR"
fi
echo "== Activating $VENV_DIR =="
source "$VENV_DIR/bin/activate"

# PIP_CONSTRAINT applies to pip's isolated build environments (for anything
# still built via pip/mim below, e.g. mmdeploy's editable install) --
# unlike a plain `pip install "setuptools<81"`, which does NOT reach those.
cat > /tmp/panel_mmocr_pip_constraints.txt << 'CONSTRAINTS'
setuptools>=64,<81
CONSTRAINTS
export PIP_CONSTRAINT=/tmp/panel_mmocr_pip_constraints.txt

echo "== Installing PyTorch (CPU) =="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "== Installing mmengine =="
pip install openmim
mim install mmengine

echo "== Building mmcv FROM SOURCE via setup.py (not pip/mim) =="
# This is the actual fix for the 7th failure above: MMCV_WITH_OPS=1 only
# reliably takes effect when mmcv's setup.py runs directly in this
# interpreter, not inside pip's isolated build subprocess. v2.0.1 pinned
# to match the mmdet<3.2.0 / mmcv<2.1.0 compatibility chain discovered
# during the first attempt.
if [ ! -d "$MMCV_SRC_DIR" ]; then
    git clone --branch v2.0.1 --depth 1 https://github.com/open-mmlab/mmcv.git "$MMCV_SRC_DIR"
fi
pip install -r "$MMCV_SRC_DIR/requirements/runtime.txt"
(
    cd "$MMCV_SRC_DIR"
    export MMCV_WITH_OPS=1
    python setup.py install
)
python3 -c "import mmcv; import mmcv._ext; print('mmcv._ext OK, mmcv version:', mmcv.__version__)"

echo "== Installing mmdet + mmocr (pinned to the compatibility chain above) =="
mim install "mmdet>=3.0.0,<3.2.0"
mim install "mmocr==1.0.1"

echo "== Installing ONNX Runtime (validation) + misc =="
pip install onnx onnxruntime psutil

echo "== Cloning MMDeploy (for the export tool + deployment configs) if not already present =="
if [ ! -d "$MMDEPLOY_SRC_DIR" ]; then
    git clone --depth 1 https://github.com/open-mmlab/mmdeploy.git "$MMDEPLOY_SRC_DIR"
fi
# --no-build-isolation: this editable install specifically needs to see
# this venv's own (correctly ranged) setuptools, not a fresh isolated one.
pip install --no-build-isolation -e "$MMDEPLOY_SRC_DIR"

# PyTorch 2.6+ defaults torch.load() to weights_only=True, which rejects
# OpenMMLab checkpoints' pickled objects -- these are from OpenMMLab's own
# official release, so forcing legacy full-unpickling load is appropriate.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd "$MMDEPLOY_SRC_DIR"

# ---------------------------------------------------------------------------
# Detection: DBNet-ResNet18
# ---------------------------------------------------------------------------
echo "== Downloading DBNet config + checkpoint (official OpenMMLab release) =="
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
echo "== Downloading CRNN config + checkpoint (official OpenMMLab release) =="
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
echo "  2. benchmark_config.json already has mmocr_dbnet_det_onnx_cpu --"
echo "     add a mmocr_crnn_rec_onnx_cpu entry (runtime: onnx, input"
echo "     320x48, matching the pattern used for PaddleOCR's recognition"
echo "     config) since this version produces the recognition model too."
echo "  3. CTC dictionary for CRNN decode is not saved by this script --"
echo "     see the note at the top of this file before wiring up real"
echo "     output parsing."