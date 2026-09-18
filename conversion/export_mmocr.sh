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
# THIRD ATTEMPT at getting this environment right, and finally the simple
# one -- worth knowing why the first two weren't:
#
# Attempt 1 hit SEVEN distinct real environment failures in a row
# (setuptools/build-isolation conflicts, cascading mmocr->mmdet->mmcv
# version constraints, and finally mmcv's compiled ops extension not
# building even with the documented MMCV_WITH_OPS=1 flag). Attempt 2 fixed
# that last failure by building mmcv from source via `python setup.py
# install` directly (bypassing pip's build isolation, which was silently
# dropping MMCV_WITH_OPS=1) -- pinned to mmcv v2.0.1, which doesn't have a
# prebuilt wheel for recent torch releases, hence needing a source build
# at all.
#
# The actual fix, confirmed against MMOCR's own official install docs
# (https://mmocr.readthedocs.io/en/dev-1.x/get_started/install.html) and a
# real successful run (`tools/infer.py` producing real OCR output, then a
# real successful ONNX export): mmcv==2.1.0 -- ONE version newer than what
# attempt 2 used -- has a prebuilt wheel available, so it never needs to
# build from source in the first place. All that setup.py/MMCV_WITH_OPS
# machinery in attempt 2 was solving a problem specific to picking an
# older mmcv version than necessary, not a problem with `pip`/`mim`
# installs generally. Simple `mim install "mmcv==2.1.0"` is sufficient.
#
# The mmdet<3.2.0 / mmocr==1.0.1 pins from attempt 1 remain correct --
# confirmed exactly matching MMOCR's own documented compatibility table
# (mmengine 0.7.1-1.1.0, mmcv 2.0.0rc4-2.1.0, mmdet 3.0.0rc5-3.2.0 for
# both mmocr 1.0.1 and dev-1.x).
#
# The PIP_CONSTRAINT / --no-build-isolation / TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
# fixes below are still needed -- those are for MMDeploy specifically
# (the separate ONNX-export tool, not covered by MMOCR's own install docs
# at all), not for the mmcv version issue this attempt fixes.
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
mkdir -p "$OUT_DIR"

VENV_DIR="./panel-mmocr-venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "== Creating dedicated venv for MMOCR export: $VENV_DIR =="
    python3 -m venv "$VENV_DIR"
fi
echo "== Activating $VENV_DIR =="
source "$VENV_DIR/bin/activate"

# PIP_CONSTRAINT applies to pip's isolated build environments (needed for
# MMDeploy's editable install below) -- unlike a plain
# `pip install "setuptools<81"`, which does NOT reach those.
cat > /tmp/panel_mmocr_pip_constraints.txt << 'CONSTRAINTS'
setuptools>=64,<81
CONSTRAINTS
export PIP_CONSTRAINT=/tmp/panel_mmocr_pip_constraints.txt

echo "== Installing PyTorch (CPU) =="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "== Installing OpenMMLab stack via mim =="
pip install openmim
mim install mmengine
# mmcv==2.1.0: has a prebuilt wheel for recent torch releases (2.0.1 does
# not, confirmed the hard way) -- see the header note above. Also the top
# of MMOCR's own documented compatible range (2.0.0rc4 <= mmcv < 2.1.0 per
# the docs table, though 2.1.0 itself was confirmed working in practice).
mim install "mmcv==2.1.0"
# mmdet/mmocr range confirmed against MMOCR's own compatibility table.
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