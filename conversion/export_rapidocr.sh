#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Gets RapidOCR's current default detection+recognition pair -- PP-OCRv6
# "small" -- as ready ONNX files. No toolchain, no venv: this is a straight
# download, same pattern as export_mmocr.sh's pre-exported detection model.
#
# LICENSING: RapidOCR's own wrapper code (the Python/C++ inference library)
# is LGPL-licensed -- NOT what this script uses. The actual model WEIGHTS
# are Baidu's PaddleOCR models, Apache-2.0, which RapidOCR just repackages
# as ready ONNX files. This script downloads the weight files directly and
# runs them through this repo's own ONNX Runtime code -- no RapidOCR code
# is incorporated, so the LGPL wrapper license doesn't apply here. Source:
# https://huggingface.co/DjB314/RapidOCR (explicitly Apache-2.0, a
# checksummed byte-for-byte mirror of RapidAI's own ModelScope release,
# not affiliated with RapidAI/PaddlePaddle -- redistributed unmodified per
# Apache-2.0 Section 4).
#
# WHY THIS CANDIDATE, GIVEN WE ALREADY HAVE A PADDLEOCR RESULT: RapidOCR's
# current default is PP-OCRv6 -- a newer PaddleOCR generation than the
# PP-OCRv3 (det) + PP-OCRv4 (rec) combination this repo already exported
# manually via export_paddleocr.sh. RapidOCR's own release notes claim
# "PP-OCRv6 delivers significantly better performance than PP-OCRv4" --
# this is a genuine different-generation comparison, not a duplicate of
# the existing paddleocr_det/rec_onnx_cpu results.
#
# Usage:
#   ./export_rapidocr.sh
#
# IMPORTANT, read before wiring up real inference (not just benchmarking):
# - Same PP-OCR family conventions as export_paddleocr.sh: detection
#   outputs a probability MAP (needs threshold + contour-finding), Ain
#   recognition outputs a CTC sequence (needs dict-based decode).
# - Input preprocessing conventions (channels/height/width) are NOT
#   assumed here -- the app's OnnxRuntime.kt reads each model's own
#   declared input shape directly (a fix made after MMOCR's CRNN turned
#   out to want grayscale/height-32 input, differently from PaddleOCR's
#   own recognizer) so this should adapt automatically even if PP-OCRv6's
#   exact preprocessing differs from v3/v4.
# ---------------------------------------------------------------------------

OUT_DIR="../models/ocr"
mkdir -p "$OUT_DIR"

MIRROR_BASE="https://huggingface.co/DjB314/RapidOCR/resolve/main/v3.9.2/onnx/PP-OCRv6"
DET_URL="$MIRROR_BASE/det/PP-OCRv6_det_small.onnx"
REC_URL="$MIRROR_BASE/rec/PP-OCRv6_rec_small.onnx"
DICT_URL="$MIRROR_BASE/rec/ppocrv6_small_dict.txt"

echo "== Downloading RapidOCR PP-OCRv6-small detection model =="
if [ ! -f "$OUT_DIR/rapidocr_v6_det.onnx" ]; then
    curl -fL -o "$OUT_DIR/rapidocr_v6_det.onnx" "$DET_URL"
fi
if [ ! -s "$OUT_DIR/rapidocr_v6_det.onnx" ]; then
    echo "ERROR: $OUT_DIR/rapidocr_v6_det.onnx is missing or empty after download." >&2
    echo "Check the URL, or browse the mirror's actual file tree if it moved:" >&2
    echo "  https://huggingface.co/DjB314/RapidOCR/tree/main/v3.9.2/onnx/PP-OCRv6/det" >&2
    exit 1
fi

echo "== Downloading RapidOCR PP-OCRv6-small recognition model =="
if [ ! -f "$OUT_DIR/rapidocr_v6_rec.onnx" ]; then
    curl -fL -o "$OUT_DIR/rapidocr_v6_rec.onnx" "$REC_URL"
fi
if [ ! -s "$OUT_DIR/rapidocr_v6_rec.onnx" ]; then
    echo "ERROR: $OUT_DIR/rapidocr_v6_rec.onnx is missing or empty after download." >&2
    echo "Check the URL, or browse the mirror's actual file tree if it moved:" >&2
    echo "  https://huggingface.co/DjB314/RapidOCR/tree/main/v3.9.2/onnx/PP-OCRv6/rec" >&2
    exit 1
fi

echo "== Downloading character dictionary (for later CTC decode, not needed for benchmarking) =="
if [ ! -f "$OUT_DIR/rapidocr_v6_dict.txt" ]; then
    if ! curl -fL -o "$OUT_DIR/rapidocr_v6_dict.txt" "$DICT_URL"; then
        echo "WARNING: dict download failed (URL may differ from the guessed path) -- not fatal for" >&2
        echo "this week's latency/memory benchmarking, but needed later for real decode. Check:" >&2
        echo "  https://huggingface.co/DjB314/RapidOCR/tree/main/v3.9.2/onnx/PP-OCRv6/rec" >&2
        rm -f "$OUT_DIR/rapidocr_v6_dict.txt"
    fi
fi

echo ""
echo "== Done. Downloaded: =="
ls -la "$OUT_DIR"/rapidocr_v6_*

echo ""
echo "Next:"
echo "  1. Copy into the Android app's assets:"
echo "     cp $OUT_DIR/rapidocr_v6_det.onnx $OUT_DIR/rapidocr_v6_rec.onnx \\"
echo "        ../android-benchmark-app/app/src/main/assets/models/ocr/"
echo "     (also rapidocr_v6_dict.txt if it downloaded successfully)"
echo "  2. Add benchmark_config.json entries -- TWO configs (runtime: onnx),"
echo "     start with the same 640x640 / 320x48 input sizes used for"
echo "     PaddleOCR's own det/rec configs; OnnxRuntime.kt will read the"
echo "     model's actual declared shape at runtime regardless."
echo "  3. Same post-processing caveats as PaddleOCR: detection needs"
echo "     threshold+contour-finding, recognition needs CTC decode with"
echo "     the character dictionary."
