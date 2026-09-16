#!/usr/bin/env python3
"""
Generic post-training dynamic quantization for an existing ONNX model, using onnxruntime's
quantization tools. Useful for OCR models (e.g. a CRNN) that you've exported to ONNX but
not yet quantized.

Requires: pip install onnxruntime onnx --break-system-packages

Usage:
    python quantize_int8.py --model ../models/ocr/crnn_mobile_fp32.onnx \
        --out ../models/ocr/crnn_mobile_int8.onnx
"""
import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    quantize_dynamic(
        model_input=args.model,
        model_output=args.out,
        weight_type=QuantType.QInt8,
    )
    print(f"Wrote {args.out}")
    print("NOTE: dynamic quantization only quantizes weights, not activations -- "
          "for max speedup on-device, consider static quantization with a calibration "
          "set if this isn't fast enough.")


if __name__ == "__main__":
    main()
