#!/usr/bin/env python3
"""
Exports a trained YOLOv8 checkpoint to TFLite (fp32 and int8) for on-device benchmarking.
Requires: pip install ultralytics --break-system-packages

Usage:
    python export_yolov8.py --weights runs/train/exp/weights/best.pt \
        --imgsz 640 --out ../models/detector

int8 export requires a small representative image set for calibration --
point --calib-dir at ~100-200 sample panel crops for realistic quantization.
"""
import argparse
import shutil
from pathlib import Path


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", required=True)
    parser.add_argument("--calib-dir", default=None, help="Image dir for int8 calibration")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    # fp32 TFLite
    fp32_path = model.export(format="tflite", imgsz=args.imgsz, int8=False)
    shutil.copy(fp32_path, out_dir / "yolov8n_fp32.tflite")
    print(f"Wrote {out_dir / 'yolov8n_fp32.tflite'}")

    # int8 TFLite (needs calibration data for reasonable accuracy)
    if args.calib_dir:
        int8_path = model.export(format="tflite", imgsz=args.imgsz, int8=True, data=args.calib_dir)
        shutil.copy(int8_path, out_dir / "yolov8n_int8.tflite")
        print(f"Wrote {out_dir / 'yolov8n_int8.tflite'}")
    else:
        print("Skipped int8 export -- pass --calib-dir with representative panel images.")


if __name__ == "__main__":
    main()
