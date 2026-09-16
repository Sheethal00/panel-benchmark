#!/usr/bin/env python3
"""
Drives the PanelBenchmark Android app through every config in benchmark_config.json,
one at a time, over ADB -- so the whole matrix can run unattended on a physical device
(e.g. overnight, or while you work on something else).

Usage:
    python run_benchmark_matrix.py --config ../android-benchmark-app/app/src/main/assets/benchmark_config.json \
        --package com.panelbench.app --out ../results/run_$(date +%Y-%m-%d)

Requires: adb on PATH, exactly one device/emulator connected (or pass --serial).
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

APP_PACKAGE = "com.panelbench.app"
APP_ACTIVITY = ".MainActivity"
RESULTS_DEVICE_DIR = f"/sdcard/Android/data/{APP_PACKAGE}/files/results"

POLL_INTERVAL_SEC = 2
MAX_WAIT_SEC = 180  # generous timeout per config -- large fp32 detectors can be slow to load


def adb(args, serial=None):
    cmd = ["adb"] + (["-s", serial] if serial else []) + args
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def launch_config(config_name: str, serial=None):
    adb([
        "shell", "am", "start", "-n", f"{APP_PACKAGE}/{APP_ACTIVITY}",
        "--es", "config_name", config_name,
        "--ez", "auto_run", "true",
    ], serial=serial)


def launch_pipeline(pipeline_name: str, serial=None):
    adb([
        "shell", "am", "start", "-n", f"{APP_PACKAGE}/{APP_ACTIVITY}",
        "--es", "pipeline_name", pipeline_name,
        "--ez", "auto_run", "true",
    ], serial=serial)


def result_filename_exists(filename: str, serial=None) -> bool:
    res = adb(["shell", "ls", f"{RESULTS_DEVICE_DIR}/{filename}"], serial=serial)
    return res.returncode == 0 and "No such file" not in res.stderr + res.stdout


def wait_for_file(filename: str, serial=None) -> bool:
    waited = 0
    while waited < MAX_WAIT_SEC:
        if result_filename_exists(filename, serial=serial):
            return True
        time.sleep(POLL_INTERVAL_SEC)
        waited += POLL_INTERVAL_SEC
    return False


def pull_file(filename: str, out_dir: Path, serial=None):
    device_path = f"{RESULTS_DEVICE_DIR}/{filename}"
    local_path = out_dir / filename
    adb(["pull", device_path, str(local_path)], serial=serial)
    return local_path


def force_stop(serial=None):
    adb(["shell", "am", "force-stop", APP_PACKAGE], serial=serial)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to benchmark_config.json")
    parser.add_argument("--out", required=True, help="Local directory to write pulled results")
    parser.add_argument("--serial", default=None, help="adb device serial if multiple connected")
    parser.add_argument("--package", default=APP_PACKAGE)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(args.config).read_text())
    model_configs = config["models"]
    pipeline_configs = config.get("pipelines", [])

    print(f"Found {len(model_configs)} isolated configs and {len(pipeline_configs)} pipelines to benchmark.\n")

    for i, m in enumerate(model_configs, 1):
        name = m["name"]
        print(f"[{i}/{len(model_configs)}] Running: {name} "
              f"(runtime={m['runtime']}, delegate={m.get('delegate', 'cpu')})")

        force_stop(serial=args.serial)
        time.sleep(1)
        launch_config(name, serial=args.serial)

        filename = f"{name}.json"
        if wait_for_file(filename, serial=args.serial):
            local_path = pull_file(filename, out_dir, serial=args.serial)
            print(f"    -> pulled {local_path}")
        else:
            print(f"    !! TIMEOUT waiting for {name} -- check logcat, possible OOM/crash")

        force_stop(serial=args.serial)
        time.sleep(2)  # let device settle (thermal/memory) between candidates

    for i, p in enumerate(pipeline_configs, 1):
        name = p["name"]
        print(f"[pipeline {i}/{len(pipeline_configs)}] Running: {name} "
              f"({p['detector_config']} -> {p['ocr_config']})")

        # Deliberately NOT force-stopping before this run -- the whole point of a pipeline
        # run is to measure memory across one continuous process lifetime, matching how
        # the real app will behave when it runs detector then OCR back to back.
        launch_pipeline(name, serial=args.serial)

        filename = f"pipeline_{name}.json"
        if wait_for_file(filename, serial=args.serial):
            local_path = pull_file(filename, out_dir, serial=args.serial)
            print(f"    -> pulled {local_path}")
        else:
            print(f"    !! TIMEOUT waiting for pipeline {name} -- check logcat, possible OOM/crash")

        force_stop(serial=args.serial)
        time.sleep(2)

    print(f"\nDone. Results in {out_dir}")
    print("Next: python ../report/generate_report.py --results-dir", out_dir)


if __name__ == "__main__":
    main()
