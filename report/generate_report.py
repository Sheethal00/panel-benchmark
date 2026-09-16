#!/usr/bin/env python3
"""
Aggregates all per-config result JSON files pulled from the device into a single
comparison report (HTML + CSV) so you can eyeball latency/memory/size trade-offs
across every model candidate in one place.

Usage:
    python generate_report.py --results-dir ../results/run_2026-09-16 --out ../results/report.html
"""
import argparse
import json
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    raise SystemExit("This script needs pandas. Install with: pip install pandas --break-system-packages")

# Target device profile: Android 9.0+ / Snapdragon 845-era / 3GB RAM / CPU or GPU only.
# 200MB is the TOTAL app memory budget on that device tier, not per-model -- if your
# real app loads a detector + OCR model concurrently, budget each model's peak PSS
# against a fraction of 200MB, not the full amount. Override with --budget-mb.
DEFAULT_MEMORY_BUDGET_MB = 200


def load_results(results_dir: Path, budget_mb: float) -> "pd.DataFrame":
    rows = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name.startswith("pipeline_"):
            continue  # handled separately by load_pipeline_results
        data = json.loads(f.read_text())
        pss_peak_mb = round(data.get("pss_peak_during_inference_kb", 0) / 1024, 1)
        rows.append({
            "config": data.get("config_name"),
            "task": data.get("task"),
            "runtime": data.get("runtime"),
            "requested_delegate": data.get("requested_delegate"),
            "actual_delegate": data.get("actual_delegate_info"),
            "model_size_mb": round(data.get("model_size_bytes", 0) / (1024 * 1024), 2),
            "load_time_ms": data.get("load_time_ms"),
            "latency_p50_ms": round(data.get("latency_p50_ms", -1), 2),
            "latency_p90_ms": round(data.get("latency_p90_ms", -1), 2),
            "latency_p99_ms": round(data.get("latency_p99_ms", -1), 2),
            "pss_after_load_mb": round(data.get("pss_after_load_kb", 0) / 1024, 1),
            "pss_peak_mb": pss_peak_mb,
            "within_budget": "PASS" if 0 < pss_peak_mb <= budget_mb else "FAIL",
            "device": data.get("device_model"),
            "soc": data.get("soc"),
            "error": data.get("error", "")[:120] if data.get("error") else "",
        })
    if not rows:
        raise SystemExit(f"No isolated-config result JSON files found in {results_dir}")
    return pd.DataFrame(rows)


def load_pipeline_results(results_dir: Path, budget_mb: float) -> "pd.DataFrame":
    rows = []
    for f in sorted(results_dir.glob("pipeline_*.json")):
        data = json.loads(f.read_text())
        peak_mb = round(data.get("pss_peak_overall_kb", 0) / 1024, 1)
        after_detector_release_mb = round(data.get("pss_after_detector_release_kb", 0) / 1024, 1)
        after_detector_load_mb = round(data.get("pss_after_detector_load_kb", 0) / 1024, 1)
        rows.append({
            "pipeline": data.get("pipeline_name"),
            "detector": data.get("detector_config"),
            "ocr": data.get("ocr_config"),
            "iterations": data.get("iterations"),
            "detector_p50_ms": round(data.get("detector_latency_p50_ms", -1), 2),
            "ocr_p50_ms": round(data.get("ocr_latency_p50_ms", -1), 2),
            "end_to_end_p50_ms": round(data.get("end_to_end_p50_ms", -1), 2),
            "pss_after_detector_load_mb": after_detector_load_mb,
            "pss_after_detector_release_mb": after_detector_release_mb,
            "detector_mem_reclaimed_mb": round(after_detector_load_mb - after_detector_release_mb, 1),
            "pss_peak_overall_mb": peak_mb,
            "within_budget": "PASS" if 0 < peak_mb <= budget_mb else "FAIL",
            "error": data.get("error", "")[:120] if data.get("error") else "",
        })
    return pd.DataFrame(rows)


def render_html(df: "pd.DataFrame", pipeline_df: "pd.DataFrame", budget_mb: float) -> str:
    device_label = df["device"].iloc[0] if not df.empty else "unknown"
    soc_label = df["soc"].iloc[0] if not df.empty else "unknown"

    detector_df = df[df["task"] == "detector"].sort_values("latency_p50_ms")
    ocr_df = df[df["task"] == "ocr"].sort_values("latency_p50_ms")
    errors_df = df[df["error"] != ""]
    over_budget_df = df[(df["within_budget"] == "FAIL") & (df["error"] == "")]

    pipeline_over_budget = pipeline_df[
        (pipeline_df["within_budget"] == "FAIL") & (pipeline_df["error"] == "")
    ] if not pipeline_df.empty else pipeline_df

    def table_html(d):
        cols = ["config", "runtime", "requested_delegate", "actual_delegate",
                "model_size_mb", "load_time_ms", "latency_p50_ms", "latency_p90_ms",
                "latency_p99_ms", "pss_after_load_mb", "pss_peak_mb", "within_budget"]
        styled = d[cols].copy()
        return styled.to_html(index=False, border=0, classes="results-table", escape=False,
                               formatters={"within_budget": lambda v:
                                   f'<span class="{"pass" if v == "PASS" else "fail"}">{v}</span>'})

    def pipeline_table_html(d):
        cols = ["pipeline", "detector", "ocr", "iterations", "detector_p50_ms", "ocr_p50_ms",
                "end_to_end_p50_ms", "pss_after_detector_load_mb", "pss_after_detector_release_mb",
                "detector_mem_reclaimed_mb", "pss_peak_overall_mb", "within_budget"]
        styled = d[cols].copy()
        return styled.to_html(index=False, border=0, classes="results-table", escape=False,
                               formatters={"within_budget": lambda v:
                                   f'<span class="{"pass" if v == "PASS" else "fail"}">{v}</span>'})

    pipeline_section = ""
    if not pipeline_df.empty:
        pipeline_section = f"""
  <h2>Sequential pipeline runs (detector &rarr; OCR, one process lifetime)</h2>
  {pipeline_table_html(pipeline_df.sort_values("pss_peak_overall_mb"))}
  <div class="note">
    <strong>pss_peak_overall_mb</strong> is the number that matters for the 200MB budget --
    it's the true worst-case memory seen across the whole detector-then-OCR cycle, not each
    stage's isolated peak. <strong>detector_mem_reclaimed_mb</strong> shows how much memory
    actually came back after calling release() on the detector, before the OCR model loaded --
    a small or negative value here means native/delegate memory isn't being freed promptly,
    which is worth investigating in the real app even if the pipeline still passes budget.
    {"<br><br><strong>" + str(len(pipeline_over_budget)) + " pipeline(s) exceed the memory budget</strong> when run end-to-end, even if their isolated stage numbers looked fine individually." if not pipeline_over_budget.empty else ""}
  </div>
"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Panel Benchmark Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 32px; color: #1a1a1a; }}
  h1 {{ margin-bottom: 4px; }}
  .meta {{ color: #666; margin-bottom: 24px; }}
  h2 {{ margin-top: 40px; border-bottom: 2px solid #eee; padding-bottom: 6px; }}
  table.results-table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 14px; }}
  table.results-table th {{ text-align: left; background: #f5f5f5; padding: 8px 10px; }}
  table.results-table td {{ padding: 8px 10px; border-top: 1px solid #eee; }}
  table.results-table tr:first-child td {{ font-weight: 600; background: #f0f8f0; }}
  .errors {{ color: #b00020; }}
  .note {{ font-size: 13px; color: #666; margin-top: 8px; }}
  .pass {{ color: #0a7d2c; font-weight: 600; }}
  .fail {{ color: #b00020; font-weight: 600; }}
  .budget-banner {{ background: #fff8e1; border: 1px solid #f0d878; padding: 10px 14px;
                     border-radius: 6px; margin-bottom: 20px; font-size: 14px; }}
</style>
</head>
<body>
  <h1>Panel Benchmark Report</h1>
  <div class="meta">Device: {device_label} &nbsp;|&nbsp; SoC: {soc_label} &nbsp;|&nbsp; {len(df)} isolated configs, {len(pipeline_df)} pipelines</div>
  <div class="budget-banner">
    Target device profile: Android 9.0+, Snapdragon 845-era (2018+), 3GB RAM, CPU/GPU only.
    Memory budget: <strong>{budget_mb:.0f} MB</strong> peak PSS.
    {"<strong>" + str(len(over_budget_df)) + " isolated config(s) exceed this budget</strong> -- see FAIL rows below." if not over_budget_df.empty else "All isolated configs are within budget."}
  </div>

  <h2>Detector models (isolated)</h2>
  {table_html(detector_df) if not detector_df.empty else "<p>No detector results.</p>"}
  <div class="note">Sorted by p50 latency, fastest first. Top row is the current leader (subject to passing the memory budget).</div>

  <h2>OCR models (isolated)</h2>
  {table_html(ocr_df) if not ocr_df.empty else "<p>No OCR results.</p>"}
  <div class="note">Sorted by p50 latency, fastest first.</div>
{pipeline_section}
  {"<h2 class='errors'>Failed configs (crashed / errored)</h2>" + errors_df[["config", "error"]].to_html(index=False, border=0) if not errors_df.empty else ""}

  <div class="note">
    Reminder: check "actual_delegate" against "requested_delegate" -- NNAPI/GPU delegates
    silently fall back to CPU on unsupported devices, which will make a candidate look
    artificially slow if you only glance at the delegate column you asked for. This matters
    especially on Snapdragon 845-era devices where GPU delegate op coverage is more limited
    than on newer chipsets.
    <br><br>
    Isolated config numbers above use force-stop-between-runs, so they show each model's
    memory footprint in complete isolation. The pipeline section is the more realistic
    number for your production app's actual worst case, since detector and OCR release/load
    happen within one continuous process lifetime there, matching how the real app behaves.
  </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", default=None, help="Output HTML path (default: <results-dir>/report.html)")
    parser.add_argument("--csv", default=None, help="Also write a flat CSV (default: <results-dir>/report.csv)")
    parser.add_argument("--budget-mb", type=float, default=DEFAULT_MEMORY_BUDGET_MB,
                         help=f"Peak PSS memory budget per model in MB (default: {DEFAULT_MEMORY_BUDGET_MB})")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_path = Path(args.out) if args.out else results_dir / "report.html"
    csv_path = Path(args.csv) if args.csv else results_dir / "report.csv"

    df = load_results(results_dir, args.budget_mb)
    pipeline_df = load_pipeline_results(results_dir, args.budget_mb)
    df.to_csv(csv_path, index=False)
    if not pipeline_df.empty:
        pipeline_csv_path = csv_path.with_name(csv_path.stem + "_pipelines.csv")
        pipeline_df.to_csv(pipeline_csv_path, index=False)
    out_path.write_text(render_html(df, pipeline_df, args.budget_mb))

    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")
    n_fail = ((df["within_budget"] == "FAIL") & (df["error"] == "")).sum()
    print(f"\n{len(df)} isolated configs summarized ({(df['error'] != '').sum()} errors, {n_fail} over the {args.budget_mb:.0f}MB memory budget).")
    if not pipeline_df.empty:
        n_pipeline_fail = ((pipeline_df["within_budget"] == "FAIL") & (pipeline_df["error"] == "")).sum()
        print(f"{len(pipeline_df)} pipeline runs summarized ({n_pipeline_fail} over budget end-to-end).")


if __name__ == "__main__":
    main()
