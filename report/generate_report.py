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
        pss_baseline_mb = round(data.get("pss_baseline_kb", 0) / 1024, 1)
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
            "pss_baseline_mb": pss_baseline_mb,
            "pss_after_load_mb": round(data.get("pss_after_load_kb", 0) / 1024, 1),
            "pss_peak_mb": pss_peak_mb,
            # Isolates this model's OWN memory cost from the shared overhead of
            # bundling three runtimes (TFLite+GPU, ONNX Runtime, ML Kit) in this
            # benchmark harness -- the absolute pss_peak_mb (checked against the
            # 200MB budget below) is what matters for "will this fit on device";
            # this delta is what matters for "which candidate costs more than
            # another," a question the absolute number can't answer when every
            # config's absolute peak is dominated by the same shared baseline.
            "pss_delta_mb": round(pss_peak_mb - pss_baseline_mb, 1) if pss_peak_mb > 0 and pss_baseline_mb > 0 else -1.0,
            # Memory cost specifically at model load time, before any inference runs --
            # this is where a model's weight buffers actually get allocated, so it
            # correlates with model_size_mb far more directly than pss_delta_mb does
            # for small models (1-10MB), where peak-during-inference barely differs
            # from after-load and both can round to ~0 against a large shared baseline.
            # load_delta_mb: guard against the -1 sentinel Kotlin writes for
            # pss_after_load_kb when a config errors before/during load() --
            # without this check, a failed config's -1 got treated as ~0.0 MB
            # and subtracted from baseline, producing a large nonsensical
            # negative number instead of the proper "N/A" -1.0 sentinel every
            # other column already uses for missing data.
            "load_delta_mb": (
                round(round(data.get("pss_after_load_kb", 0) / 1024, 1) - pss_baseline_mb, 1)
                if pss_baseline_mb > 0 and data.get("pss_after_load_kb", -1) >= 0
                else -1.0
            ),
            # RSS (from /proc/self/status) alongside PSS as a diagnostic --
            # PSS via ActivityManager showed baseline/after-load/peak as
            # byte-for-byte identical in a real run even with multi-sample
            # delays added, consistent with that cross-process Binder IPC
            # query being throttled/cached at the OS level. RSS is a
            # same-process file read, not subject to the same throttling --
            # compare rss_load_delta_mb against load_delta_mb to see whether
            # RSS actually shows real signal where PSS didn't.
            "rss_baseline_mb": round(data.get("rss_baseline_kb", -1) / 1024, 1) if data.get("rss_baseline_kb", -1) >= 0 else -1.0,
            "rss_after_load_mb": round(data.get("rss_after_load_kb", -1) / 1024, 1) if data.get("rss_after_load_kb", -1) >= 0 else -1.0,
            "rss_peak_mb": round(data.get("rss_peak_during_inference_kb", -1) / 1024, 1) if data.get("rss_peak_during_inference_kb", -1) >= 0 else -1.0,
            "rss_load_delta_mb": (
                round(data.get("rss_after_load_kb", 0) / 1024 - data.get("rss_baseline_kb", 0) / 1024, 1)
                if data.get("rss_baseline_kb", -1) >= 0 and data.get("rss_after_load_kb", -1) >= 0
                else -1.0
            ),
            "within_budget": "PASS" if 0 < pss_peak_mb <= budget_mb else "FAIL",
            # Real-world confirmed finding (3 repeat runs, ML Kit: 360-411MB
            # consistently): PSS via ActivityManager was shown to be stale/
            # cached -- byte-for-byte IDENTICAL across three unrelated configs
            # in one session -- while RSS showed genuine, repeatable movement.
            # This is a SEPARATE, likely more trustworthy budget verdict based
            # on absolute RSS peak (not a delta, which is itself noisy early
            # in a process's life from zygote copy-on-write settling) -- check
            # this one, not just within_budget, before trusting any PASS.
            "within_budget_rss": (
                "PASS" if 0 < data.get("rss_peak_during_inference_kb", -1) / 1024 <= budget_mb else "FAIL"
            ) if data.get("rss_peak_during_inference_kb", -1) >= 0 else "N/A",
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
        error = data.get("error", "")[:200] if data.get("error") else ""
        peak_mb = round(data.get("pss_peak_overall_kb", 0) / 1024, 1)
        baseline_mb = round(data.get("pss_baseline_kb", 0) / 1024, 1)
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
            "pss_baseline_mb": baseline_mb,
            "pss_after_detector_load_mb": after_detector_load_mb,
            "pss_after_detector_release_mb": after_detector_release_mb,
            "detector_mem_reclaimed_mb": round(after_detector_load_mb - after_detector_release_mb, 1),
            "pss_peak_overall_mb": peak_mb,
            # Same isolation logic as load_results()'s pss_delta_mb -- this
            # pipeline's own memory cost above the shared three-runtime baseline.
            "pss_delta_mb": round(peak_mb - baseline_mb, 1) if peak_mb > 0 and baseline_mb > 0 else -1.0,
            # A pipeline that errored is never a budget PASS, regardless of what its
            # (likely baseline-only, pre-failure) peak memory number happens to show.
            "within_budget": "FAIL" if error else ("PASS" if 0 < peak_mb <= budget_mb else "FAIL"),
            "error": error,
        })
    return pd.DataFrame(rows)


def render_html(df: "pd.DataFrame", pipeline_df: "pd.DataFrame", budget_mb: float) -> str:
    device_label = df["device"].iloc[0] if not df.empty else "unknown"
    soc_label = df["soc"].iloc[0] if not df.empty else "unknown"

    detector_df = df[df["task"] == "detector"].sort_values("latency_p50_ms")
    ocr_df = df[df["task"] == "ocr"].sort_values("latency_p50_ms")
    errors_df = df[df["error"] != ""]
    over_budget_df = df[(df["within_budget"] == "FAIL") & (df["error"] == "")]
    over_budget_rss_df = df[(df["within_budget_rss"] == "FAIL") & (df["error"] == "")]

    pipeline_over_budget = pipeline_df[
        (pipeline_df["within_budget"] == "FAIL") & (pipeline_df["error"] == "")
    ] if not pipeline_df.empty else pipeline_df

    def table_html(d):
        cols = ["config", "runtime", "requested_delegate", "actual_delegate",
                "model_size_mb", "load_time_ms", "latency_p50_ms", "latency_p90_ms",
                "latency_p99_ms", "rss_peak_mb", "within_budget_rss", "pss_peak_mb", "within_budget"]
        styled = d[cols].copy()
        return styled.to_html(index=False, border=0, classes="results-table", escape=False,
                               formatters={
                                   "within_budget": lambda v:
                                       f'<span class="{"pass" if v == "PASS" else "fail"}">{v}</span>',
                                   "within_budget_rss": lambda v:
                                       f'<span class="{"pass" if v == "PASS" else "fail"}">{v}</span>',
                               })

    def pipeline_table_html(d):
        cols = ["pipeline", "detector", "ocr", "iterations", "detector_p50_ms", "ocr_p50_ms",
                "end_to_end_p50_ms", "pss_peak_overall_mb", "pss_delta_mb",
                "detector_mem_reclaimed_mb", "within_budget"]
        styled = d[cols].copy()
        return styled.to_html(index=False, border=0, classes="results-table", escape=False,
                               formatters={"within_budget": lambda v:
                                   f'<span class="{"pass" if v == "PASS" else "fail"}">{v}</span>'})

    pipeline_section = ""
    if not pipeline_df.empty:
        pipeline_errors_df = pipeline_df[pipeline_df["error"] != ""]
        ok_pipeline_df = pipeline_df[pipeline_df["error"] == ""]
        pipeline_section = f"""
  <h2>Sequential pipeline runs (detector &rarr; OCR, one process lifetime)</h2>
  {pipeline_table_html(ok_pipeline_df.sort_values("pss_peak_overall_mb")) if not ok_pipeline_df.empty else "<p>No successful pipeline runs.</p>"}
  <div class="note">
    <strong>pss_peak_overall_mb</strong> is the number that matters for the 200MB budget --
    it's the true worst-case memory seen across the whole detector-then-OCR cycle, not each
    stage's isolated peak. <strong>pss_delta_mb</strong> isolates this pipeline's own cost above
    its fresh-process baseline, for comparing candidates against each other rather than checking
    device fit. <strong>detector_mem_reclaimed_mb</strong> shows how much memory
    actually came back after calling release() on the detector, before the OCR model loaded --
    a small or negative value here means native/delegate memory isn't being freed promptly,
    which is worth investigating in the real app even if the pipeline still passes budget.
    {"<br><br><strong>" + str(len(pipeline_over_budget)) + " pipeline(s) exceed the memory budget</strong> when run end-to-end, even if their isolated stage numbers looked fine individually." if not pipeline_over_budget.empty else ""}
    <br><br>
    <strong>CAVEAT:</strong> the pipeline numbers above are PSS-only. RSS diagnostics
    (added after this section was originally built) confirmed PSS via ActivityManager can be
    stale/cached -- identical across unrelated configs in one session. RSS wasn't yet threaded
    through the pipeline path, only the isolated-config path, so treat every PASS/budget number
    in this pipeline table with the same skepticism the isolated tables' within_budget_rss
    column exists to address, until RSS is added here too.
  </div>
  {"<h2 class='errors'>Failed pipelines (crashed / errored mid-run)</h2>" + pipeline_errors_df[["pipeline", "detector", "ocr", "error"]].to_html(index=False, border=0) if not pipeline_errors_df.empty else ""}
  {"<div class='note'>A failed pipeline's latency/memory columns above (when shown at all) reflect only whatever completed before the error, not a real end-to-end measurement -- treat these as broken, not as data points.</div>" if not pipeline_errors_df.empty else ""}
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
  .budget-banner-critical {{ background: #fde8e8; border: 1px solid #e57373; padding: 10px 14px;
                     border-radius: 6px; margin-bottom: 20px; font-size: 14px; }}
</style>
</head>
<body>
  <h1>Panel Benchmark Report</h1>
  <div class="meta">Device: {device_label} &nbsp;|&nbsp; SoC: {soc_label} &nbsp;|&nbsp; {len(df)} isolated configs, {len(pipeline_df)} pipelines</div>
  <div class="{"budget-banner-critical" if not over_budget_rss_df.empty else "budget-banner"}">
    Target device profile: Android 9.0+, Snapdragon 845-era (2018+), 3GB RAM, CPU/GPU only.
    Memory budget: <strong>{budget_mb:.0f} MB</strong>.
    <strong>within_budget</strong> (PSS via ActivityManager) --
    {"<strong>" + str(len(over_budget_df)) + " config(s) exceed this</strong>." if not over_budget_df.empty else "all configs within budget."}
    <strong>within_budget_rss</strong> (RSS, confirmed via repeat runs to be more reliable --
    PSS was found identical across unrelated configs in one session, i.e. stale/cached) --
    {"<strong>" + str(len(over_budget_rss_df)) + " config(s) exceed this</strong>, including some that PASS on the PSS check -- trust this verdict over within_budget." if not over_budget_rss_df.empty else "all configs within budget."}
  </div>

  <h2>Detector models (isolated)</h2>
  {table_html(detector_df) if not detector_df.empty else "<p>No detector results.</p>"}
  <div class="note">
    Sorted by p50 latency, fastest first. Top row is the current leader (subject to passing the memory budget).
    <strong>pss_peak_mb</strong> is the absolute number checked against the 200MB budget.
    <strong>pss_delta_mb</strong> (peak minus this config's own fresh-process baseline) isolates
    this model's own memory cost from the ~176MB shared overhead of bundling three runtimes
    (TFLite+GPU, ONNX Runtime, ML Kit) in this benchmark harness -- use pss_delta_mb to compare
    candidates against each other, and pss_peak_mb to check device fit. For small models
    (a few MB), pss_delta_mb often rounds to ~0 since peak-during-inference barely exceeds
    the after-load footprint -- <strong>load_delta_mb</strong> (memory right after load(),
    before any inference) is usually the more useful number for small models: it isolates
    exactly the cost of mapping the model's weights into memory, which scales with
    model_size_mb far more visibly than pss_delta_mb does at this size range.
    Your real production app will ship only the one winning runtime, so its actual baseline
    will be far lower than this harness's -- pss_peak_mb here is a conservative
    (over-)estimate for that reason, not a final number.
    <br><br>
    <strong>rss_load_delta_mb</strong> is an alternate load-time delta computed from RSS
    (/proc/self/status) instead of PSS -- included because PSS's cross-process
    ActivityManager query was observed showing baseline/after-load/peak as identical even
    with multi-sample delays added, suggesting that query may be throttled/cached at the OS
    level rather than genuinely reflecting no change. If rss_load_delta_mb shows real,
    model-size-correlated values where load_delta_mb doesn't, that confirms the PSS query
    itself is the limitation, not the models' actual memory behavior.
  </div>

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