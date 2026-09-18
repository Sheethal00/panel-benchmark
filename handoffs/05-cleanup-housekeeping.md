# HANDOFF: Cleanup / housekeeping

## PROJECT CONTEXT
PanelBenchmark: weeks of benchmarking across 6 detector architectures and
4 OCR candidates, settled on YOLOX-Nano + ML Kit. Several small
loose-ends accumulated along the way that don't block other work but
should get closed out. These are independent of each other -- pick
whichever, in any order.

## ITEM 1: Final consolidated report
Reports have been generated incrementally throughout the benchmarking
process (`report/generate_report.py` against various `results/run_*/`
folders), but RTMDet, PaddleOCR, MMOCR, and the CORRECTED RapidOCR
numbers (the first RapidOCR run had a false hang + a 10x-inflated
recognition latency from a one-time cold-start cost, both fixed via
verified repeat runs) may not all be in one single, final, definitive
report yet.

**To do:** run the full suite fresh
(`scripts/run_benchmark_matrix.py`) against the complete, current
`benchmark_config.json`, generate one clean report, and treat that as the
canonical record of this feasibility phase. Check `results/` for what
already exists before re-running everything from scratch -- may only
need to merge/regenerate the report rather than re-run every config on
device again.

## ITEM 2: README Google Drive link
`README.md` has a `[ADD SHARED DRIVE LINK HERE]` placeholder in the
"Model files: hosted on Google Drive, not in this repo" section --
model binaries were moved to Google Drive rather than tracked in git
(repo would otherwise be too large), but the actual shareable link was
never filled in. Trivial fix, but blocks anyone else from actually
following that setup step.

## ITEM 3: Memory metric reliability
The benchmark report currently states outright that **neither PSS nor
RSS is fully trustworthy** as a memory metric on this device:
- **PSS** (via `ActivityManager.getProcessMemoryInfo()`): confirmed
  stale/cached at the OS level -- shown byte-for-byte IDENTICAL across
  unrelated configs in the same session, even with a multi-sample
  settling delay added.
- **RSS** (via `/proc/self/status`): shows real movement, but is noisy
  early in a process's life from zygote copy-on-write page inheritance
  settling -- confirmed via repeat runs where "after-load" was
  inconsistently LOWER than "baseline," which isn't physically sensible.

Current reports rank/PASS-FAIL on PSS "for now" with RSS shown as
unverified reference data, explicitly flagged as a rough indicator, not
a precise measurement.

**To do (if picked up):** investigate a more reliable memory sampling
approach -- candidates worth trying: much longer settling windows before
sampling, averaging RSS across many repeat runs to smooth zygote-settling
noise, or a different Android API entirely (e.g. `Debug.MemoryInfo` via
`getMemoryInfo()`, which reads differently than
`ActivityManager.getProcessMemoryInfo()` and might not have the same
staleness issue -- untested, worth checking). This affects the
trustworthiness of every budget PASS/FAIL verdict across the whole
project, so worth resolving before treating any past memory numbers as
final, especially for whichever detector/OCR combination ends up
shipping.

## RELEVANT FILES
- `report/generate_report.py` -- report generation, including the
  `within_budget` / `within_budget_rss` dual-verdict logic and its
  explanatory banner text.
- `android-benchmark-app/app/src/main/java/com/panelbench/app/metrics/MemoryProfiler.kt`
  -- where PSS/RSS sampling actually happens; item 3's investigation
  would extend or replace logic here.
- `README.md` -- item 2's placeholder location.

## WHAT "DONE" LOOKS LIKE
Each item is independent:
- Item 1: one clean, current, complete report exists and is treated as
  canonical.
- Item 2: the Drive link is real and the setup instructions actually work
  for a fresh clone.
- Item 3: either a genuinely more reliable memory metric is found and
  adopted, or a documented, deliberate decision to accept current
  limitations with a clear rationale.
