"""
Compare two pipeline_benchmark.py reports (baseline vs modified) and print the
per-phase delta so a faster_whisper change can be judged a speedup/regression.

Usage:
    python benchmark/compare.py benchmark/results/baseline.json benchmark/results/modified.json
    python benchmark/compare.py --threshold 1.5 baseline.json modified.json
"""

import argparse
import json


PHASES = ["decode", "vad", "extract", "forward", "transcribe", "total"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_seconds(value: float) -> str:
    return f"{value:9.2f}s"


def fmt_delta(pct: float) -> str:
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def pct_delta(modified: float, baseline: float) -> float:
    """Percent change of modified vs baseline, safe for zero baselines.

    Short clips round fast phases (decode/vad/extract) to 0.00 s, which used
    to crash the comparison with ZeroDivisionError. Both-zero means no
    measurable change; zero-to-nonzero is reported as +inf.
    """
    if baseline == 0:
        return 0.0 if modified == 0 else float("inf")
    return (modified - baseline) / baseline * 100.0


def verdict(median_pct: float, threshold: float) -> str:
    if median_pct < -threshold:
        return "SPEEDUP"
    if median_pct > threshold:
        return "REGRESSION"
    return "~noise"


def main():
    parser = argparse.ArgumentParser(description="Compare two pipeline_benchmark reports.")
    parser.add_argument("baseline", help="Path to the baseline JSON report.")
    parser.add_argument("modified", help="Path to the modified JSON report.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Absolute percent change below which a result is considered noise.",
    )
    args = parser.parse_args()

    base = load(args.baseline)
    mod = load(args.modified)

    print(f"Baseline : commit {base.get('commit')}  ({base.get('timestamp', '')})")
    print(f"Modified : commit {mod.get('commit')}  ({mod.get('timestamp', '')})")
    print()

    base_params, mod_params = base["params"], mod["params"]
    changed = {
        k for k in base_params.keys() & mod_params.keys()
        if base_params[k] != mod_params[k]
    }
    if changed:
        print("WARNING: runs used different parameters:", sorted(changed))
        print()

    print(
        f"{'Phase':<12}{'baseline min/med':>22}{'modified min/med':>22}"
        f"{'med delta':>12}  verdict"
    )
    print("-" * 72)

    for phase in PHASES:
        if phase not in base["phases"] or phase not in mod["phases"]:
            continue
        b_min, b_med = base["phases"][phase]["min"], base["phases"][phase]["median"]
        m_min, m_med = mod["phases"][phase]["min"], mod["phases"][phase]["median"]
        delta_med = pct_delta(m_med, b_med)
        print(
            f"{phase:<12}"
            f"{f'{fmt_seconds(b_min)} / {fmt_seconds(b_med)}':>22}"
            f"{f'{fmt_seconds(m_min)} / {fmt_seconds(m_med)}':>22}"
            f"{fmt_delta(delta_med):>12}  {verdict(delta_med, args.threshold)}"
        )

    base_total, mod_total = base["phases"]["total"], mod["phases"]["total"]
    delta = pct_delta(mod_total["median"], base_total["median"])
    print("-" * 72)

    base_dur = base["audio"]["duration_seconds"]
    mod_dur = mod["audio"]["duration_seconds"]
    if base_dur != mod_dur:
        print(f"NOTE: audio duration differs (base={base_dur:.0f}s, mod={mod_dur:.0f}s)")
    else:
        base_rtf = base_dur / base_total["median"]
        mod_rtf = mod_dur / mod_total["median"]
        rtf_delta = (mod_rtf - base_rtf) / base_rtf * 100.0
        print(
            f"Realtime factor: baseline {base_rtf:.2f}x -> modified {mod_rtf:.2f}x "
            f"({fmt_delta(rtf_delta)})"
        )

    if base.get("gpu") and mod.get("gpu"):
        print("GPU max      : "
              f"util {base['gpu'].get('util_max')}->{mod['gpu'].get('util_max')}%  "
              f"power {base['gpu'].get('power_max_w')}->{mod['gpu'].get('power_max_w')}W  "
              f"temp {base['gpu'].get('temp_max_c')}->{mod['gpu'].get('temp_max_c')}C")


if __name__ == "__main__":
    main()
