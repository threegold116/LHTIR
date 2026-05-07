#!/usr/bin/env python3
# Copyright 2026
"""
Parse ToolAgentLoop ``instance_id finished`` lines from training logs and relate
step index to per-step generation length (``step_length_list`` token counts).

Only trajectories that emit a finished line are counted. If training is killed
mid-rollout, those instances may be missing. Ray may append
``[repeated Nx across cluster]`` to lines; we strip it before parsing. If
aggregated counts look too low vs expected rollouts, try ``RAY_DEDUP_LOGS=0``
on future runs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_LOG = (
    "/ssd/project/LHTIR/checkpoints/qwen3-4b_ftrl_multiturn/"
    "qwen3-4b-2507_ftrl_multiturn-no_kl_no_ent-n_8-step_3200/2026-04-05_16-14-44.log"
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RAY_REPEATED_RE = re.compile(r"\s*\[repeated\s+\d+x\s+across\s+cluster\]\s*$")

# instance_id finished: 228, response_length: 1420, assistant_turns: 1, step_length_list: [1420], ...
FINISHED_RE = re.compile(
    r"instance_id finished:\s*(\d+),\s*"
    r"response_length:\s*(\d+),\s*"
    r"assistant_turns:\s*(\d+),\s*"
    r"step_length_list:\s*(\[[^\]]*\])"
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def strip_ray_suffix(text: str) -> str:
    return RAY_REPEATED_RE.sub("", text)


@dataclass(frozen=True)
class FinishedRecord:
    instance_id: int
    response_length: int
    assistant_turns: int
    step_length_list: list[int]


def parse_finished_line(line: str) -> FinishedRecord | None:
    raw = strip_ray_suffix(strip_ansi(line))
    m = FINISHED_RE.search(raw)
    if not m:
        return None
    instance_id = int(m.group(1))
    response_length = int(m.group(2))
    assistant_turns = int(m.group(3))
    try:
        lst = ast.literal_eval(m.group(4))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(lst, list) or not lst:
        return None
    if not all(isinstance(x, int) for x in lst):
        return None
    return FinishedRecord(
        instance_id=instance_id,
        response_length=response_length,
        assistant_turns=assistant_turns,
        step_length_list=lst,
    )


def iter_log_lines(path: Path, max_lines: int | None) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            yield line


def load_records_single_pass(path: Path, max_lines: int | None) -> tuple[list[FinishedRecord], int]:
    """Single pass: keep last record per instance_id; duplicate_hits = extra lines for same id."""
    last_by_id: dict[int, FinishedRecord] = {}
    duplicate_hits = 0
    for line in iter_log_lines(path, max_lines):
        rec = parse_finished_line(line)
        if rec is None:
            continue
        if rec.instance_id in last_by_id:
            duplicate_hits += 1
        last_by_id[rec.instance_id] = rec
    records = sorted(last_by_id.values(), key=lambda r: r.instance_id)
    return records, duplicate_hits


def aggregate_by_step(records: list[FinishedRecord]) -> dict[int, list[int]]:
    by_step: dict[int, list[int]] = defaultdict(list)
    for rec in records:
        for k, length in enumerate(rec.step_length_list):
            by_step[k].append(length)
    return by_step


def step_stats(lengths: list[int]) -> dict[str, float | int]:
    n = len(lengths)
    if n == 0:
        return {"count": 0}
    return {
        "count": n,
        "mean": statistics.mean(lengths),
        "std": statistics.stdev(lengths) if n > 1 else 0.0,
        "min": min(lengths),
        "max": max(lengths),
        "median": statistics.median(lengths),
    }


def print_summary(records: list[FinishedRecord], by_step: dict[int, list[int]], one_based: bool) -> None:
    by_step_sorted = sorted(by_step.items(), key=lambda x: x[0])
    print("\nby_step_index (length = tokens generated in that assistant step):")
    for k, lengths in by_step_sorted:
        idx = k + 1 if one_based else k
        st = step_stats(lengths)
        if st["count"] == 0:
            continue
        print(
            f"  step_{idx}: count={st['count']} mean={st['mean']:.2f} std={st['std']:.2f} "
            f"min={st['min']} max={st['max']} median={st['median']:.2f}"
        )


def write_csv_per_instance(path: Path, records: list[FinishedRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "instance_id",
                "response_length",
                "assistant_turns",
                "num_steps",
                "step_length_list",
            ]
        )
        for rec in records:
            w.writerow(
                [
                    rec.instance_id,
                    rec.response_length,
                    rec.assistant_turns,
                    len(rec.step_length_list),
                    repr(rec.step_length_list),
                ]
            )


def write_csv_by_step(path: Path, by_step: dict[int, list[int]], one_based: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["step_index", "count", "mean", "std", "min", "max", "median"]
        )
        for k in sorted(by_step.keys()):
            lengths = by_step[k]
            st = step_stats(lengths)
            if st["count"] == 0:
                continue
            idx = k + 1 if one_based else k
            w.writerow(
                [
                    idx,
                    st["count"],
                    f"{st['mean']:.6f}",
                    f"{st['std']:.6f}",
                    st["min"],
                    st["max"],
                    f"{st['median']:.6f}",
                ]
            )


def try_plot(by_step: dict[int, list[int]], out_path: Path | None, one_based: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip --plot", file=sys.stderr)
        return
    if not by_step:
        return
    ks = sorted(by_step.keys())
    means = [statistics.mean(by_step[k]) for k in ks]
    stds = [statistics.stdev(by_step[k]) if len(by_step[k]) > 1 else 0.0 for k in ks]
    xs = [k + 1 if one_based else k for k in ks]
    plt.figure(figsize=(8, 4))
    plt.errorbar(xs, means, yerr=stds, fmt="o-", capsize=3)
    plt.xlabel("step_index" + (" (1-based)" if one_based else " (0-based)"))
    plt.ylabel("mean step length (tokens)")
    plt.title("Per-step generation length vs step index")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path is None and not os.environ.get("DISPLAY"):
        out_path = Path("step_length_vs_step.png")
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"wrote plot: {out_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=str, default=DEFAULT_LOG, help="Path to training .log file.")
    p.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Read at most this many lines (for smoke tests).",
    )
    p.add_argument(
        "--one-based-step",
        action="store_true",
        help="Display step index starting at 1 in text and CSV step_index column.",
    )
    p.add_argument(
        "--csv-per-instance",
        type=str,
        default=None,
        help="Write per-instance CSV to this path.",
    )
    p.add_argument(
        "--csv-by-step",
        type=str,
        default=None,
        help="Write per-step aggregate CSV to this path.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Plot mean±std vs step (requires matplotlib).",
    )
    p.add_argument(
        "--plot-path",
        type=str,
        default=None,
        help="Save plot to this file (png). If --plot without this, uses pyplot.show().",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.log)
    if not path.is_file():
        print(f"error: log file not found: {path}", file=sys.stderr)
        sys.exit(1)
    records, dup_same_id = load_records_single_pass(path, args.max_lines)
    by_step = aggregate_by_step(records)
    print(f"parsed_finished_unique_instances: {len(records)}")
    print(f"duplicate_finished_lines_same_instance_id: {dup_same_id} (kept last)")
    print_summary(records, by_step, args.one_based_step)
    if args.csv_per_instance:
        write_csv_per_instance(Path(args.csv_per_instance), records)
        print(f"wrote: {args.csv_per_instance}")
    if args.csv_by_step:
        write_csv_by_step(Path(args.csv_by_step), by_step, args.one_based_step)
        print(f"wrote: {args.csv_by_step}")
    if args.plot:
        plot_out = Path(args.plot_path) if args.plot_path else None
        try_plot(by_step, plot_out, args.one_based_step)


if __name__ == "__main__":
    main()
