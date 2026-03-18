import argparse
import json
import math
import os
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize accuracy for JSONL result files and export to Excel."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/results/",
        help=(
            "Directory containing JSONL result files "
            "(default: results/ToolHop/Qwen3-4B)."
        ),
    )
    parser.add_argument(
        "--output_excel",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/results/summary.xlsx",
        help=(
            "Path to the output Excel file "
            "(default: results/ToolHop/Qwen3-4B/summary.xlsx)."
        ),
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".jsonl",
        help="Only files ending with this suffix will be processed (default: .jsonl).",
    )
    return parser.parse_args()


def compute_file_accuracy(path: str, base_dir: str) -> Dict[str, object]:
    total = 0
    correct = 0.0

    try:
        with open(path, "r", encoding="utf8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:  # noqa: BLE001
                    print(f"Failed to parse JSON in {path} line {lineno}: {e}")
                    continue

                metrics = obj.get("metrics") or {}
                val = metrics.get("answer_correctness")
                if isinstance(val, (int, float)):
                    total += 1
                    correct += float(val)
    except FileNotFoundError:
        print(f"File not found: {path}")

    rel_path = os.path.relpath(path, base_dir)
    parts = rel_path.split(os.sep)
    dataset = parts[0] if len(parts) >= 1 else ""
    model_name = parts[1] if len(parts) >= 2 else ""
    if total > 0:
        acc = correct / total
    else:
        acc = math.nan

    return {
        "file": rel_path,
        "dataset": dataset,
        "model_name": model_name,
        "accuracy": acc,
        "num_samples": total,
    }


def collect_results(results_dir: str, suffix: str) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    if not os.path.isdir(results_dir):
        print(f"Results directory does not exist: {results_dir}")
        return results

    for root, _, files in os.walk(results_dir):
        for name in files:
            if not name.endswith(suffix):
                continue
            full_path = os.path.join(root, name)
            stats = compute_file_accuracy(full_path, results_dir)
            results.append(stats)

    results.sort(key=lambda x: x["file"])
    return results


def main() -> None:
    args = parse_args()
    rows = collect_results(args.results_dir, args.suffix)

    if not rows:
        print("No result files found. Nothing to summarize.")
        return

    df = pd.DataFrame(
        rows,
        columns=["file", "dataset", "model_name", "accuracy", "num_samples"],
    )

    os.makedirs(os.path.dirname(args.output_excel), exist_ok=True)
    df.to_excel(args.output_excel, index=False)
    print(f"Summary written to {args.output_excel}")


if __name__ == "__main__":
    main()

