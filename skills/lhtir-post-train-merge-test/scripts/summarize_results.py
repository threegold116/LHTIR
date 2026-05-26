#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def read_toolhop_metric(path: Path) -> tuple[int, float]:
    values = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                values.append(float(json.loads(line)["metrics"]["answer_correctness"]))
    return len(values), (sum(values) / len(values) if values else 0.0)


def read_bfcl_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolhop-dir", required=True)
    parser.add_argument("--toolhop-prefix", required=True)
    parser.add_argument("--bfcl-dir", required=True)
    args = parser.parse_args()

    toolhop_dir = Path(args.toolhop_dir)
    bfcl_dir = Path(args.bfcl_dir)

    toolhop_scores = {}
    for scenario in ["Free", "Direct", "Mandatory"]:
        path = toolhop_dir / f"{args.toolhop_prefix}-{scenario}-vllm-4096.jsonl"
        count, acc = read_toolhop_metric(path)
        toolhop_scores[scenario] = (count, acc)

    overall = sum(v[1] for v in toolhop_scores.values()) / len(toolhop_scores)

    overall_rows = read_bfcl_csv(bfcl_dir / "scores" / "data_overall.csv")
    mt_rows = read_bfcl_csv(bfcl_dir / "scores" / "data_multi_turn.csv")

    print("ToolHop")
    for scenario, (count, acc) in toolhop_scores.items():
        print(f"{scenario}: count={count} acc={acc:.6f}")
    print(f"Overall: {overall:.6f}")

    print("\nBFCL")
    if overall_rows:
        row = overall_rows[0]
        print(f"Overall Acc: {row.get('Overall Acc', '')}")
        print(f"Multi Turn Acc: {row.get('Multi Turn Acc', '')}")
    if mt_rows:
        row = mt_rows[0]
        for key in ["Multi Turn Overall Acc", "Base", "Miss Func", "Miss Param", "Long Context"]:
            if key in row:
                print(f"{key}: {row[key]}")


if __name__ == "__main__":
    main()
