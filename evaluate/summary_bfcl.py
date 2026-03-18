import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize BFCL results and export unified metrics."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/results/BFCL/",
        help="BFCL results root directory (default: results/BFCL/).",
    )
    parser.add_argument(
        "--output_excel",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/results/BFCL/summary_bfcl.xlsx",
        help="Path to output Excel file.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional path to output CSV file.",
    )
    return parser.parse_args()


def _warn(message: str) -> None:
    print(f"[WARN] {message}")


def _safe_float(value: object) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return math.nan

    try:
        if text.endswith("%"):
            return float(text[:-1].strip()) / 100.0
        return float(text)
    except Exception:  # noqa: BLE001
        return math.nan


def _normalize_metric_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = normalized.replace("%", "pct")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace("/", "_")
    normalized = normalized.replace("(", "")
    normalized = normalized.replace(")", "")
    normalized = normalized.replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _extract_run_context(scores_dir: str, results_dir: str) -> Dict[str, str]:
    rel_dir = os.path.relpath(scores_dir, results_dir)
    parts = rel_dir.split(os.sep)
    model_family = parts[0] if len(parts) >= 1 else ""
    run_name = parts[1] if len(parts) >= 2 else ""
    return {"model_family": model_family, "run_name": run_name}


def _build_row(
    *,
    model_family: str,
    run_name: str,
    model_name: str,
    metric_group: str,
    metric_name: str,
    accuracy: float,
    correct_count: Optional[float],
    num_samples: Optional[float],
    source_file: str,
) -> Dict[str, object]:
    return {
        "model_family": model_family,
        "run_name": run_name,
        "model_name": model_name,
        "metric_group": metric_group,
        "metric_name": metric_name,
        "accuracy": accuracy,
        "correct_count": correct_count if correct_count is not None else math.nan,
        "num_samples": num_samples if num_samples is not None else math.nan,
        "source_file": source_file,
    }


def _read_csv_records(path: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    try:
        with open(path, "r", encoding="utf8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({k: (v if v is not None else "") for k, v in row.items()})
    except Exception as exc:  # noqa: BLE001
        _warn(f"Failed to read CSV {path}: {exc}")
        return []
    return records


def _parse_overall_csv(scores_dir: str, results_dir: str) -> List[Dict[str, object]]:
    path = os.path.join(scores_dir, "data_overall.csv")
    if not os.path.isfile(path):
        _warn(f"Missing file: {path}")
        return []

    records = _read_csv_records(path)
    if not records:
        _warn(f"Empty CSV: {path}")
        return []

    ctx = _extract_run_context(scores_dir, results_dir)
    source_file = os.path.relpath(path, results_dir)
    rows: List[Dict[str, object]] = []

    for rec in records:
        model_name = str(rec.get("Model", "")).strip()
        if not model_name:
            model_name = f"{ctx['model_family']}/{ctx['run_name']}"

        acc = _safe_float(rec.get("Overall Acc"))
        if not math.isnan(acc):
            rows.append(
                _build_row(
                    model_family=ctx["model_family"],
                    run_name=ctx["run_name"],
                    model_name=model_name,
                    metric_group="overall",
                    metric_name="overall_acc",
                    accuracy=acc,
                    correct_count=None,
                    num_samples=None,
                    source_file=source_file,
                )
            )

    return rows


def _parse_multi_turn_csv(scores_dir: str, results_dir: str) -> List[Dict[str, object]]:
    path = os.path.join(scores_dir, "data_multi_turn.csv")
    if not os.path.isfile(path):
        _warn(f"Missing file: {path}")
        return []

    records = _read_csv_records(path)
    if not records:
        _warn(f"Empty CSV: {path}")
        return []

    ctx = _extract_run_context(scores_dir, results_dir)
    source_file = os.path.relpath(path, results_dir)
    rows: List[Dict[str, object]] = []

    metric_columns = []
    if records:
        metric_columns = [
            c
            for c in records[0].keys()
            if c not in {"Rank", "Model"} and not c.lower().endswith("rank")
        ]
    for rec in records:
        model_name = str(rec.get("Model", "")).strip()
        if not model_name:
            model_name = f"{ctx['model_family']}/{ctx['run_name']}"

        for col in metric_columns:
            acc = _safe_float(rec.get(col))
            if math.isnan(acc):
                continue
            rows.append(
                _build_row(
                    model_family=ctx["model_family"],
                    run_name=ctx["run_name"],
                    model_name=model_name,
                    metric_group="multi_turn",
                    metric_name=_normalize_metric_name(col),
                    accuracy=acc,
                    correct_count=None,
                    num_samples=None,
                    source_file=source_file,
                )
            )

    return rows


def _metric_name_from_score_file(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    if base.endswith("_score"):
        base = base[: -len("_score")]
    if base.startswith("BFCL_v4_"):
        base = base[len("BFCL_v4_") :]
    return _normalize_metric_name(base)


def _parse_score_json(path: str, results_dir: str) -> Optional[Dict[str, object]]:
    try:
        with open(path, "r", encoding="utf8") as f:
            first_line = f.readline().strip()
    except Exception as exc:  # noqa: BLE001
        _warn(f"Failed to read score file {path}: {exc}")
        return None

    if not first_line:
        _warn(f"Empty score file: {path}")
        return None

    try:
        header = json.loads(first_line)
    except Exception as exc:  # noqa: BLE001
        _warn(f"Failed to parse first JSON line in {path}: {exc}")
        return None

    rel_path = os.path.relpath(path, results_dir)
    parts = rel_path.split(os.sep)
    model_family = parts[0] if len(parts) >= 1 else ""
    run_name = parts[1] if len(parts) >= 2 else ""

    model_name = str(header.get("model_name", "")).strip()
    if not model_name and len(parts) >= 4:
        model_name = parts[3]
    if not model_name:
        model_name = f"{model_family}/{run_name}"

    accuracy = _safe_float(header.get("accuracy"))
    correct_count = _safe_float(header.get("correct_count"))
    num_samples = _safe_float(header.get("total_count"))

    if math.isnan(accuracy) and not math.isnan(correct_count) and not math.isnan(num_samples) and num_samples > 0:
        accuracy = correct_count / num_samples

    return _build_row(
        model_family=model_family,
        run_name=run_name,
        model_name=model_name,
        metric_group="score_json",
        metric_name=_metric_name_from_score_file(path),
        accuracy=accuracy,
        correct_count=correct_count if not math.isnan(correct_count) else None,
        num_samples=num_samples if not math.isnan(num_samples) else None,
        source_file=rel_path,
    )


def _collect_score_json_rows(scores_dir: str, results_dir: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for root, _, files in os.walk(scores_dir):
        for name in files:
            if not name.endswith("_score.json"):
                continue
            full_path = os.path.join(root, name)
            row = _parse_score_json(full_path, results_dir)
            if row is not None:
                rows.append(row)
    return rows


def collect_results(results_dir: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not os.path.isdir(results_dir):
        print(f"Results directory does not exist: {results_dir}")
        return rows

    score_dirs: List[str] = []
    for root, dirnames, _ in os.walk(results_dir):
        for dirname in dirnames:
            if dirname == "scores":
                score_dirs.append(os.path.join(root, dirname))

    if not score_dirs:
        _warn(f"No scores directories found in: {results_dir}")
        return rows

    score_dirs.sort()
    for scores_dir in score_dirs:
        rows.extend(_parse_overall_csv(scores_dir, results_dir))
        rows.extend(_parse_multi_turn_csv(scores_dir, results_dir))
        rows.extend(_collect_score_json_rows(scores_dir, results_dir))

    rows.sort(
        key=lambda x: (
            str(x["model_family"]),
            str(x["run_name"]),
            str(x["model_name"]),
            str(x["metric_group"]),
            str(x["metric_name"]),
            str(x["source_file"]),
        )
    )
    return rows


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_csv(path: str, rows: List[Dict[str, object]], columns: List[str]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows = collect_results(args.results_dir)
    if not rows:
        print("No BFCL metrics found. Nothing to summarize.")
        return

    columns = [
        "model_family",
        "run_name",
        "model_name",
        "metric_group",
        "metric_name",
        "accuracy",
        "correct_count",
        "num_samples",
        "source_file",
    ]
    if pd is not None:
        df = pd.DataFrame(rows, columns=columns)
        _ensure_parent(args.output_excel)
        df.to_excel(args.output_excel, index=False)
        print(f"Summary written to {args.output_excel}")
        if args.output_csv:
            _ensure_parent(args.output_csv)
            df.to_csv(args.output_csv, index=False)
            print(f"CSV summary written to {args.output_csv}")
    else:
        _warn("pandas is not installed, skip Excel export.")
        csv_path = args.output_csv
        if not csv_path:
            base, _ = os.path.splitext(args.output_excel)
            csv_path = f"{base}.csv"
            _warn(f"--output_csv not set, fallback to: {csv_path}")
        _write_csv(csv_path, rows, columns)
        print(f"CSV summary written to {csv_path}")


if __name__ == "__main__":
    main()
