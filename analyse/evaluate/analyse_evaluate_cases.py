"""
检查 results/ 下评测产物（BFCL 等）的逐条 case 质量。

*_result.json / *_score.json 在本仓库中通常为 JSONL（每行一个 JSON 对象），
与 analyse/rollout/analyse_rollout_cases.py 的用法类似，可通过 CLI 或 Notebook 调用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union


def iter_jsonl(path: Union[str, Path]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """按行解析 JSONL，产出 (行号, 对象)。空行跳过；解析失败打印警告并跳过。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error in {path} line {lineno}: {e}")
                continue
            if not isinstance(obj, dict):
                print(f"[WARN] {path} line {lineno}: expected object, got {type(obj).__name__}")
                continue
            yield lineno, obj


def find_eval_files(root: Union[str, Path], suffix: str) -> List[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob(f"*{suffix}") if p.is_file())


def summarize_record_structure(
    records: Iterable[Dict[str, Any]],
    max_records: int = 100,
) -> Dict[str, Any]:
    keys_count: Dict[str, int] = {}
    example_by_key: Dict[str, Any] = {}
    total = 0
    for rec in records:
        total += 1
        for k, v in rec.items():
            keys_count[k] = keys_count.get(k, 0) + 1
            if k not in example_by_key:
                example_by_key[k] = v
        if total >= max_records:
            break
    return {
        "sampled_records": total,
        "keys_count": keys_count,
        "example_by_key": example_by_key,
    }


def _is_empty_result(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _is_aggregate_score_row(rec: Dict[str, Any]) -> bool:
    """BFCL score 文件首行常为整体指标，无 per-case id。"""
    if "id" in rec:
        return False
    if "accuracy" in rec and "total_count" in rec:
        return True
    return False


def check_result_cases(
    path: Union[str, Path],
    records: List[Dict[str, Any]],
    *,
    id_key: str = "id",
    result_key: str = "result",
) -> Dict[str, Any]:
    """检查模型输出结果文件（如 *_result.json）。"""
    total = len(records)
    missing_id = 0
    empty_result = 0
    duplicate_ids: List[str] = []
    seen: Set[str] = set()
    zero_in_tokens = 0
    zero_out_tokens = 0
    missing_latency = 0

    for rec in records:
        rid = rec.get(id_key)
        if rid is None or (isinstance(rid, str) and not rid.strip()):
            missing_id += 1
        else:
            sid = str(rid)
            if sid in seen:
                duplicate_ids.append(sid)
            seen.add(sid)
        if _is_empty_result(rec.get(result_key)):
            empty_result += 1
        ic = rec.get("input_token_count")
        oc = rec.get("output_token_count")
        if ic == 0:
            zero_in_tokens += 1
        if oc == 0:
            zero_out_tokens += 1
        if "latency" not in rec:
            missing_latency += 1

    return {
        "file": str(path),
        "kind": "result",
        "total_records": total,
        "unique_ids": len(seen),
        "missing_id": missing_id,
        "empty_result": empty_result,
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_id_examples": duplicate_ids[:20],
        "input_token_count_eq_0": zero_in_tokens,
        "output_token_count_eq_0": zero_out_tokens,
        "missing_latency_field": missing_latency,
    }


def check_score_cases(
    path: Union[str, Path],
    records: List[Dict[str, Any]],
    *,
    id_key: str = "id",
    valid_key: str = "valid",
) -> Dict[str, Any]:
    """检查打分文件（如 *_score.json）中的逐条 case（跳过首行 aggregate）。"""
    case_rows = [r for r in records if not _is_aggregate_score_row(r)]
    total = len(case_rows)
    missing_id = 0
    valid_true = 0
    valid_false = 0
    missing_valid = 0
    duplicate_ids: List[str] = []
    seen: Set[str] = set()

    for rec in case_rows:
        rid = rec.get(id_key)
        if rid is None or (isinstance(rid, str) and not rid.strip()):
            missing_id += 1
        else:
            sid = str(rid)
            if sid in seen:
                duplicate_ids.append(sid)
            seen.add(sid)
        if valid_key not in rec:
            missing_valid += 1
        elif rec[valid_key]:
            valid_true += 1
        else:
            valid_false += 1

    return {
        "file": str(path),
        "kind": "score",
        "total_case_rows": total,
        "skipped_aggregate_rows": len(records) - total,
        "unique_ids": len(seen),
        "missing_id": missing_id,
        "valid_true": valid_true,
        "valid_false": valid_false,
        "missing_valid_field": missing_valid,
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_id_examples": duplicate_ids[:20],
    }


def load_all_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
    return [obj for _, obj in iter_jsonl(path)]


def analyse_single_file(
    path: Union[str, Path],
    *,
    kind: str = "auto",
    print_structure: bool = False,
    max_structure_records: int = 100,
) -> Dict[str, Any]:
    path = Path(path)
    records = load_all_records(path)
    print(f"=== {path} ({len(records)} records) ===")

    if print_structure and records:
        summary = summarize_record_structure(iter(records), max_records=max_structure_records)
        print("\n[STRUCTURE SUMMARY]")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    resolved = kind
    if kind == "auto":
        name = path.name.lower()
        if name.endswith("_score.json"):
            resolved = "score"
        elif name.endswith("_result.json"):
            resolved = "result"
        elif records and _is_aggregate_score_row(records[0]) and any("valid" in r for r in records[1:3]):
            resolved = "score"
        else:
            resolved = "result"

    if resolved == "score":
        stats = check_score_cases(path, records)
    else:
        stats = check_result_cases(path, records)

    print("\n[CASE CHECK]")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print()
    return stats


def analyse_eval_root(
    root: Union[str, Path],
    *,
    suffix: str = "_result.json",
    kind: str = "auto",
    print_structure: bool = False,
    max_structure_records: int = 100,
) -> List[Dict[str, Any]]:
    root = Path(root)
    files = find_eval_files(root, suffix)
    if not files:
        print(f"No files matching *{suffix} under {root}")
        return []

    print(f"Found {len(files)} files under {root} (suffix={suffix})\n")
    out: List[Dict[str, Any]] = []
    for path in files:
        out.append(
            analyse_single_file(
                path,
                kind=kind,
                print_structure=print_structure,
                max_structure_records=max_structure_records,
            )
        )
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检查 results 下评测 JSONL（*_result.json / *_score.json）的 case 完整性。",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/results",
        help="评测结果根目录。",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="只分析单个文件（与 --root 互斥使用时优先）。",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_result.json",
        help="在 --root 下递归匹配的文件后缀，例如 _score.json。",
    )
    parser.add_argument(
        "--kind",
        type=str,
        choices=("auto", "result", "score"),
        default="auto",
        help="auto：根据文件名推断；result/score：强制按对应规则检查。",
    )
    parser.add_argument(
        "--print-structure",
        action="store_true",
        help="打印前 N 条记录的字段与示例值。",
    )
    parser.add_argument(
        "--max-structure-records",
        type=int,
        default=100,
        help="--print-structure 时最多采样条数。",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.file:
        analyse_single_file(
            args.file,
            kind=args.kind,
            print_structure=args.print_structure,
            max_structure_records=args.max_structure_records,
        )
    else:
        analyse_eval_root(
            args.root,
            suffix=args.suffix,
            kind=args.kind,
            print_structure=args.print_structure,
            max_structure_records=args.max_structure_records,
        )


if __name__ == "__main__":
    main()
