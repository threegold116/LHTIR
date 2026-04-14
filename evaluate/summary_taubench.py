import argparse
import json
import math
import os
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize tau-bench JSONL result files and export to Excel."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/results/tau-bench",
        help="Directory containing tau-bench JSONL result files.",
    )
    parser.add_argument(
        "--output_excel",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/results/summary_3.xlsx",
        help="Path to the output Excel file.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".jsonl",
        help="Only files ending with this suffix will be processed (default: .jsonl).",
    )
    return parser.parse_args()


def parse_file_name(file_name: str) -> tuple[str, str]:
    if file_name.endswith("-retail-test.jsonl"):
        return file_name[: -len("-retail-test.jsonl")], "retail-test"
    if file_name.endswith("-airline-test.jsonl"):
        return file_name[: -len("-airline-test.jsonl")], "airline-test"
    raise ValueError(f"Unsupported tau-bench result filename: {file_name}")


def compute_file_metrics(path: str, base_dir: str) -> Dict[str, object]:
    total = 0
    reward_sum = 0.0
    r_actions_sum = 0.0
    r_outputs_sum = 0.0
    r_outputs_count = 0

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj["stop_reason"] == "error":   # 暂时对有问题的case跳过
                continue
            metrics = obj["metrics"]
            reward_sum += float(metrics["reward"])
            r_actions_sum += float(metrics["r_actions"])
            if metrics["r_outputs"] is not None:
                r_outputs_sum += float(metrics["r_outputs"])
                r_outputs_count += 1
            total += 1

    rel_path = os.path.relpath(path, base_dir)
    parts = rel_path.split(os.sep)
    dataset = os.path.basename(base_dir.rstrip(os.sep))
    model_dir = parts[0]
    file_name = os.path.basename(path)
    model_tag, scenario = parse_file_name(file_name)

    return {
        "file": rel_path,
        "dataset": dataset,
        "model_dir": model_dir,
        "model_name": model_tag,
        "scenario": scenario,
        "reward": reward_sum / total if total > 0 else math.nan,
        "r_actions": r_actions_sum / total if total > 0 else math.nan,
        "r_outputs": r_outputs_sum / r_outputs_count if r_outputs_count > 0 else math.nan,
        "num_samples": total,
    }


def collect_results(results_dir: str, suffix: str) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for root, _, files in os.walk(results_dir):
        for name in files:
            if not name.endswith(suffix):
                continue
            if not (
                name.endswith("-retail-test.jsonl")
                or name.endswith("-airline-test.jsonl")
            ):
                continue
            full_path = os.path.join(root, name)
            stats = compute_file_metrics(full_path, results_dir)
            results.append(stats)

    results.sort(key=lambda x: x["file"])
    return results


def build_summary_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        key = (row["model_dir"], row["model_name"])
        if key not in grouped:
            grouped[key] = {
                "dataset": row["dataset"],
                "model_dir": row["model_dir"],
                "model_name": row["model_name"],
                "retail_reward": math.nan,
                "retail_r_actions": math.nan,
                "retail_r_outputs": math.nan,
                "retail_num_samples": math.nan,
                "airline_reward": math.nan,
                "airline_r_actions": math.nan,
                "airline_r_outputs": math.nan,
                "airline_num_samples": math.nan,
            }

        target = grouped[key]
        if row["scenario"] == "retail-test":
            target["retail_reward"] = row["reward"]
            target["retail_r_actions"] = row["r_actions"]
            target["retail_r_outputs"] = row["r_outputs"]
            target["retail_num_samples"] = row["num_samples"]
        elif row["scenario"] == "airline-test":
            target["airline_reward"] = row["reward"]
            target["airline_r_actions"] = row["r_actions"]
            target["airline_r_outputs"] = row["r_outputs"]
            target["airline_num_samples"] = row["num_samples"]
        else:
            raise ValueError(f"Unsupported scenario: {row['scenario']}")

    summary_rows = list(grouped.values())
    summary_rows.sort(key=lambda x: (x["dataset"], x["model_dir"], x["model_name"]))
    return summary_rows


def excel_column_name(index: int) -> str:
    result = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def dataframe_to_sheet_xml(df: pd.DataFrame) -> str:
    rows = [list(df.columns)] + df.astype(object).where(pd.notna(df), "").values.tolist()
    xml_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            cell_ref = f"{excel_column_name(col_idx)}{row_idx}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{cell_ref}"><v>{value}</v></c>')
            else:
                text = escape(str(value))
                cells.append(
                    f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'
                )
        xml_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        "</worksheet>"
    )


def write_excel(path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    workbook_sheets = []
    workbook_rels = []
    content_type_overrides = []
    worksheet_xml_map = {}

    for idx, (sheet_name, df) in enumerate(sheets.items(), start=1):
        sheet_id = idx
        rel_id = f"rId{idx}"
        workbook_sheets.append(
            f'<sheet name="{escape(sheet_name)}" sheetId="{sheet_id}" r:id="{rel_id}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        )
        content_type_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
        worksheet_xml_map[f"xl/worksheets/sheet{idx}.xml"] = dataframe_to_sheet_xml(df)

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets>'
        "</workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(workbook_rels)}'
        "</Relationships>"
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f'{"".join(content_type_overrides)}'
        "</Types>"
    )

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        for worksheet_path, worksheet_xml in worksheet_xml_map.items():
            zf.writestr(worksheet_path, worksheet_xml)


def main() -> None:
    args = parse_args()
    rows = collect_results(args.results_dir, args.suffix)

    if not rows:
        raise ValueError("No tau-bench result files found.")

    raw_df = pd.DataFrame(
        rows,
        columns=[
            "file",
            "dataset",
            "model_dir",
            "model_name",
            "scenario",
            "reward",
            "r_actions",
            "r_outputs",
            "num_samples",
        ],
    )

    summary_rows = build_summary_rows(rows)
    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "dataset",
            "model_dir",
            "model_name",
            "retail_reward",
            "retail_r_actions",
            "retail_r_outputs",
            "retail_num_samples",
            "airline_reward",
            "airline_r_actions",
            "airline_r_outputs",
            "airline_num_samples",
        ],
    )

    os.makedirs(os.path.dirname(args.output_excel), exist_ok=True)
    write_excel(
        args.output_excel,
        {
            "summary": summary_df,
            "raw": raw_df,
        },
    )
    print(f"Summary written to {args.output_excel}")


if __name__ == "__main__":
    main()
