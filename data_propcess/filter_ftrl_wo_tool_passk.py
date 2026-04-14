#!/usr/bin/env python3
"""
Probe FTRL questions with model direct answers (w/o tools), then filter out
questions with pass@k > 0.
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import error, request

import datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="/ssd/project/LHTIR/data/ftrl/train.parquet",
        help="Input parquet path.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/ssd/project/LHTIR/data/ftrl/train_wo_tool_hard.parquet",
        help="Output parquet path after filtering pass@k>0 samples.",
    )
    parser.add_argument(
        "--probe_log_file",
        type=str,
        default="/ssd/project/LHTIR/data/ftrl/train_wo_tool_probe_log.jsonl",
        help="Per-sample probing log path.",
    )
    parser.add_argument("--model_name", type=str, required=True, help="Served model name.")
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key.")
    parser.add_argument("--k", type=int, default=8, help="k for pass@k.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=64, help="Max tokens per sampled answer.")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel worker count.")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per request.")
    parser.add_argument("--request_timeout", type=int, default=60, help="HTTP timeout (seconds).")
    parser.add_argument(
        "--metric",
        type=str,
        default="subem",
        choices=["em", "subem", "f1"],
        help="Correctness metric between prediction and GT.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=1.0,
        help="A sample is counted as pass when score >= threshold.",
    )
    parser.add_argument("--limit", type=int, default=-1, help="Probe only first N rows (-1 means all).")
    parser.add_argument(
        "--save_log",
        action="store_true",
        help="Whether to save probe logs to --probe_log_file.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp/hf_datasets_cache",
        help="HF datasets cache dir (must be writable).",
    )
    return parser.parse_args()


def _check_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"input parquet not found: {path}")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def score_answer(pred: str, gt: str, metric: str) -> float:
    pred = normalize_answer(pred)
    gt = normalize_answer(gt)
    if not pred:
        return 0.0
    if metric == "em":
        return 1.0 if pred == gt else 0.0
    if metric == "subem":
        return 1.0 if gt in pred else 0.0
    pred_tokens = set(pred.split())
    gt_tokens = set(gt.split())
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = pred_tokens & gt_tokens
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gt_tokens)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def extract_question(messages) -> str:
    if isinstance(messages, list) and messages:
        content = messages[-1].get("content", "")
        if isinstance(content, str):
            m = re.search(r"Question:\s*(.*)\s*$", content, flags=re.S)
            if m:
                return m.group(1).strip()
            return content.strip()
    return ""


def extract_answer(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    return m.group(1).strip() if m else text.strip()


def chat_once(args: argparse.Namespace, question: str) -> str:
    endpoint = args.api_base.rstrip("/") + "/chat/completions"
    user_prompt = (
        "Answer the question directly without any tool calls.\n"
        "Only output the final answer inside <answer></answer>.\n"
        f"Question: {question}"
    )
    payload = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    last_err = None
    for retry in range(1, args.max_retries + 1):
        try:
            req = request.Request(endpoint, data=body, headers=headers, method="POST")
            with request.urlopen(req, timeout=args.request_timeout) as resp:
                rsp = json.loads(resp.read().decode("utf-8"))
            return rsp["choices"][0]["message"]["content"]
        except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_err = exc
            if retry < args.max_retries:
                time.sleep(min(2**retry, 8))
    raise RuntimeError(f"request failed: {last_err}")


def probe_one(idx: int, row: dict, args: argparse.Namespace) -> dict:
    question = extract_question(row.get("messages"))
    gt = str(row.get("answer", "")).strip()
    if not question or not gt:
        return {
            "index": idx,
            "question": question,
            "gt_answer": gt,
            "attempts": [],
            "pass_count": 0,
            "pass_at_k": 0.0,
            "filtered": False,
            "error": "missing question or answer",
        }

    attempts = []
    pass_count = 0
    for _ in range(args.k):
        raw = chat_once(args, question)
        pred = extract_answer(raw)
        score = score_answer(pred, gt, args.metric)
        is_pass = score >= args.score_threshold
        pass_count += int(is_pass)
        attempts.append({"pred": pred, "score": score, "is_pass": is_pass})

    pass_at_k = 1.0 if pass_count > 0 else 0.0
    return {
        "index": idx,
        "question": question,
        "gt_answer": gt,
        "attempts": attempts,
        "pass_count": pass_count,
        "pass_at_k": pass_at_k,
        "filtered": pass_at_k > 0.0,
        "error": "",
    }


def main() -> None:
    args = parse_args()
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir

    _check_file_exists(args.input_file)
    ds = datasets.load_dataset("parquet", data_files=args.input_file)["train"]

    probe_size = len(ds) if args.limit < 0 else min(len(ds), args.limit)
    results = [None] * probe_size
    print(f"dataset_size={len(ds)} probe_size={probe_size} k={args.k}")

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        futures = {executor.submit(probe_one, i, ds[i], args): i for i in range(probe_size)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[i] = {
                    "index": i,
                    "question": "",
                    "gt_answer": str(ds[i].get("answer", "")),
                    "attempts": [],
                    "pass_count": 0,
                    "pass_at_k": 0.0,
                    "filtered": False,
                    "error": str(exc),
                }
            done += 1
            if done % 50 == 0 or done == probe_size:
                print(f"probed {done}/{probe_size}")

    keep_idx = [i for i, rec in enumerate(results) if not rec["filtered"]]
    if probe_size < len(ds):
        keep_idx.extend(range(probe_size, len(ds)))
    filtered_ds = ds.select(keep_idx)

    filtered_count = sum(1 for rec in results if rec["filtered"])
    error_count = sum(1 for rec in results if rec["error"])
    print(
        f"filtered={filtered_count} kept={len(filtered_ds)} "
        f"errors={error_count} pass_rate={filtered_count / max(probe_size, 1):.4f}"
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    filtered_ds.to_parquet(args.output_file)
    print(f"saved: {args.output_file}")

    if args.save_log:
        os.makedirs(os.path.dirname(args.probe_log_file), exist_ok=True)
        with open(args.probe_log_file, "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"saved: {args.probe_log_file}")


if __name__ == "__main__":
    main()
