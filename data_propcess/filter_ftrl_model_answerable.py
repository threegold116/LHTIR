#!/usr/bin/env python3
# Copyright 2026
"""
Filter out FTRL samples that a base model can answer without tools.
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib import error, request

import datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_train",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/data/ftrl/with_agent_name/train.parquet",
        help="Input train parquet path.",
    )
    parser.add_argument(
        "--input_test",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/data/ftrl/with_agent_name/test.parquet",
        help="Input test parquet path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/data/ftrl/with_agent_name_filtered",
        help="Output directory for filtered parquet files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name served by an OpenAI-compatible chat endpoint.",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key for the endpoint.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for direct-answer probing.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="Max generation tokens for direct-answer probing.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=1.0,
        help="Answer score threshold. >= threshold is treated as model-answerable.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["f1", "em", "subem"],
        help="Scoring metric between model direct answer and dataset answer.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of concurrent API workers.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Max retries for failed requests.",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=60,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--limit_train",
        type=int,
        default=-1,
        help="Only probe first N train samples (-1 means all).",
    )
    parser.add_argument(
        "--limit_test",
        type=int,
        default=-1,
        help="Only probe first N test samples (-1 means all).",
    )
    parser.add_argument(
        "--save_probe_log",
        action="store_true",
        help="Save per-sample probing result JSONL.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp/hf_datasets_cache",
        help="HF datasets cache dir (must be writable in restricted environments).",
    )
    return parser.parse_args()


def _check_file_exists(path: str, name: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} parquet not found: {path}")


def preprocess_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_score(pred: str, gt: str, metric: str) -> float:
    pred = preprocess_text(pred)
    gt = preprocess_text(gt)
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
    common_tokens = pred_tokens & gt_tokens
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extract_question(messages: Any) -> str:
    if isinstance(messages, list) and messages:
        content = messages[-1].get("content", "")
        if isinstance(content, str):
            matched = re.search(r"Question:\s*(.*)\s*$", content, flags=re.S)
            if matched:
                return matched.group(1).strip()
            return content.strip()
    return ""


def extract_answer(text: str) -> str:
    if not text:
        return ""
    matched = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if matched:
        return matched.group(1).strip()
    return text.strip()


def chat_completion(
    api_base: str,
    api_key: str,
    model_name: str,
    question: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    max_retries: int,
) -> str:
    endpoint = api_base.rstrip("/") + "/chat/completions"
    prompt = (
        "Answer the question directly without using any tools.\n"
        "Only output your final answer inside <answer></answer>.\n"
        f"Question: {question}"
    )
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            req = request.Request(endpoint, data=data, headers=headers, method="POST")
            with request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (error.HTTPError, error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"chat completion failed: {last_err}")


def evaluate_sample(idx: int, sample: dict, args: argparse.Namespace) -> dict:
    question = extract_question(sample.get("messages"))
    gt_answer = str(sample.get("answer", ""))
    if not question or not gt_answer:
        return {
            "index": idx,
            "question": question,
            "gt_answer": gt_answer,
            "pred_answer": "",
            "score": 0.0,
            "is_answerable": False,
            "error": "missing question or answer",
        }

    raw_pred = chat_completion(
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model_name,
        question=question,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.request_timeout,
        max_retries=args.max_retries,
    )
    pred_answer = extract_answer(raw_pred)
    score = compute_score(pred_answer, gt_answer, args.metric)
    return {
        "index": idx,
        "question": question,
        "gt_answer": gt_answer,
        "pred_answer": pred_answer,
        "score": score,
        "is_answerable": score >= args.score_threshold,
        "error": "",
    }


def filter_split(dataset: datasets.Dataset, split_name: str, args: argparse.Namespace) -> tuple[datasets.Dataset, list[dict]]:
    size = len(dataset)
    limit = args.limit_train if split_name == "train" else args.limit_test
    if limit is not None and limit > 0:
        size = min(size, limit)

    results: list[dict] = [None] * size
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        futures = {executor.submit(evaluate_sample, i, dataset[i], args): i for i in range(size)}
        finished = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[i] = {
                    "index": i,
                    "question": "",
                    "gt_answer": str(dataset[i].get("answer", "")),
                    "pred_answer": "",
                    "score": 0.0,
                    "is_answerable": False,
                    "error": str(exc),
                }
            finished += 1
            if finished % 100 == 0 or finished == size:
                print(f"[{split_name}] probed {finished}/{size}")

    keep_indices = [i for i, item in enumerate(results) if not item["is_answerable"]]
    if limit is not None and limit > 0 and len(dataset) > size:
        keep_indices.extend(range(size, len(dataset)))
    filtered = dataset.select(keep_indices)

    answerable_cnt = sum(1 for item in results if item["is_answerable"])
    err_cnt = sum(1 for item in results if item["error"])
    print(
        f"[{split_name}] total={len(dataset)} probed={size} "
        f"answerable={answerable_cnt} kept={len(filtered)} errors={err_cnt}"
    )
    return filtered, results


def dump_log(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir

    _check_file_exists(args.input_train, "Train")
    _check_file_exists(args.input_test, "Test")
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = datasets.load_dataset("parquet", data_files=args.input_train)["train"]
    test_dataset = datasets.load_dataset("parquet", data_files=args.input_test)["train"]

    filtered_train, train_results = filter_split(train_dataset, "train", args)
    filtered_test, test_results = filter_split(test_dataset, "test", args)

    train_output = os.path.join(args.output_dir, "train.parquet")
    test_output = os.path.join(args.output_dir, "test.parquet")
    filtered_train.to_parquet(train_output)
    filtered_test.to_parquet(test_output)
    print(f"saved: {train_output}")
    print(f"saved: {test_output}")

    if args.save_probe_log:
        train_log = os.path.join(args.output_dir, "train_probe_log.jsonl")
        test_log = os.path.join(args.output_dir, "test_probe_log.jsonl")
        dump_log(train_log, train_results)
        dump_log(test_log, test_results)
        print(f"saved: {train_log}")
        print(f"saved: {test_log}")


if __name__ == "__main__":
    main()
