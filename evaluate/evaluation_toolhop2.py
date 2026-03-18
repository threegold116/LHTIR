import argparse
import asyncio
import json
import logging
import os
from tqdm import trange

from prog_env.utils.utils import get_params, chat_open, get_feedback, chat_close
from prog_env.utils.llm_server import AsyncLLMServer
from prog_env.utils.chatvllm import ChatVLLM
from prog_env.utils.parse_output import get_parse_output

from evaluate.datasets import get_adapter
from evaluate.rollout import (
    rollout_multi_turn,
    rollout_batch_sync,
    rollout_async_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _run_one_sample_close(sample, args, adapter):
    """Single-sample close API path: build messages, rollout with chat_close, return save_sample or None."""
    scenario = args.scenario
    data_source = sample.get("data_source", adapter.get_default_data_source(scenario))
    save_sample = {
        "messages": [],
        "data_source": data_source,
        "model": args.model_path,
        "metrics": {"answer_correctness": 0.0},
    }
    messages = adapter.build_initial_messages(sample, scenario)
    tools = adapter.get_tools(sample, scenario)
    functions = adapter.get_functions(sample)
    max_turns = 1 if scenario == "Direct" else args.max_turns

    def chat_fn(msgs, t):
        return chat_close(msgs, args, tools=t)

    def get_feedback_fn(tool_calls):
        return get_feedback(tool_calls, functions)

    messages = rollout_multi_turn(
        messages, tools, max_turns, chat_fn, args.parse_output, get_feedback_fn
    )
    save_sample["messages"] = messages.copy()
    save_sample["metrics"] = adapter.compute_metrics(sample, messages)
    return save_sample


def _run_batch_direct(fliter_batch, args, adapter, chat_engine):
    """Direct scenario: single round, no tools."""
    B = len(fliter_batch)
    batch_messages = []
    for sample in fliter_batch:
        batch_messages.append(adapter.build_initial_messages(sample, args.scenario))
    batch_tools = [[] for _ in range(B)]
    llm_outputs = chat_engine.chat_open_batch(batch_messages, batch_tools)
    save_results = []
    for i in range(B):
        sample = fliter_batch[i]
        out = args.parse_output(llm_outputs[i])
        batch_messages[i].append(out.copy())
        data_source = sample.get("data_source", adapter.get_default_data_source(args.scenario))
        save_results.append({
            "messages": batch_messages[i].copy(),
            "data_source": data_source,
            "model": args.model_path,
            "metrics": adapter.compute_metrics(sample, batch_messages[i]),
        })
    return save_results


def _run_batch_multi_turn_sync(fliter_batch, args, adapter, chat_engine):
    """Mandatory/Free batch sync: use rollout_batch_sync."""
    B = len(fliter_batch)
    batch_messages = [adapter.build_initial_messages(s, args.scenario) for s in fliter_batch]
    batch_tools = [adapter.get_tools(s, args.scenario) for s in fliter_batch]
    batch_functions = [adapter.get_functions(s) for s in fliter_batch]

    def chat_batch_fn(active_messages, active_tools):
        return chat_engine.chat_open_batch(active_messages, active_tools)

    rollout_batch_sync(
        batch_messages,
        batch_tools,
        batch_functions,
        args.max_turns,
        chat_batch_fn,
        args.parse_output,
        get_feedback,
    )
    save_results = []
    for i in range(B):
        sample = fliter_batch[i]
        data_source = sample.get("data_source", adapter.get_default_data_source(args.scenario))
        save_results.append({
            "messages": batch_messages[i].copy(),
            "data_source": data_source,
            "model": args.model_path,
            "metrics": adapter.compute_metrics(sample, batch_messages[i]),
        })
    return save_results


async def _run_batch_multi_turn_async(fliter_batch, args, adapter, chat_engine):
    """Mandatory/Free batch async: parallel per-instance rollouts with chat_one_async."""
    B = len(fliter_batch)
    batch_messages = [adapter.build_initial_messages(s, args.scenario) for s in fliter_batch]
    batch_tools = [adapter.get_tools(s, args.scenario) for s in fliter_batch]
    batch_functions = [adapter.get_functions(s) for s in fliter_batch]
    max_turns = args.max_turns
    if args.scenario == "Direct":
        max_turns = 1

    async def chat_async_fn(messages, tools):
        return await chat_engine.chat_one_async(messages, tools)

    all_messages = await rollout_async_batch(
        batch_messages,
        batch_tools,
        batch_functions,
        max_turns,
        chat_async_fn,
        args.parse_output,
        get_feedback,
    )
    save_results = []
    for i in range(B):
        sample = fliter_batch[i]
        data_source = sample.get("data_source", adapter.get_default_data_source(args.scenario))
        save_results.append({
            "messages": all_messages[i].copy(),
            "data_source": data_source,
            "model": args.model_path,
            "metrics": adapter.compute_metrics(sample, all_messages[i]),
        })
    return save_results


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toolhop", help="Dataset adapter name")
    parser.add_argument("--scenario", type=str, default="Direct", choices=["Direct", "Mandatory", "Free"])
    parser.add_argument("--series", type=str, default="qwen", choices=["gpt", "qwen", "claude", "gemini"])
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--base_url", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--engine", type=str, default="local", choices=["local", "remote"])
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--input_file", type=str, default="/share/home/sxjiang/myproject/MatchTIR/Data/ToolHop.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_turns", type=int, default=9)
    parser.add_argument("--batch_mode", type=str, default="sync", choices=["sync", "async"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_arguments()
    adapter = get_adapter(args.dataset)
    data = adapter.load(args.input_file)
    if args.end_id == -1:
        args.end_id = len(data)
    data = data[min(args.start_id, args.end_id) : min(args.end_id, len(data))]

    out_dir = "/".join(args.output_file.split("/")[:-1])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ids = []
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf8") as f:
            for line in f.readlines():
                try:
                    ids.append(adapter.get_resume_id_from_result_line(json.loads(line)))
                except Exception:
                    pass

    if args.series in ["qwen"]:
        from transformers import AutoTokenizer

        args.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        args.params = get_params(args.series, args)
        args.parse_output = get_parse_output(args.series)
        chat_engine = ChatVLLM(args)
        batch_size = args.batch_size
        for i in trange(0, len(data), batch_size):
            if isinstance(chat_engine.engine, AsyncLLMServer):
                chat_engine.engine.set_sem()
            batch = data[i : i + batch_size]
            fliter_batch = [
                s for s in batch
                if adapter.get_sample_id(s, args.scenario) not in ids
            ]
            if not fliter_batch:
                continue
            if args.scenario == "Direct":
                results = _run_batch_direct(fliter_batch, args, adapter, chat_engine)
            elif args.scenario in ("Mandatory", "Free"):
                if args.batch_mode == "async":
                    results = asyncio.run(_run_batch_multi_turn_async(fliter_batch, args, adapter, chat_engine))
                else:
                    results = _run_batch_multi_turn_sync(fliter_batch, args, adapter, chat_engine)
            else:
                raise NotImplementedError
            with open(args.output_file, "a", encoding="utf8") as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f.flush()
    else:
        args.params = get_params(args.series, args)
        try:
            args.parse_output = get_parse_output(args.series)
        except NotImplementedError:
            # Close API returns message dict already
            args.parse_output = lambda x: x if isinstance(x, dict) else {"role": "assistant", "content": str(x), "tool_calls": None}
        for i in trange(len(data)):
            sample = data[i]
            if adapter.get_sample_id(sample, args.scenario) not in ids:
                save_sample = _run_one_sample_close(sample, args, adapter)
                if save_sample:
                    with open(args.output_file, "a", encoding="utf8") as f:
                        f.write(json.dumps(save_sample, ensure_ascii=False) + "\n")
                        f.flush()


if __name__ == "__main__":
    main()
