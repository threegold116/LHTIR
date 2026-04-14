import argparse
import asyncio
import copy
import json
import os

from tqdm import trange

from prog_env.utils.chatvllm import ChatVLLM
from prog_env.utils.llm_server import AsyncLLMServer

from evaluate.acebench.simulator import AsyncChatResponder, run_agent_multi_step, run_agent_multi_turn


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True, choices=["agent_multi_step", "agent_multi_turn"])
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--series", type=str, default="qwen", choices=["qwen"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_alias", type=str, default="MatchTIR")
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--engine", type=str, default="remote", choices=["remote"])
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--possible_answer_file", type=str)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1)
    parser.add_argument("--max_dialog_turns", type=int, default=40)
    parser.add_argument("--acebench_root", type=str, default="/share/home/sxjiang/myproject/ACEBench")
    parser.add_argument("--user_backend", type=str, default="vllm", choices=["vllm", "openai"])
    parser.add_argument("--user_model_name", type=str, default="MatchTIR")
    parser.add_argument("--user_base_url", type=str)
    parser.add_argument("--user_api_key", type=str)
    parser.add_argument("--user_temperature", type=float, default=0.0)
    parser.add_argument("--user_max_tokens", type=int, default=4096)
    return parser.parse_args()


def load_jsonl_like(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def default_possible_answer_path(input_file: str) -> str:
    parent = os.path.dirname(input_file)
    return os.path.join(parent, "possible_answer", os.path.basename(input_file))


def load_data_with_answers(args):
    data = load_jsonl_like(args.input_file)
    possible_answer_file = args.possible_answer_file or default_possible_answer_path(args.input_file)
    possible_answers = load_jsonl_like(possible_answer_file)
    answer_map = {item["id"]: item for item in possible_answers}
    merged = []
    for sample in data:
        sample = copy.deepcopy(sample)
        answer = answer_map[sample["id"]]
        sample["ground_truth"] = answer["ground_truth"]
        sample["mile_stone"] = answer["mile_stone"]
        sample["data_source"] = f"ACEBench/{args.scenario}"
        merged.append(sample)
    return merged


async def process_batch(batch_samples, args, agent_responder, user_responder=None):
    tasks = []
    for sample in batch_samples:
        if args.scenario == "agent_multi_step":
            tasks.append(run_agent_multi_step(sample, args, agent_responder))
        else:
            tasks.append(run_agent_multi_turn(sample, args, agent_responder, user_responder))
    return await asyncio.gather(*tasks)


def main():
    args = parse_arguments()
    data = load_data_with_answers(args)
    if args.end_id == -1:
        args.end_id = len(data)
    data = data[min(args.start_id, args.end_id) : min(args.end_id, len(data))]

    os.makedirs("/".join(args.output_file.split("/")[:-1]), exist_ok=True)

    ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf8") as f:
            for line in f:
                item = json.loads(line)
                ids.add((item["id"], item["data_source"]))

    from transformers import AutoTokenizer

    args.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    agent_chat_engine = ChatVLLM(args)
    agent_responder = AsyncChatResponder(
        backend="vllm",
        model_name=args.model_alias,
        chat_engine=agent_chat_engine,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    user_responder = None
    user_chat_engine = None
    if args.scenario == "agent_multi_turn":
        if args.user_backend == "vllm":
            user_base_url = args.user_base_url or args.base_url
            user_model_name = args.user_model_name or args.model_alias
            if user_base_url == args.base_url and user_model_name == args.model_alias:
                user_responder = AsyncChatResponder(
                    backend="vllm",
                    model_name=user_model_name,
                    chat_engine=agent_chat_engine,
                    temperature=args.user_temperature,
                    max_tokens=args.user_max_tokens,
                )
            else:
                user_args = copy.deepcopy(args)
                user_args.base_url = user_base_url
                user_args.concurrency = 128
                user_args.enable_thinking = False
                user_args.model_alias = user_model_name
                user_args.max_tokens = args.user_max_tokens
                user_args.temperature = args.user_temperature
                user_chat_engine = ChatVLLM(user_args)
                user_responder = AsyncChatResponder(
                    backend="vllm",
                    model_name=user_model_name,
                    chat_engine=user_chat_engine,
                    temperature=args.user_temperature,
                    max_tokens=args.user_max_tokens,
                )
        else:
            user_responder = AsyncChatResponder(
                backend="openai",
                model_name=args.user_model_name,
                api_key=args.user_api_key,
                base_url=args.user_base_url,
                temperature=args.user_temperature,
                max_tokens=args.user_max_tokens,
            )

    for i in trange(0, len(data), args.batch_size):
        if isinstance(agent_chat_engine.engine, AsyncLLMServer):
            agent_chat_engine.engine.set_sem()
        if user_chat_engine is not None and isinstance(user_chat_engine.engine, AsyncLLMServer):
            user_chat_engine.engine.set_sem()
        batch = data[i : i + args.batch_size]
        filter_batch = [sample for sample in batch if (sample["id"], f"ACEBench/{args.scenario}") not in ids]
        if not filter_batch:
            continue
        results = asyncio.run(process_batch(filter_batch, args, agent_responder, user_responder))
        with open(args.output_file, "a", encoding="utf8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
