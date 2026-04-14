import argparse
import asyncio
import json
import os

from tqdm import trange

from prog_env.utils.chatvllm import ChatVLLM
from prog_env.utils.llm_server import AsyncLLMServer

from evaluate.tau2bench.bootstrap import ensure_tau2_importable
from evaluate.tau2bench.env_adapter import get_task_count
from evaluate.tau2bench.runtime import AsyncVLLMResponder, run_one_task_safe


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True, choices=["mock", "airline", "retail", "telecom"])
    parser.add_argument("--task_split", type=str, default="base")
    parser.add_argument("--series", type=str, default="qwen", choices=["qwen"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_alias", type=str, default="MatchTIR")
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--user_base_url", type=str, required=True)
    parser.add_argument("--engine", type=str, default="remote", choices=["remote"])
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--user_max_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--user_temperature", type=float, default=0.0)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--tau2_root", type=str, default="/share/home/sxjiang/myproject/tau2-bench")
    parser.add_argument("--user_model_name", type=str, default="MatchTIR")
    parser.add_argument("--user_model_path", type=str, default=None)
    return parser.parse_args()


def load_finished_keys(output_file: str) -> set[tuple[int, str]]:
    keys = set()
    if not os.path.exists(output_file):
        return keys
    with open(output_file, "r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            keys.add((item["task_index"], item["data_source"]))
    return keys


async def process_batch(task_ids, args, agent_responder, user_responder):
    tasks = [run_one_task_safe(task_id, args, agent_responder, user_responder) for task_id in task_ids]
    return await asyncio.gather(*tasks)


def main():
    args = parse_arguments()
    ensure_tau2_importable(args.tau2_root)

    task_count = get_task_count(args.tau2_root, args.domain, args.task_split)
    if args.end_id == -1:
        args.end_id = task_count
    task_ids = list(range(min(args.start_id, args.end_id), min(args.end_id, task_count)))

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    finished = load_finished_keys(args.output_file)

    from transformers import AutoTokenizer

    args.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    agent_chat_engine = ChatVLLM(args)

    user_args = argparse.Namespace(**vars(args))
    user_args.base_url = args.user_base_url
    user_args.model_alias = args.user_model_name
    user_args.model_path = args.user_model_path or args.model_path
    user_args.max_tokens = args.user_max_tokens
    user_args.temperature = args.user_temperature
    user_args.enable_thinking = False
    user_args.tokenizer = AutoTokenizer.from_pretrained(user_args.model_path, trust_remote_code=True)
    if args.user_base_url == args.base_url and args.user_model_name == args.model_alias and user_args.model_path == args.model_path:
        user_chat_engine = agent_chat_engine
    else:
        user_chat_engine = ChatVLLM(user_args)

    agent_responder = AsyncVLLMResponder(
        chat_engine=agent_chat_engine,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    user_responder = AsyncVLLMResponder(
        chat_engine=user_chat_engine,
        temperature=args.user_temperature,
        max_tokens=args.user_max_tokens,
    )

    data_source = f"tau2-bench/{args.domain}/{args.task_split}"
    for i in trange(0, len(task_ids), args.batch_size):
        if isinstance(agent_chat_engine.engine, AsyncLLMServer):
            agent_chat_engine.engine.set_sem()
        if user_chat_engine is not agent_chat_engine and isinstance(user_chat_engine.engine, AsyncLLMServer):
            user_chat_engine.engine.set_sem()
        batch_task_ids = task_ids[i : i + args.batch_size]
        batch_task_ids = [task_id for task_id in batch_task_ids if (task_id, data_source) not in finished]
        if not batch_task_ids:
            continue
        results = asyncio.run(process_batch(batch_task_ids, args, agent_responder, user_responder))
        with open(args.output_file, "a", encoding="utf8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()

