"""
 Copyright 2025 Bytedance Ltd. and/or its affiliates

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from utils.utils import get_feedback, chat_close
from utils.parse_output import get_parse_output
from utils.llm_server import AsyncLLMServer, VLLMEngine
from utils.chatvllm import ChatVLLM
import argparse
import json
import os
from tqdm import trange
from copy import deepcopy
from hashlib import sha256


def _get_info(sample, args):
    data_source = sample.get("data_source", "tau-bench")
    if data_source == "tau-bench/airline":
        with open(os.path.join(args.info_dir, "airline", "tools.json")) as f:
            tools = json.load(f)
        with open(os.path.join(args.info_dir, "airline", "codes.json")) as f:
            codes = json.load(f)
        with open(os.path.join(args.info_dir, "airline", "flights.json")) as f:
            flights = json.load(f)
        with open(os.path.join(args.info_dir, "airline", "reservations.json")) as f:
            reservations = json.load(f)
        with open(os.path.join(args.info_dir, "airline", "users.json")) as f:
            users = json.load(f)
        data = {
            "flights": flights,
            "reservations": reservations,
            "users": users
        }
        with open(os.path.join(args.info_dir, "airline", "wiki.md")) as f:
            WIKI = f.read()
        messages = [
            {
                "role": "system",
                "content": WIKI
            }
        ]
    elif data_source == "tau-bench/retail":
        with open(os.path.join(args.info_dir, "retail", "tools.json")) as f:
            tools = json.load(f)
        with open(os.path.join(args.info_dir, "retail", "codes.json")) as f:
            codes = json.load(f)
        with open(os.path.join(args.info_dir, "retail", "orders.json")) as f:
            orders = json.load(f)
        with open(os.path.join(args.info_dir, "retail", "products.json")) as f:
            products = json.load(f)
        with open(os.path.join(args.info_dir, "retail", "users.json")) as f:
            users = json.load(f)
        data = {
            "orders": orders,
            "products": products,
            "users": users
        }
        with open(os.path.join(args.info_dir, "retail", "wiki.md")) as f:
            WIKI = f.read()
        messages = [
            {
                "role": "system",
                "content": WIKI
            }
        ]
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    return tools, codes, data, messages


def _ask_to_user(user_messages: list, args):
    response = chat_close(user_messages, args, base_url=args.user_base_url,
                          api_key=args.user_api_key, model="gpt-4o-2024-08-06", max_tokens=512, temperature=0.0)
    if response:
        user_messages.append(response.copy())
        return response["content"]
    else:
        return None


def _calculate_reward(sample, args, messages, data):
    def _to_hashable(item):
        if isinstance(item, dict):
            return tuple((key, _to_hashable(value)) for key, value in sorted(item.items()))
        elif isinstance(item, list):
            return tuple(_to_hashable(element) for element in item)
        elif isinstance(item, set):
            return tuple(sorted(_to_hashable(element) for element in item))
        else:
            return item

    def _consistent_hash(value):
        return sha256(str(value).encode("utf-8")).hexdigest()

    def _get_data_hash(data):
        return _consistent_hash(_to_hashable(data))

    data_hash = _get_data_hash(data)

    _, codes, data, _ = _get_info(sample, args)
    actions = sample["actions"]
    outputs = sample["outputs"]
    _ = get_feedback(actions, codes, data=data)
    gt_data_hash = _get_data_hash(data)

    reward = 1. if data_hash == gt_data_hash else 0.

    if reward == 1. and len(outputs) > 0:
        for output in outputs:
            found = False
            for message in messages:
                if message["role"] == "assistant" and message.get("tool_calls", None):
                    if output.lower() in message["content"].lower().replace(",", ""):
                        found = True
                        break
            if not found:
                reward = 0.
                break

    return reward


def sample_process_open(sample, args, chat_engine):
    save_sample = {"messages": [], "instruction": sample["instruction"], "data_source": sample.get(
        "data_source", "tau-bench"), "model": args.model_path, "metrics": {"pass^1": 0.}}
    tools, codes, data, messages = _get_info(sample, args)

    user_messages = [
        {
            "role": "system",
            "content": f"You are a user interacting with an agent.\n\nInstruction: {sample['instruction']}\nRules:\n- Just generate one line at a time to simulate the user's message.\n- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.\n- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.\n- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.\n- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.\n- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."
        },
        {
            "role": "user",
            "content": "Hi! How can I help you today?"
        }
    ]
    question = _ask_to_user(user_messages=user_messages, args=args)
    if question:
        messages.append({"role": "user", "content": question})
    else:
        return None

    for _ in range(args.max_turns):
        tools_list = [tools] if tools else [None]
        raw_outputs = chat_engine.chat_open_batch([messages], tools_list)
        response = raw_outputs[0]
        response = args.parse_output(response, one_tool_only=True)
        messages.append(response.copy())
        if response.get("tool_calls", None):
            feedback = get_feedback(response["tool_calls"], codes, data=data)
            messages.extend(feedback.copy())
            if response['tool_calls'][0]['function']['name'] == 'transform_to_user_agent':
                reward = _calculate_reward(sample, args, messages, data)
                save_sample["metrics"]["pass^1"] = reward
                break
        else:
            user_messages.append(
                {"role": "user", "content": response["content"]})
            user_response = _ask_to_user(
                user_messages=user_messages, args=args)
            if user_response:
                messages.append({"role": "user", "content": user_response})
            else:
                return None

            if "###STOP###" in user_response:
                reward = _calculate_reward(sample, args, messages, data)
                save_sample["metrics"]["pass^1"] = reward
                break

    save_sample["messages"] = messages.copy()

    return save_sample


def sample_process_open_batch(batch_samples, args, chat_engine):
    B = len(batch_samples)
    batch_messages = []
    batch_tools = []
    batch_codes = []
    batch_data = []
    batch_user_messages = []
    save_results = []

    for sample in batch_samples:
        tools, codes, data, messages = _get_info(sample, args)
        batch_messages.append(messages)
        batch_tools.append(tools)
        batch_codes.append(codes)
        batch_data.append(data)
        save_results.append({
            "messages": [],
            "instruction": sample["instruction"],
            "data_source": sample.get("data_source", "tau-bench"),
            "model": args.model_path,
            "metrics": {"pass^1": 0.}
        })
        user_messages = [
            {
                "role": "system",
                "content": f"You are a user interacting with an agent.\n\nInstruction: {sample['instruction']}\nRules:\n- Just generate one line at a time to simulate the user's message.\n- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.\n- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.\n- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.\n- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.\n- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."
            },
            {
                "role": "user",
                "content": "Hi! How can I help you today?"
            }
        ]
        batch_user_messages.append(user_messages)

    active_indices = []
    for i in range(B):
        question = _ask_to_user(user_messages=batch_user_messages[i], args=args)
        if question:
            batch_messages[i].append({"role": "user", "content": question})
            active_indices.append(i)

    for turn in range(args.max_turns):
        if not active_indices:
            break
        active_messages = [batch_messages[i] for i in active_indices]
        active_tools = [batch_tools[i] for i in active_indices]
        llm_outputs = chat_engine.chat_open_batch(active_messages, active_tools)
        new_active_indices = []
        for idx_in_active, original_idx in enumerate(active_indices):
            out = args.parse_output(llm_outputs[idx_in_active], one_tool_only=True)
            batch_messages[original_idx].append(out.copy())
            sample = batch_samples[original_idx]
            if out.get("tool_calls", None):
                feedback = get_feedback(
                    out["tool_calls"],
                    batch_codes[original_idx],
                    data=batch_data[original_idx]
                )
                batch_messages[original_idx].extend(feedback.copy())
                if out["tool_calls"][0]["function"]["name"] == "transform_to_user_agent":
                    reward = _calculate_reward(
                        sample, args,
                        batch_messages[original_idx],
                        batch_data[original_idx]
                    )
                    save_results[original_idx]["metrics"]["pass^1"] = reward
                else:
                    new_active_indices.append(original_idx)
            else:
                batch_user_messages[original_idx].append(
                    {"role": "user", "content": out["content"]}
                )
                user_response = _ask_to_user(
                    user_messages=batch_user_messages[original_idx], args=args
                )
                if user_response is None:
                    pass
                else:
                    batch_messages[original_idx].append(
                        {"role": "user", "content": user_response}
                    )
                    if "###STOP###" in user_response:
                        reward = _calculate_reward(
                            sample, args,
                            batch_messages[original_idx],
                            batch_data[original_idx]
                        )
                        save_results[original_idx]["metrics"]["pass^1"] = reward
                    else:
                        new_active_indices.append(original_idx)
        active_indices = new_active_indices

    for i in range(B):
        save_results[i]["messages"] = batch_messages[i].copy()
        if save_results[i]["metrics"]["pass^1"] == 0. and batch_messages[i]:
            reward = _calculate_reward(
                batch_samples[i], args,
                batch_messages[i],
                batch_data[i]
            )
            save_results[i]["metrics"]["pass^1"] = reward

    return save_results


def sample_process_close(sample, args):
    save_sample = {"messages": [], "instruction": sample["instruction"], "data_source": sample.get(
        "data_source", "tau-bench"), "model": args.model_path, "metrics": {"pass^1": 0.}}
    tools, codes, data, messages = _get_info(sample, args)

    user_messages = [
        {
            "role": "system",
            "content": f"You are a user interacting with an agent.\n\nInstruction: {sample['instruction']}\nRules:\n- Just generate one line at a time to simulate the user's message.\n- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.\n- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.\n- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.\n- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.\n- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."
        },
        {
            "role": "user",
            "content": "Hi! How can I help you today?"
        }
    ]
    question = _ask_to_user(user_messages=user_messages, args=args)
    if question:
        messages.append({"role": "user", "content": question})
    else:
        return None

    for _ in range(args.max_turns):
        response = chat_close(messages, args, tools=tools, one_tool_only=True)
        if response:
            messages.append(response.copy())
            if response.get("tool_calls", None):
                feedback = get_feedback(
                    response["tool_calls"], codes, data=data)
                messages.extend(feedback.copy())
                if response['tool_calls'][0]['function']['name'] == 'transform_to_user_agent':
                    reward = _calculate_reward(sample, args, messages, data)
                    save_sample["metrics"]["pass^1"] = reward
                    break
            else:
                user_messages.append(
                    {"role": "user", "content": response["content"]})
                user_response = _ask_to_user(
                    user_messages=user_messages, args=args)
                if user_response:
                    messages.append({"role": "user", "content": user_response})
                else:
                    return None

                if "###STOP###" in user_response:
                    break
        else:
            return None

    save_sample["messages"] = messages.copy()
    reward = _calculate_reward(sample, args, messages, data)
    save_sample["metrics"]["pass^1"] = reward

    return save_sample


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=str, default="qwen",
                        choices=["gpt", "qwen", "claude", "gemini"])
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--base_url", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--engine", type=str, default="local",
                        choices=["local", "remote"])
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--user_base_url", type=str)
    parser.add_argument("--user_api_key", type=str)
    parser.add_argument("--input_file", type=str,
                        default="Data/jsonl/raw/tau-bench.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--info_dir", default="Data/jsonl/raw/tau-bench")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        '--max_turns', type=int, default=30)

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    with open(args.input_file, 'r', encoding='utf8') as f:
        data = [json.loads(line) for line in f.readlines()]

    if args.end_id == -1:
        args.end_id = len(data)
    data = data[min(args.start_id, args.end_id): min(args.end_id, len(data))]

    os.makedirs("/".join(args.output_file.split("/")[:-1]), exist_ok=True)

    ids = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf8') as f:
            ids = [json.loads(line)["instruction"]
                   for line in f.readlines()]

    if args.series in ["qwen"]:
        from transformers import AutoTokenizer
        args.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        args.parse_output = get_parse_output(args.series)
        chat_engine = ChatVLLM(args)
        batch_size = args.batch_size

        for i in trange(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            filter_batch = [
                s for s in batch
                if s["instruction"] not in ids
            ]
            if filter_batch:
                results = sample_process_open_batch(
                    filter_batch, args, chat_engine
                )
                with open(args.output_file, "a", encoding="utf8") as f:
                    for res in results:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                        f.flush()

    else:
        from utils.utils import get_params
        args.params = get_params(args.series, args)
        for i in trange(len(data)):
            sample = data[i]
            if sample["instruction"] not in ids:
                responses = sample_process_close(sample, args)
                if responses:
                    with open(args.output_file, "a") as f:
                        f.write(json.dumps(responses, ensure_ascii=False) + "\n")
                        f.flush()


if __name__ == "__main__":
    main()
