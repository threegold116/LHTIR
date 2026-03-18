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

from utils.utils import load_model, get_params, chat_open, answer_verify, get_feedback, chat_close
from utils.parse_output import get_parse_output
from utils.chatvllm import ChatVLLM
import argparse
import json
import os
from tqdm import trange
from copy import deepcopy
from collections import Counter


def sample_process_open(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", "RoTBench"), "model": args.model_path, "metrics": {
        "TS": 0., "PI": 0., "CF": 0.}}

    messages: list = deepcopy(sample['messages'])
    response = chat_open([messages], args)[0]
    messages.append({"role": "assistant", "content": response.strip()})

    answers = []
    for answer in sample['answer']:
        tool = answer.split('Action: ')[1].split('Action Input: ')[0].strip()
        parameters = json.loads(answer.split('Action Input: ')[1].strip())
        assert type(
            parameters) == dict, f"parameters should be dict, but got {type(parameters)}"
        answers.append({"tool": tool, "parameters": parameters.copy()})

    TS, PI, CF = 0., 0., 0.

    try:
        tool = response.split('Action: ')[1].split('Action Input: ')[0].strip()
        parameters = json.loads(response.split('Action Input: ')[1].strip())
        assert type(
            parameters) == dict, f"parameters should be dict, but got {type(parameters)}"

        for answer in answers:
            if answer['tool'] == tool:
                TS = 1.
                if not answer['parameters'].keys() ^ parameters.keys():
                    PI = 1.
                    if Counter(answer['parameters'].values()) == Counter(parameters.values()):
                        CF = 1.
                        break
    except:
        pass

    save_sample["messages"] = deepcopy(messages)
    save_sample["metrics"] = {
        "TS": TS,
        "PI": PI,
        "CF": CF
    }

    return save_sample


def _compute_ts_pi_cf(response: str, sample: dict):
    """Parse response and sample['answer'] to compute TS, PI, CF. Used by batch path."""
    answers = []
    for answer in sample["answer"]:
        tool = answer.split("Action: ")[1].split("Action Input: ")[0].strip()
        parameters = json.loads(answer.split("Action Input: ")[1].strip())
        assert type(parameters) == dict, f"parameters should be dict, but got {type(parameters)}"
        answers.append({"tool": tool, "parameters": parameters.copy()})
    TS, PI, CF = 0.0, 0.0, 0.0
    try:
        tool = response.split("Action: ")[1].split("Action Input: ")[0].strip()
        parameters = json.loads(response.split("Action Input: ")[1].strip())
        assert type(parameters) == dict, f"parameters should be dict, but got {type(parameters)}"
        for answer in answers:
            if answer["tool"] == tool:
                TS = 1.0
                if not answer["parameters"].keys() ^ parameters.keys():
                    PI = 1.0
                    if Counter(answer["parameters"].values()) == Counter(parameters.values()):
                        CF = 1.0
                        break
    except Exception:
        pass
    return TS, PI, CF


def sample_process_open_batch(batch_samples, args, chat_engine):
    batch_messages = [deepcopy(s["messages"]) for s in batch_samples]
    batch_tools = [None] * len(batch_samples)
    raw_outputs = chat_engine.chat_open_batch(batch_messages, batch_tools)
    save_results = []
    for i in range(len(batch_samples)):
        response = raw_outputs[i].strip()
        batch_messages[i].append({"role": "assistant", "content": response})
        TS, PI, CF = _compute_ts_pi_cf(response, batch_samples[i])
        save_sample = {
            "messages": deepcopy(batch_messages[i]),
            "data_source": batch_samples[i].get("data_source", "RoTBench"),
            "model": args.model_path,
            "metrics": {"TS": TS, "PI": PI, "CF": CF},
        }
        save_results.append(save_sample)
    return save_results


def sample_process_close(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", "RoTBench"), "model": args.model_path, "metrics": {
        "TS": 0., "PI": 0., "CF": 0.}}

    messages: list = deepcopy(sample['messages'])
    response = chat_close(messages, args)
    if response:
        messages.append(response)

        answers = []
        for answer in sample['answer']:
            tool = answer.split('Action: ')[1].split(
                'Action Input: ')[0].strip()
            parameters = json.loads(answer.split('Action Input: ')[1].strip())
            assert type(
                parameters) == dict, f"parameters should be dict, but got {type(parameters)}"
            answers.append({"tool": tool, "parameters": parameters.copy()})

        TS, PI, CF = 0., 0., 0.

        try:
            tool = response["content"].split(
                'Action: ')[1].split('Action Input: ')[0].strip()
            parameters = json.loads(
                response["content"].split('Action Input: ')[1].strip())
            assert type(
                parameters) == dict, f"parameters should be dict, but got {type(parameters)}"

            for answer in answers:
                if answer['tool'] == tool:
                    TS = 1.
                    if not answer['parameters'].keys() ^ parameters.keys():
                        PI = 1.
                        if Counter(answer['parameters'].values()) == Counter(parameters.values()):
                            CF = 1.
                            break
        except:
            pass

        save_sample["messages"] = deepcopy(messages)
        save_sample["metrics"] = {
            "TS": TS,
            "PI": PI,
            "CF": CF
        }

        return save_sample
    return None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=str, default="qwen",
                        choices=["gpt", "qwen", "claude", "gemini"])
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--base_url", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--engine", type=str, default="local",
                        choices=["local", "remote"])
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_file", type=str,
                        default="Data/jsonl/raw/RoTBench.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        '--max_turns', type=int, default=1)

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

    ids = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf8') as f:
            ids = [(json.loads(line)["messages"][0]["content"], json.loads(line)["messages"][1]["content"], json.loads(line)["data_source"])
                   for line in f.readlines()]

    if args.series in ["qwen"]:
        from transformers import AutoTokenizer
        args.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        args.params = get_params(args.series, args)
        args.parse_output = get_parse_output(args.series)
        chat_engine = ChatVLLM(args)
        batch_size = args.batch_size

        for i in trange(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            filter_batch = [
                sample
                for sample in batch
                if (sample["messages"][0]["content"], sample["messages"][1]["content"], sample["data_source"]) not in ids
            ]
            if not filter_batch:
                continue
            results = sample_process_open_batch(filter_batch, args, chat_engine)
            with open(args.output_file, "a", encoding="utf8") as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f.flush()

    else:
        args.params = get_params(args.series, args)
        for i in trange(len(data)):
            sample = data[i]
            if (sample["messages"][0]["content"], sample["messages"][1]["content"], sample["data_source"]) not in ids:
                responses = sample_process_close(sample, args)
                if responses:
                    with open(args.output_file, 'a') as f:
                        f.write(json.dumps(responses, ensure_ascii=False) + '\n')
                        f.flush()


if __name__ == "__main__":
    main()
