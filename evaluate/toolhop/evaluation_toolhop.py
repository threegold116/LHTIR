import argparse
import asyncio
from prog_env.utils.utils import  get_params,  answer_verify, get_feedback, chat_close
from prog_env.utils.llm_server import AsyncLLMServer
from prog_env.utils.chatvllm import ChatVLLM
import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from prog_env.utils.parse_output import get_parse_output
import json
import os
from tqdm import trange


def _answer_correct(messages, answer):
    if messages[-1]['role'] == 'assistant':
        if messages[-1]["content"]:
            if answer_verify(messages[-1]["content"], answer):
                return True
            elif messages[-2]["role"] == "tool" and answer_verify(messages[-2]["content"], answer):
                return True
    return False



def sample_process_close_direct(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", f"ToolHop/{args.scenario}"), "model": args.model_path, "metrics": {
        "answer_correctness": 0.}}

    if sample.get('messages', None):
        messages = sample['messages']
    else:
        messages = [
            {"role": "user", "content": f"You will be asked a question, and should provide a short answer.\nIf the answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the answer is a name, format it as follows: Firstname Lastname\nIf the answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the answer in the following format: <answer>your answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
    response = chat_close(messages, args)
    if response:
        messages.append(response.copy())
        save_sample['messages'] = messages.copy()
        save_sample["metrics"]["answer_correctness"] = 1. if _answer_correct(
            messages, sample['answer']) else 0.

        return save_sample
    return None




def sample_process_close_mandatory(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", f"ToolHop/{args.scenario}"), "model": args.model_path, "metrics": {
        "answer_correctness": 0.}}

    if sample.get('messages'):
        messages = sample['messages']
    else:
        messages = [
            {"role": "user", "content": f"You will be asked a question with some tools, and should provide a short final answer.\nPlease note that you must call the tool at every step, you must not use your own knowledge. Your final answer must also be returned from the tool.\nIf the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the final nswer is a name, format it as follows: Firstname Lastname\nIf the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the final answer in the following format: <answer>final answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
    tools = [{"type": "function", "function": tool.copy()}
             for tool in sample["tools"].values()]

    for _ in range(args.max_turns):
        response = chat_close(messages, args, tools=tools)
        if response:
            messages.append(response.copy())
            if response.get('tool_calls', None):
                feedback = get_feedback(
                    response['tool_calls'], sample['functions'])
                messages.extend(feedback)
            else:
                break
        else:
            return None
    save_sample['messages'] = messages.copy()
    save_sample["metrics"]["answer_correctness"] = 1. if _answer_correct(
        messages, sample['answer']) else 0.
    return save_sample




def sample_process_close_free(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", f"ToolHop/{args.scenario}"), "model": args.model_path, "metrics": {
        "answer_correctness": 0.}}

    if sample.get('messages'):
        messages = sample['messages']
    else:
        messages = [
            {"role": "user", "content": f"You will be asked a question with some tools, and should provide a short final answer.\nIf the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the final nswer is a name, format it as follows: Firstname Lastname\nIf the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the final answer in the following format: <answer>final answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
    tools = [{"type": "function", "function": tool.copy()}
             for tool in sample["tools"].values()]

    for _ in range(args.max_turns):
        response = chat_close(messages, args, tools=tools)
        if response:
            messages.append(response.copy())
            if response.get('tool_calls', None):
                feedback = get_feedback(
                    response['tool_calls'], sample['functions'])
                messages.extend(feedback)
            else:
                break
        else:
            return None
    save_sample['messages'] = messages.copy()
    save_sample["metrics"]["answer_correctness"] = 1. if _answer_correct(
        messages, sample['answer']) else 0.
    return save_sample


def sample_process_open_direct_batch(batch_samples, args, chat_engine):
    B = len(batch_samples)
    save_results = []
    
    for sample in batch_samples:
        save_results.append({
            "messages": [],
            "data_source": sample.get("data_source", f"ToolHop/{args.scenario}"),
            "model": args.model_path,
            "metrics": {"answer_correctness": 0.}
        })
    
    batch_messages = []
    batch_tools = []
    
    for sample in batch_samples:
        if sample.get('messages'):
            messages = sample['messages']
        else:
            messages = [
                {"role": "user", "content": f"You will be asked a question, and should provide a short answer.\nIf the answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the answer is a name, format it as follows: Firstname Lastname\nIf the answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the answer in the following format: <answer>your answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
        
        batch_messages.append(messages)
        batch_tools.append([])  
    
    llm_outputs = chat_engine.chat_open_batch(batch_messages, batch_tools)
    
    for i in range(B):
        out = args.parse_output(llm_outputs[i])
        batch_messages[i].append(out.copy())
        save_results[i]["messages"] = batch_messages[i].copy()
        save_results[i]["metrics"]["answer_correctness"] = 1. if _answer_correct(
            batch_messages[i], batch_samples[i]['answer']) else 0.
    
    return save_results

def sample_process_open_mandatory_batch(batch_samples, args, chat_engine):
    B = len(batch_samples)
    save_results = []
    
    for sample in batch_samples:
        save_results.append({
            "messages": [],
            "data_source": sample.get("data_source", f"ToolHop/{args.scenario}"),
            "model": args.model_path,
            "metrics": {"answer_correctness": 0.}
        })
    
    batch_messages = []
    batch_tools = []
    
    for sample in batch_samples:
        if sample.get('messages'):
            messages = sample['messages']
        else:
            messages = [
                {"role": "user", "content": f"You will be asked a question with some tools, and should provide a short final answer.\nPlease note that you must call the tool at every step, you must not use your own knowledge. Your final answer must also be returned from the tool.\nIf the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the final nswer is a name, format it as follows: Firstname Lastname\nIf the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the final answer in the following format: <answer>final answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
        
        tools = [{"type": "function", "function": tool.copy()}
                 for tool in sample["tools"].values()]
        
        batch_messages.append(messages)
        batch_tools.append(tools)
    
    active_indices = list(range(B))
    for turn in range(args.max_turns):
        if not active_indices:
            break
            
        active_messages = [batch_messages[i] for i in active_indices]
        active_tools = [batch_tools[i] for i in active_indices]
        
        llm_outputs = chat_engine.chat_open_batch(active_messages, active_tools)
        new_active_indices = []
        for idx_in_active, original_idx in enumerate(active_indices):
            i = original_idx
            out = args.parse_output(llm_outputs[idx_in_active])
            batch_messages[i].append(out.copy())
            
            if out.get('tool_calls', None):
                feedback = get_feedback(out['tool_calls'], batch_samples[i]['functions'])
                batch_messages[i].extend(feedback)
                new_active_indices.append(original_idx)
            else:
                continue  
        
        active_indices = new_active_indices
    
    for i in range(B):
        save_results[i]["messages"] = batch_messages[i].copy()
        save_results[i]["metrics"]["answer_correctness"] = 1. if _answer_correct(
            batch_messages[i], batch_samples[i]['answer']) else 0.
    
    return save_results

def sample_process_open_free_batch(batch_samples, args, chat_engine):
    B = len(batch_samples)
    save_results = []
    
    for sample in batch_samples:
        save_results.append({
            "messages": [],
            "data_source": sample.get("data_source", f"ToolHop/{args.scenario}"),
            "model": args.model_path,
            "metrics": {"answer_correctness": 0.}
        })
    
    batch_messages = []
    batch_tools = []
    
    for sample in batch_samples:
        if sample.get('messages'):
            messages = sample['messages']
        else:
            messages = [
                {"role": "user", "content": f"You will be asked a question with some tools, and should provide a short final answer.\nIf the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the final nswer is a name, format it as follows: Firstname Lastname\nIf the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the final answer in the following format: <answer>final answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
        
        tools = [{"type": "function", "function": tool.copy()}
                 for tool in sample["tools"].values()]
        
        batch_messages.append(messages)
        batch_tools.append(tools)
    
    active_indices = list(range(B))
    print("max_turns", args.max_turns)
    for turn in range(args.max_turns):
        if not active_indices:
            break
            
        active_messages = [batch_messages[i] for i in active_indices]
        active_tools = [batch_tools[i] for i in active_indices]
        
        llm_outputs = chat_engine.chat_open_batch(active_messages, active_tools)
        
        new_active_indices = []
        for idx_in_active, original_idx in enumerate(active_indices):
            i = original_idx
            out = args.parse_output(llm_outputs[idx_in_active])
            batch_messages[i].append(out.copy())
            
            if out.get('tool_calls', None):
                feedback = get_feedback(out['tool_calls'], batch_samples[i]['functions'])
                batch_messages[i].extend(feedback)
                new_active_indices.append(original_idx)
            else:
                continue  
        
        active_indices = new_active_indices
    
    for i in range(B):
        save_results[i]["messages"] = batch_messages[i].copy()
        save_results[i]["metrics"]["answer_correctness"] = 1. if _answer_correct(
            batch_messages[i], batch_samples[i]['answer']) else 0.
    
    return save_results

debug_index = 0
async def rollout_one_instance(messages, tools, functions, args, engine: ChatVLLM):
    """对单个样本进行多轮 Mandatory 对话 rollout。"""
    global debug_index
    debug_index += 1
    local_debug_index = debug_index
    for _ in range(args.max_turns):
        # chat_batch 接受列表，这里对单个样本构造长度为 1 的 batch
        llm_output = await engine.chat_one_async(messages, tools)
        logger.info(f"This is the {_}th turn of the {local_debug_index}th instance.")
        out_text = llm_output
        out = args.parse_output(out_text)
        messages.append(out.copy())
        if out.get('tool_calls', None):
            feedback = get_feedback(out['tool_calls'], functions)
            messages.extend(feedback)
        else:
            break
    # logger.info(f"This is the final messages of the {debug_index} instance.")
    return messages



def build_messages(batch_samples, args):
    pass
    #TODO: 根据args.scenario构建messages，最好还是不要放到batch推理里面

async def sample_process_async_batch(batch_samples, args, engine: ChatVLLM):
    """基于 AsyncLLMServer 的多实例异步rollout，每个 instance 独立多轮对话。"""
    B = len(batch_samples)
    save_results = []

    for sample in batch_samples:
        save_results.append({
            "messages": [],
            "data_source": sample.get("data_source", f"ToolHop/{args.scenario}"),
            "model": args.model_path,
            "metrics": {"answer_correctness": 0.}
        })

    # 为每个样本准备初始 messages / tools / functions，并并发 rollout
    tasks = []
    for sample in batch_samples:
        if sample.get('messages'):
            messages = sample['messages']
        else:
            if args.scenario == "Mandatory":
                messages = [
                    {"role": "user", "content": f"You will be asked a question with some tools, and should provide a short final answer.\nPlease note that you must call the tool at every step, you must not use your own knowledge. Your final answer must also be returned from the tool.\nIf the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the final nswer is a name, format it as follows: Firstname Lastname\nIf the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the final answer in the following format: <answer>final answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
            elif args.scenario == "Free":
                messages = [
                    {"role": "user", "content": f"You will be asked a question with some tools, and should provide a short final answer.\nIf the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the final nswer is a name, format it as follows: Firstname Lastname\nIf the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the final answer in the following format: <answer>final answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
            elif args.scenario == "Direct":
                messages = [
                    {"role": "user", "content": f"You will be asked a question, and should provide a short answer.\nIf the answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the answer is a name, format it as follows: Firstname Lastname\nIf the answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the answer in the following format: <answer>your answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
            else:
                raise NotImplementedError

        tools = [{"type": "function", "function": tool.copy()}
                 for tool in sample["tools"].values()]
        functions = sample["functions"]
        if args.scenario=="Direct":
            args.max_turns = 1
            tools = []
        tasks.append(rollout_one_instance(messages, tools, functions, args, engine))

    all_messages = await asyncio.gather(*tasks)

    for i in range(B):
        msgs = all_messages[i]
        save_results[i]["messages"] = msgs.copy()
        save_results[i]["metrics"]["answer_correctness"] = 1. if _answer_correct(
            msgs, batch_samples[i]['answer']) else 0.

    return save_results


def sample_process_close(sample: dict, args):
    if args.scenario == "Direct":
        return sample_process_close_direct(sample, args)
    elif args.scenario == "Mandatory":
        return sample_process_close_mandatory(sample, args)
    elif args.scenario == "Free":
        return sample_process_close_free(sample, args)
    else:
        raise NotImplementedError


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="Direct",
                        choices=["Direct", "Mandatory", "Free"])
    parser.add_argument("--series", type=str, default="qwen",
                        choices=["gpt", "qwen", "claude", "gemini"])
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--base_url", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--engine", type=str, default="local",
                        choices=["local", "remote"])
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--input_file", type=str,
                        default="/share/home/sxjiang/myproject/MatchTIR/Data/ToolHop.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        '--max_turns', type=int, default=9)
    parser.add_argument("--batch_mode", type=str, default="sync", choices=["sync", "async"])
    parser.add_argument("--batch_size", type=int, default=256)
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
            ids = [(json.loads(line)["messages"][0]["content"].split('\nQuestion: ')[-1], json.loads(line)["data_source"])
                   for line in f.readlines()]
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
            batch = data[i:i+batch_size]
            fliter_batch = []
            for sample in batch:
                if (sample["question"], f"ToolHop/{args.scenario}") not in ids:
                    fliter_batch.append(sample)

            if not fliter_batch:
                continue

            if args.scenario == "Direct":
                if args.batch_mode == "async":
                    results = asyncio.run(sample_process_async_batch(fliter_batch, args, chat_engine))
                else:
                    results = sample_process_open_direct_batch(fliter_batch, args, chat_engine)
            elif args.scenario == "Mandatory":
                if args.batch_mode == "async":
                    results = asyncio.run(sample_process_async_batch(fliter_batch, args, chat_engine))
                else:
                    results = sample_process_open_mandatory_batch(fliter_batch, args, chat_engine)
            elif args.scenario == "Free":
                if args.batch_mode == "async":
                    results = asyncio.run(sample_process_async_batch(fliter_batch, args, chat_engine))
                else:
                    results = sample_process_open_free_batch(fliter_batch, args, chat_engine)
            else:
                raise NotImplementedError

            with open(args.output_file, 'a', encoding="utf8") as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f.flush()

    else:
        args.params = get_params(args.series, args)
        for i in trange(len(data)):
            sample = data[i]
            if (sample["question"], f"ToolHop/{args.scenario}") not in ids:
                responses = sample_process_close(sample, args)
                if responses:
                    with open(args.output_file, 'a') as f:
                        f.write(json.dumps(responses, ensure_ascii=False) + '\n')
                        f.flush()


if __name__ == "__main__":
    main()
