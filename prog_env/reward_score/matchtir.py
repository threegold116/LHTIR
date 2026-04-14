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

from prog_env.utils.utils import answer_verify, get_feedback
from prog_env.utils.parse_output import parse_qwen
import re
import json
import os
from collections import Counter
import string
import numpy as np
import ot
from typing import List, Dict, Tuple, Optional

def compute_solve_pr(response: str, codes: dict, unsolved_set: dict, solve_rate: float, split: str, answer: str = None):
    unsolved_cnt = sum([len(v) for v in unsolved_set.values()])
    if split == "train":
        response = response.strip().removesuffix(
            '<|endoftext|>').strip().removesuffix('<|im_end|>').strip()
        parsed_response = parse_qwen(response)
        if parsed_response.get('tool_calls', None):
            tmp = 0
            feedback = get_feedback(parsed_response['tool_calls'], codes)
            for tool_call, feed in zip(parsed_response['tool_calls'], feedback):
                tool_name = tool_call['function']['name']
                answers = unsolved_set.get(tool_name, [])
                for answer in answers:
                    if answer_verify(feed['content'], answer):
                        answers.remove(answer)
                        tmp += 1
                        break
            score = tmp * tmp / len(parsed_response["tool_calls"])
        else:
            if parsed_response['content'] is None or parsed_response['content'] == '':
                score = -0.5
            elif '<tool_call>' in parsed_response['content'] or '</tool_call>' in parsed_response['content'] or '<tool_parsed_response>' in parsed_response['content']:
                score = -0.3
            elif answer is not None and answer.lower().replace(',', '').strip() in parsed_response['content'].lower().replace(',', '').strip():
                score = 1 / (1 + unsolved_cnt)
            elif solve_rate == 1.:
                score = 0.5
            else:
                score = 0.
        return score

    elif split == "test":
        raise NotImplementedError("Not implemented for test split.")

    else:
        raise ValueError(f"Unknown split: {split}")


def compute_solve_rate(response: str, codes: dict, unsolved_set: dict, solve_rate: float, split: str, answer: str = None):
    unsolved_cnt = sum([len(v) for v in unsolved_set.values()])
    if split == "train":
        response = response.strip().removesuffix(
            '<|endoftext|>').strip().removesuffix('<|im_end|>').strip()
        parsed_response = parse_qwen(response)
        if parsed_response.get('tool_calls', None):
            tmp = 0
            feedback = get_feedback(parsed_response['tool_calls'], codes)
            
            for tool_call, feed in zip(parsed_response['tool_calls'], feedback):
                tool_name = tool_call['function']['name']
                answers = unsolved_set.get(tool_name, [])
                for answer in answers:
                    if answer_verify(feed['content'], answer):
                        answers.remove(answer)
                        tmp += 1
                        break
            score = float(tmp)
        else:
            if parsed_response['content'] is None or parsed_response['content'] == '':
                score = -0.5
            elif '<tool_call>' in parsed_response['content'] or '</tool_call>' in parsed_response['content'] or '<tool_parsed_response>' in parsed_response['content']:
                score = -0.3
            elif answer is not None and answer.lower().replace(',', '').strip() in parsed_response['content'].lower().replace(',', '').strip():
                score = 1 / (1 + unsolved_cnt)
            elif solve_rate == 1.:
                score = 0.5
            else:
                score = 0.
        return score

    elif split == "test":
        raise NotImplementedError("Not implemented for test split.")

    else:
        raise ValueError(f"Unknown split: {split}")

def compute_solve_precision(response: str, codes: dict, unsolved_set: dict, solve_rate: float, split: str, answer: str = None):
    unsolved_cnt = sum([len(v) for v in unsolved_set.values()])
    if split == "train":
        response = response.strip().removesuffix(
            '<|endoftext|>').strip().removesuffix('<|im_end|>').strip()
        parsed_response = parse_qwen(response)
        if parsed_response.get('tool_calls', None):
            tmp = 0
            feedback = get_feedback(parsed_response['tool_calls'], codes)
            for tool_call, feed in zip(parsed_response['tool_calls'], feedback):
                tool_name = tool_call['function']['name']
                answers = unsolved_set.get(tool_name, [])
                for answer in answers:
                    if answer_verify(feed['content'], answer):
                        answers.remove(answer)
                        tmp += 1
                        break
            score = tmp / len(parsed_response["tool_calls"])
        else:
            if parsed_response['content'] is None or parsed_response['content'] == '':
                score = -0.5
            elif '<tool_call>' in parsed_response['content'] or '</tool_call>' in parsed_response['content'] or '<tool_parsed_response>' in parsed_response['content']:
                score = -0.3
            elif answer is not None and answer.lower().replace(',', '').strip() in parsed_response['content'].lower().replace(',', '').strip():
                score = 1 / (1 + unsolved_cnt)
            elif solve_rate == 1.:
                score = 0.5
            else:
                score = 0.
        return score

    elif split == "test":
        raise NotImplementedError("Not implemented for test split.")

    else:
        raise ValueError(f"Unknown split: {split}")

def compute_solve_f1(response: str, codes: dict, unsolved_set: dict, solve_rate: float, split: str, answer: str = None, gt_tool_call: list = None, tokenizer=None, valid_response_ids=None):
    unsolved_cnt = sum([len(v) for v in unsolved_set.values()])
    if split == "train":
        response = response.strip().removesuffix(
            '<|endoftext|>').strip().removesuffix('<|im_end|>').strip()
        parsed_response = parse_qwen(response)
        if parsed_response.get('tool_calls', None):
            tmp = 0
            feedback = get_feedback(parsed_response['tool_calls'], codes)
            for tool_call, feed in zip(parsed_response['tool_calls'], feedback):
                tool_name = tool_call['function']['name']
                answers = unsolved_set.get(tool_name, [])
                for answer in answers:
                    if answer_verify(feed['content'], answer):
                        answers.remove(answer)
                        tmp += 1
                        break
            score = 2 * tmp / (len(parsed_response["tool_calls"]) + 1)
        else:
            if parsed_response['content'] is None or parsed_response['content'] == '':
                score = -0.5
            elif '<tool_call>' in parsed_response['content'] or '</tool_call>' in parsed_response['content'] or '<tool_parsed_response>' in parsed_response['content']:
                score = -0.3
            elif answer is not None and answer.lower().replace(',', '').strip() in parsed_response['content'].lower().replace(',', '').strip():
                score = 1 / (1 + unsolved_cnt)
            elif solve_rate == 1.:
                score = 0.5
            else:
                score = 0.
        return score

    elif split == "test":
        response = response.strip().removesuffix(
            '<|endoftext|>').strip().removesuffix('<|im_end|>').strip()
        parsed_response = parse_qwen(response)
        tool_calls = parsed_response.get("tool_calls") or []

        if tool_calls:
            feedback = get_feedback(parsed_response['tool_calls'], codes)
            for tool_call, feed in zip(parsed_response['tool_calls'], feedback):
                tool_name = tool_call['function']['name']
                answers = unsolved_set.get(tool_name, [])
                for answer in answers:
                    if answer_verify(feed['content'], answer):
                        answers.remove(answer)
                        break
        solved_cnt = unsolved_cnt - sum([len(item)
                                    for item in unsolved_set.values()])
        score = {"solve_precision": solved_cnt / len(tool_calls) if len(tool_calls) != 0 else 1.,
                "solve_rate": solved_cnt / unsolved_cnt,
                "solve_f1": 2 * solved_cnt / (len(tool_calls) + unsolved_cnt),
                "score": 2 * solved_cnt / (len(tool_calls) + unsolved_cnt)}
        return score

    else:
        raise ValueError(f"Unknown split: {split}")

def match_score(list1, list2):
    """Compute a similarity score considering element frequency, ignoring order."""
    if list1 == list2:
        return 1.0
    
    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)  
    count2 = Counter(list2)  

    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection

    return intersection / max_possible if max_possible > 0 else 0.0

def compute_toolrl(response: str, codes: dict, unsolved_set: dict, solve_rate: float, split: str, answer: str = None, gt_tool_call: list = None, tokenizer=None, valid_response_ids=None):
    if split == "train":
        if isinstance(gt_tool_call, str):
            gt_tool_call = json.loads(gt_tool_call)
        gt_tool_name = []
        for tool in gt_tool_call:
            gt_tool_name.append(tool['name'])
        score = 0.0
        response = response.strip().removesuffix(
            '<|endoftext|>').strip().removesuffix('<|im_end|>').strip()
        parsed_response = parse_qwen(response)
        pd_tools = parsed_response.get("tool_calls") or []
        if pd_tools:
            pd_tool_name = []
            for pd_tool in pd_tools:
                pd_tool_name.append(pd_tool['function']['name'])
            score += match_score(gt_tool_name, pd_tool_name)
            used_pd_indices = set() 
            
            local_max_possible = 1.0
            for gt_tool in gt_tool_call:
                gt_name = gt_tool["name"]
                gt_params = gt_tool["parameters"]
                best_match = None
                best_match_score = 0.0
                best_match_index = -1
                local_max_possible += 1.0 + len(gt_params)
                for i, pd_tool in enumerate(pd_tools):
                    if i in used_pd_indices or pd_tool['function']["name"] != gt_name:
                        continue
                    
                    pd_params = json.loads(pd_tool['function']["arguments"])
                    if isinstance(pd_params, dict):
                        param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))
                        correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)
                    else:
                        param_score = 0
                        correctness_score = 0
                        
                    total_score = param_score + correctness_score
                    
                    if total_score > best_match_score:
                        best_match_score = total_score
                        best_match = pd_tool
                        best_match_index = i
                if best_match:
                    used_pd_indices.add(best_match_index)
                    score += best_match_score
            score = score * 6 / local_max_possible - 3
        return score

    elif split == "test":  
        return compute_solve_f1(response, codes, unsolved_set, solve_rate, split, answer, gt_tool_call, tokenizer, valid_response_ids)
#------THREEGOLDCHANGE--------#
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_f1_score_recall(prediction: str, ground_truths:str):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])
    
    return final_metric['f1']

def compute_answer_f1_recall(response: str, codes: dict, unsolved_set: dict, solve_rate: float, split: str, answer: str = None, gt_tool_call: list = None, tokenizer=None, valid_response_ids=None):
    response = response.strip().removesuffix(
            '<|endoftext|>').strip().removesuffix('<|im_end|>').strip()
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            pd_answer = answer_match.group(1).strip()
        else:
            return 0 
    except Exception as e:
        print(f"Error extracting answer content: {e}")
        return 0

    return get_f1_score_recall(pd_answer, answer)


#------THREEGOLDCHANGE--------#

def preprocess_text(text: str) -> str:
    """Preprocess text for dataset scoring.

    Processing steps:
    1. Convert to lowercase
    2. Remove punctuation marks (.,!?;:'"()[]{}...)
    3. Strip extra whitespaces
    """
    text = text.lower()
    
    for punct in string.punctuation:
        text = text.replace(punct, ' ')

    text = re.sub(r'\s+', ' ', text)
    
    text = text.strip()
    return text

def compute_answer_f1(response: str, codes: dict, unsolved_set: dict, solve_rate: float, split: str, answer: str = None, gt_tool_call: list = None, tokenizer=None, valid_response_ids=None):
    if split == "train":
        response = response.strip().removesuffix(
                '<|endoftext|>').strip().removesuffix('<|im_end|>').strip()
        try:
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if answer_match:
                pd_answer = answer_match.group(1).strip()
                pd_answer = preprocess_text(pd_answer)
            else:
                return 0 
        except Exception as e:
            print(f"Error extracting answer content: {e}")
            return 0

        gt_answer = preprocess_text(answer)

        pred_tokens = set(pd_answer.split())
        gt_tokens = set(gt_answer.split())

        common_tokens = pred_tokens & gt_tokens

        precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
        recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0

        if precision + recall > 0: 
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
        else:
            return 0
    elif split == "test":  
        #------THREEGOLDCHANGE--------#
        '''
        1. test和train保持一致
        通过修改reward.function.name实现
        '''
        #TODO:需要改成answer-level的f1
        # return compute_solve_f1(response, codes, unsolved_set, solve_rate, split, answer, gt_tool_call, tokenizer, valid_response_ids)
        response = response.strip().removesuffix(
                '<|endoftext|>').strip().removesuffix('<|im_end|>').strip()
        try:
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if answer_match:
                pd_answer = answer_match.group(1).strip()
                pd_answer = preprocess_text(pd_answer)
            else:
                return 0 
        except Exception as e:
            print(f"Error extracting answer content: {e}")
            return 0

        gt_answer = preprocess_text(answer)

        pred_tokens = set(pd_answer.split())
        gt_tokens = set(gt_answer.split())

        common_tokens = pred_tokens & gt_tokens

        precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
        recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0

        if precision + recall > 0: 
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
        else:
            return 0
        #------THREEGOLDCHANGE--------#

def tool_similarity(pred: dict, gt: dict) -> float:
    score = 0.0
    local_max_possible = 1.0
    name_score = 1.0 if (pred.get('name','').lower() == gt.get('name','').lower()) else 0.0
    if name_score:
        score += name_score
        pd_params = json.loads(pred["arguments"])
        gt_params = gt.get('parameters',{})
        score += match_score(list(gt_params.keys()), list(pd_params.keys()))
        local_max_possible += 1.0 + len(gt_params)
        param_value_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)
        score += param_value_score
    return score / local_max_possible

def hungarian_assignment(scores_matrix: List[List[float]]) -> List[Optional[int]]:
    P = len(scores_matrix)
    G = len(scores_matrix[0]) if P>0 else 0
    matched_gt = set()
    assignment = [None]*P
    pairs = []
    for i in range(P):
        for j in range(G):
            pairs.append((scores_matrix[i][j], i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])
    for score, i, j in pairs:
        if assignment[i] is None and j not in matched_gt:
            if score > 0:  
                assignment[i] = j
                matched_gt.add(j)
    return assignment

def assign_rewards_hungarian(preds: List[dict], gts: List[dict], unmatched_penalty=-0.5):
    P = len(preds)
    G = len(gts)
    if P == 0:
        return []
    scores = [[tool_similarity(preds[i], gts[j])
               for j in range(G)] for i in range(P)]
    assignment = hungarian_assignment(scores)
    rewards = []
    for t, pred in enumerate(preds):
        j = assignment[t]
        if j is None:
            r = unmatched_penalty 
        else:
            sim = scores[t][j]
            r = sim
        rewards.append(r)

    return rewards

def assign_rewards_ot(preds, gts, reg=0.1):
    P = len(preds)
    G = len(gts)

    if P == 0:
        return []

    S = np.zeros((P, G), dtype=float)
    for i in range(P):
        for j in range(G):
            S[i][j] = tool_similarity(preds[i], gts[j])

    C = -S

    mu = np.ones(P) / P
    nu = np.ones(G) / G

    T = ot.sinkhorn(mu, nu, C, reg=reg)

    rewards = (T * S).sum(axis=1).tolist()
    return rewards

def find_subsequence(full_tokens, sub_tokens):
    """Return the start index of sub_tokens in full_tokens. If not found, return -1."""
    n, m = len(full_tokens), len(sub_tokens)
    if m == 0:
        return -1

    for i in range(n - m + 1):
        if full_tokens[i:i+m] == sub_tokens:
            return i
    return -1

def find_all_subsequence(full_ids, sub_ids):
    n, m = len(full_ids), len(sub_ids)
    positions = []
    for i in range(n - m + 1):
        if full_ids[i:i+m] == sub_ids:
            positions.append(i)

    return positions

def compute_process_KM(response: str, codes: dict, unsolved_set: dict, solve_rate: float, split: str, answer: str = None, gt_tool_call: list = None, tokenizer=None, valid_response_ids=None):
    if split == "train":
        if isinstance(gt_tool_call, str):
            gt_tool_call = json.loads(gt_tool_call)
        scores = [0.0] * len(valid_response_ids)
       
        search_start = 0

        chats = response.split("\n<|im_start|>assistant\n")
        chats_size = len(chats)
        
        process_reward = [0.0] * chats_size
        count_per_turn = [0] * chats_size

        tool_call_list = []
        tool_to_turn_index = []
        for i, chat in enumerate(chats):
            chat = chat.strip().removesuffix('<|endoftext|>').strip().removesuffix('<|im_end|>').strip()
            answer_match = re.search(r'<answer>(.*?)</answer>', chat, re.DOTALL)
            if not answer_match:
                parsed_response = parse_qwen(chat)
                pd_tools = parsed_response.get("tool_calls") or []
                if len(pd_tools) > 0:
                    for j, pd_tool in enumerate(pd_tools):
                        tool_call_list.append(pd_tool["function"])
                        tool_to_turn_index.append(i)
            else :
                answer_score = 0.0
                pd_answer = answer_match.group(1).strip()
                pd_answer = preprocess_text(pd_answer)
                
                gt_answer = preprocess_text(answer)

                pred_tokens = set(pd_answer.split())
                gt_tokens = set(gt_answer.split())

                common_tokens = pred_tokens & gt_tokens

                precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0

                if precision + recall > 0: 
                    f1 = 2 * (precision * recall) / (precision + recall)
                    answer_score += f1
                process_reward[i] = answer_score

        tool_call_reward = assign_rewards_hungarian(tool_call_list, gt_tool_call, unmatched_penalty=0)

        for reward, turn_idx in zip(tool_call_reward, tool_to_turn_index):
            process_reward[turn_idx] += reward
            count_per_turn[turn_idx] += 1

        for i in range(chats_size):
            if count_per_turn[i] > 0:
                process_reward[i] /= count_per_turn[i]
        sep_str = "\n<|im_start|>assistant\n"
        sep_ids = tokenizer.encode(sep_str, add_special_tokens=False)

        start_positions = [0]

        start_positions += find_all_subsequence(valid_response_ids, sep_ids)

        end_str = "<|im_end|>"
        end_ids = tokenizer.encode(end_str, add_special_tokens=False)
 
        turns = []
        for s in start_positions:
            rel_pos = find_all_subsequence(valid_response_ids[s:], end_ids)
            
            if len(rel_pos) == 0:
                e = len(valid_response_ids)
            else:
                e = s + rel_pos[0] + len(end_ids)

            turns.append((s, e))

        for i, turn in enumerate(turns): #turn-level reward
            start, end = turn
            scores[start:end] = [0.0] * (end - start)
            scores[end - 1] = process_reward[i]

        return scores
    elif split == "test":  
        return compute_solve_f1(response, codes, unsolved_set, solve_rate, split, answer, gt_tool_call, tokenizer, valid_response_ids)




def compute_process_ot(response: str, codes: dict, unsolved_set: dict, solve_rate: float, split: str, answer: str = None, gt_tool_call: list = None, tokenizer=None, valid_response_ids=None):
    if split == "train":
        if isinstance(gt_tool_call, str):
            gt_tool_call = json.loads(gt_tool_call)
        scores = [0.0] * len(valid_response_ids)
       
        search_start = 0

        chats = response.split("\n<|im_start|>assistant\n")
        chats_size = len(chats)
        
        process_reward = [0.0] * chats_size
        count_per_turn = [0] * chats_size

        tool_call_list = []
        tool_to_turn_index = []
        for i, chat in enumerate(chats):
            chat = chat.strip().removesuffix('<|endoftext|>').strip().removesuffix('<|im_end|>').strip()
            answer_match = re.search(r'<answer>(.*?)</answer>', chat, re.DOTALL)
            if not answer_match:
                parsed_response = parse_qwen(chat)
                pd_tools = parsed_response.get("tool_calls") or []
                if len(pd_tools) > 0:
                    for j, pd_tool in enumerate(pd_tools):
                        tool_call_list.append(pd_tool["function"])
                        tool_to_turn_index.append(i)
            else :
                answer_score = 0.0
                pd_answer = answer_match.group(1).strip()
                pd_answer = preprocess_text(pd_answer)
                
                gt_answer = preprocess_text(answer)

                pred_tokens = set(pd_answer.split())
                gt_tokens = set(gt_answer.split())

                common_tokens = pred_tokens & gt_tokens

                precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0

                if precision + recall > 0:  
                    f1 = 2 * (precision * recall) / (precision + recall)
                    answer_score += f1
                process_reward[i] = answer_score

        tool_call_reward = assign_rewards_ot(tool_call_list, gt_tool_call)

        for reward, turn_idx in zip(tool_call_reward, tool_to_turn_index):
            process_reward[turn_idx] += reward
            count_per_turn[turn_idx] += 1

        for i in range(chats_size):
            if count_per_turn[i] > 0:
                process_reward[i] /= count_per_turn[i]
        sep_str = "\n<|im_start|>assistant\n"
        sep_ids = tokenizer.encode(sep_str, add_special_tokens=False)

        start_positions = [0]

        start_positions += find_all_subsequence(valid_response_ids, sep_ids)

        end_str = "<|im_end|>"
        end_ids = tokenizer.encode(end_str, add_special_tokens=False)
 
        turns = []
        for s in start_positions:
            rel_pos = find_all_subsequence(valid_response_ids[s:], end_ids)
            
            if len(rel_pos) == 0:
                e = len(valid_response_ids)
            else:
                e = s + rel_pos[0] + len(end_ids)

            turns.append((s, e))


        for i, turn in enumerate(turns):
            start, end = turn
            scores[start:end] = [0.0] * (end - start)
            scores[end - 1] = process_reward[i]

        return scores
    elif split == "test":  
        return compute_solve_f1(response, codes, unsolved_set, solve_rate, split, answer, gt_tool_call, tokenizer, valid_response_ids)