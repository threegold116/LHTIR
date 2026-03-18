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

from collections import defaultdict
import torch
from verl import DataProto

from verl.workers.reward_manager import register
import json


@register("matchtir")
class MatchTIRRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        # the number of batches of decoded responses to print to the console
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum(
            )
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            split = data_item.non_tensor_batch.get("split", "train")

            # valid_response_length = data_item.batch["valid_length"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum(-1)

            valid_response_ids = response_ids[:valid_response_length]


            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=False)

            codes = json.loads(str(data_item.non_tensor_batch["codes"]))
            unsolved_set = json.loads(str(
                data_item.non_tensor_batch["unsolved_set"]))
            solve_rate = data_item.non_tensor_batch["solve_rate"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            
            answer = data_item.non_tensor_batch.get("answer", None)
            gt_tool_call = None
            if split == "train":
                gt_tool_call = data_item.non_tensor_batch["ground_truth"]

            score = self.compute_score(
                # messages=messages,
                response=response_str,
                codes=codes,
                unsolved_set=unsolved_set,
                solve_rate=solve_rate,
                split=split,
                answer=answer,
                gt_tool_call=gt_tool_call,
                tokenizer=self.tokenizer,
                valid_response_ids=valid_response_ids.tolist()
            )
            
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    if key != "score":
                        reward_extra_info[key].append(value)
            else:
                reward = score

            if isinstance(reward, list):
                reward_tensor[i, :valid_response_length] = torch.tensor(
                        reward, dtype=reward_tensor.dtype, device=reward_tensor.device
                    )
            else:
                reward_tensor[i, valid_response_length - 1] = reward

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


