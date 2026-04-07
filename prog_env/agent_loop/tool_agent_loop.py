# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4

from prog_env.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from prog_env.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


#--------THREEGOLDCHANGE--------#
'''
1.增加关于利用code模拟执行tool的函数
- call_function: 利用code模拟执行tool
- get_feedback: 利用call_function模拟执行tool
2.修改导入逻辑:从verl.experimental.agent_loop->prog_env.agent_loop
'''
instance_id = 0
#--------THREEGOLDCHANGE--------#

from func_timeout import func_set_timeout, FunctionTimedOut
import contextlib
import io
buf = io.StringIO()
@func_set_timeout(10)
def call_function(name, arguments, code, **kwargs):
    #--------THREEGOLDCHANGE--------#
    '''
    1.增加关于利用code模拟执行tool的函数
    - call_function: 利用code模拟执行tool
        - 利用namespace控制code的执行:code中只能访问namespace中的变量和函数
    '''
    namespace = {}
    with contextlib.redirect_stdout(buf):
        exec(code, namespace, namespace)
    #--------THREEGOLDCHANGE--------#

    if name in namespace:
        predict = namespace[name](**arguments, **kwargs)
    else:
        raise NameError(f"name {name} is not defined")
    if type(predict) == dict or type(predict) == list:
        predict = json.dumps(predict, ensure_ascii=False)
    elif type(predict) != str:
        predict = str(predict)
    return predict

#--------THREEGOLDCHANGE--------#
'''
1.增加关于利用code模拟执行tool的函数
- get_feedback: 利用call_function模拟执行tool
    - 在FTRL基础上基于FunctionCall修改get_feedback函数
'''
#--------THREEGOLDCHANGE--------#
def get_feedback(tool_calls: list[FunctionCall], codes: list[dict[str, Any]], **kwargs):
    '''
    Args:
        tool_calls: list[FunctionCall]
        codes: list[dict[str, Any]]
        **kwargs: dict[str, Any]
    Returns:
        list[dict[str, Any]]
    '''
    res = []
    codes = json.loads(codes)
    for tool_call in tool_calls:
        try:
            #--------THREEGOLDCHANGE--------#
            '''
            1.基于FunctionCall修改get_feedback函数
            - 这里的tool_call是FunctionCall类型(NameArgument)
            '''
            # if not isinstance(tool_call, dict):
            #     raise ValueError(f"tool_call should be dict, got {type(tool_call)}")
                
            # if 'function' not in tool_call:
            #     raise ValueError("Missing 'function' key in tool_call")
                
            # func_info = tool_call['function']
            # if not isinstance(func_info, dict):
            #     raise ValueError(f"'function' should be dict, got {type(func_info)}")
                
            # tool_name = func_info.get('name')
            tool_name = tool_call.name

            if not tool_name:
                raise ValueError("Missing tool name")

            # tool_args_str = func_info.get('arguments', '{}')
            tool_args_str = tool_call.arguments

            if not isinstance(tool_args_str, str):
                tool_args_str = str(tool_args_str)
            
            tool_args = json.loads(tool_args_str)
            
            #--------THREEGOLDCHANGE--------#
            code = codes.get(tool_name)
            
            if not code:
                raise ValueError(f"No code found for tool {tool_name}")

            feedback = call_function(tool_name, tool_args, code, **kwargs)
            res.append({"role": "tool", "content": feedback})
            
        except FunctionTimedOut as e:
            res.append({"role": "tool", "content": f"Timeout when calling {tool_name}: {str(e)}"})
        except json.JSONDecodeError as e:
            res.append({"role": "tool", "content": f"Invalid arguments format for {tool_name}: {str(e)}"})
        except Exception as e:
            res.append({"role": "tool", "content": f"an error occured when call {tool_name}: {str(e)}"})

    return res

@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        #--------THREEGOLDCHANGE--------#
        '''
        1.tools从instance中获取
        '''
        # tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        # tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        # cls.tools = {tool.name: tool for tool in tool_list}
        # cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        #--------THREEGOLDCHANGE--------#
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Not initialized tools: tools are implemented by code")

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        #--------THREEGOLDCHANGE--------#
        '''
        1.新增mask_tokens的控制逻辑
        - 增加max_step_length的控制逻辑
        '''
        #--------THREEGOLDCHANGE--------#
        cls.max_step_length = config.actor_rollout_ref.rollout.multi_turn.get("max_step_length", cls.response_length)
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)

    @rollout_trace_op
    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any], tools: list[dict[str, Any]], codes: list[dict[str, Any]]) -> AgentLoopOutput:
        global instance_id
        metrics = {}
        request_id = uuid4().hex
        #--------THREEGOLDCHANGE--------#
        '''
        1.tools从instance中获取:原始是通过tool_config_path获取的的tool_schemas,现在是直接通过data中的tools获取的
        2.新增step_length统计和控制
        3.新增时间统计
        '''
        import time
        t1 = time.time()
        prompt_ids = await self.loop.run_in_executor(
            None,
            # lambda: self.tokenizer.apply_chat_template(
            #     messages, tools=self.tool_schemas, add_generation_prompt=True, tokenize=True
            # ),
            lambda: self.tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=True
            ),#TODO:加上enable_thinking的逻辑,默认就是添加的
        )
        instance_id += 1
        logger.error(f"request_id: {request_id}, instance_id: {instance_id}")
        local_instance_id = instance_id
        step_length_list = []   
        cost_time_list = []
        #--------THREEGOLDCHANGE--------#
        response_mask = []
        user_turns, assistant_turns = 0, 0
        while True:
            #--------THREEGOLDCHANGE--------#
            '''
            2.新增mask_tokens的控制逻辑:max_step_length(默认response_length)和剩余response_length(response_length-len(response_mask))的最小值
            - verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py:
                - 之前max_new_tokens默认采用min(self.config.response_length, self.config.max_model_len - len(generation_prompt_ids) - 1), 而可能还存在response_length-len(response_mask)<self.config.max_model_len - len(generation_prompt_ids) - 1的情况
                - 因为self.config.max_model_len = self.config.prompt_length + self.config.response_length而prompt_length和response_length都是模型最大长度的约束(即max_model_len是prompt_length和response_length的约束)
            '''
            logger.error(f"instance_id: {local_instance_id}, max_step_length: {self.max_step_length}, response_length: {self.response_length}")
            max_new_tokens = min(self.max_step_length, self.response_length-len(response_mask))
            sampling_params["max_new_tokens"] = max_new_tokens  
            t2 = time.time()
            logger.error(f"instance_id: {local_instance_id}, sampling_params: {sampling_params}")
            #--------THREEGOLDCHANGE--------#
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            
            #--------THREEGOLDCHANGE--------#
            '''
            3.新增step_length统计和控制
            '''
            step_length_list.append(len(response_ids))
            cost_time_list.append(time.time() - t2)
            logger.error(f"running instance_id: {local_instance_id}, step_length: {len(response_ids)}, assistant_turns: {assistant_turns}, cost_time: {time.time() - t2}")
            #--------THREEGOLDCHANGE--------#
            assistant_turns += 1

            # reach max response length
            if len(response_mask) >= self.response_length:
                break

            # reach max assistant turns
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break

            # reach max user turns
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            # no tool calls
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            if not tool_calls:
                break

            # call tools
            tasks = []
            for tool_call in tool_calls[: self.max_parallel_calls]:
                tasks.append(self._call_tool(tool_call, codes))
            with simple_timer("tool_calls", metrics):
                tool_responses = await asyncio.gather(*tasks)
            if any(isinstance(item, Exception) for item in tool_responses):
                break

            # append tool_response_ids
            tool_response_ids = await self.loop.run_in_executor(
                None,
                lambda messages=tool_responses: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                ),
            )
            tool_response_ids = tool_response_ids[len(self.system_prompt) :]

            # NOTE: last turn should not be user turn, or the EOS token reward
            # can't be propagated to previous token in GAE.
            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                break

            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)
            user_turns += 1
        #--------THREEGOLDCHANGE--------#
        '''
        4.新增打印部分:logger.error打印cost_time
        '''
        logger.error(f"instance_id finished: {local_instance_id}, response_length: {len(response_mask)}, assistant_turns: {assistant_turns}, step_length_list: {step_length_list}, cost_time_list: {cost_time_list}, total_cost_time: {time.time() - t1}")
        #--------THREEGOLDCHANGE--------#
        '''
        5.新增step_length_list的存储:__step_length__
        '''
        if step_length_list:
            metrics["step_length"] = list(step_length_list)
        #--------THREEGOLDCHANGE--------#
        response_ids = prompt_ids[-len(response_mask) :] #裁剪prompt_ids得到response_ids
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)] #裁剪response_ids得到prompt_ids

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
            codes=codes,
        )
        return output

    async def _call_tool(self, tool_call: FunctionCall, codes: list[dict[str, Any]]=None) -> dict[str, str]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            #--------THREEGOLDCHANGE--------#
            '''
            基于get_feedback函数模拟执行tool:被注释的是原agentloop实现
            '''
            if codes is None:
                tool_name = tool_call.name
                tool_args = json.loads(tool_call.arguments)
                tool = self.tools[tool_name]

                instance_id = await tool.create()
                tool_response, _, _ = await tool.execute(instance_id, tool_args)
            else:
                tool_response = get_feedback([tool_call], codes)[0]["content"]
            
            #--------THREEGOLDCHANGE--------#
        except Exception as e:
            logger.exception(f"Error when executing tool: {e}")
            return e
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        if len(tool_response) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response = tool_response[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response = "(truncated)..." + tool_response[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response = tool_response[:length] + "...(truncated)..." + tool_response[-length:]

        return {
            "role": "tool",
            "content": tool_response,
        }
