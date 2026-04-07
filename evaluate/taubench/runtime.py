from __future__ import annotations

import traceback
from typing import Any, Optional

from openai import AsyncOpenAI

from prog_env.utils.chatvllm import ChatVLLM
from prog_env.utils.llm_server import AsyncLLMServer

from evaluate.taubench.env_adapter import build_env, stringify_observation
from evaluate.taubench.metrics import compute_reward_metrics
from evaluate.taubench.parser import action_from_assistant_message, normalize_openai_message, parse_vllm_agent_output
from evaluate.taubench.user_simulator import AsyncTextResponder, TauBenchUserSimulator


class AsyncAgentResponder:
    def __init__(
        self,
        backend: str,
        model_name: str,
        chat_engine: Optional[ChatVLLM] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.chat_engine = chat_engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        if backend == "openai":
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(self, messages: list[dict], tools: list[dict]) -> dict[str, Any]:
        if self.backend == "vllm":
            assert self.chat_engine is not None
            assert isinstance(self.chat_engine.engine, AsyncLLMServer)
            raw_output = await self.chat_engine.chat_one_async(
                messages, tools, max_tokens=self.max_tokens, temperature=self.temperature
            )
            return parse_vllm_agent_output(raw_output)
        if self.backend == "openai":
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return normalize_openai_message(response.choices[0].message)
        raise NotImplementedError(f"Unsupported backend: {self.backend}")


async def run_one_task(task_index: int, args, agent_responder: AsyncAgentResponder, user_responder: AsyncTextResponder) -> dict[str, Any]:
    env = build_env(args.tau_root, args.env, args.task_split, task_index=task_index)
    user_simulator = TauBenchUserSimulator(user_responder)
    env.user = user_simulator
    env.task_index = task_index
    env.task = env.tasks[task_index]
    env.data = env.data_load_func()
    env.actions = []

    initial_user_message = await user_simulator.reset(env.task.instruction)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": env.wiki},
        {"role": "user", "content": initial_user_message},
    ]

    stop_reason = "max_steps"
    error = None
    for step in range(args.max_steps):
        assistant_message = await agent_responder.generate(messages, env.tools_info)
        if assistant_message.get("tool_calls"):
            assistant_message["tool_calls"] = assistant_message["tool_calls"][:1]
        messages.append(assistant_message)

        action = action_from_assistant_message(
            assistant_message,
            action_cls=type(env.task.actions[0]),
            respond_action_name="respond",
        )
        env.actions.append(action)

        if action.name == "respond":
            observation = await user_simulator.step(action.kwargs.get("content", ""))
            messages.append({"role": "user", "content": observation})
            if "###STOP###" in observation:
                stop_reason = "user_stop"
                break
            continue

        if action.name in env.tools_map:
            try:
                observation = env.tools_map[action.name].invoke(data=env.data, **action.kwargs)
            except Exception as exc:
                observation = f"Error: {exc}"
            tool_call = assistant_message["tool_calls"][0]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"tool_{step}"),
                    "name": tool_call["function"]["name"],
                    "content": stringify_observation(observation),
                }
            )
            if action.name in env.terminate_tools:
                stop_reason = "terminate_tool"
                break
            continue

        messages.append({"role": "tool", "name": action.name, "content": f"Unknown action {action.name}"})
        stop_reason = "unknown_action"
        break

    metrics = compute_reward_metrics(env)
    return {
        "task_id": task_index,
        "instruction": env.task.instruction,
        "messages": messages,
        "trajectory_actions": [action.model_dump() for action in env.actions],
        "ground_truth_actions": [action.model_dump() for action in env.task.actions],
        "outputs": list(env.task.outputs),
        "data_source": f"tau-bench/{args.env}/{args.task_split}",
        "model": args.model_path,
        "stop_reason": stop_reason,
        "metrics": metrics,
        "user_backend": args.user_backend,
        "agent_backend": args.agent_backend,
        "total_user_cost": user_simulator.get_total_cost(),
        "error": error,
    }


async def run_one_task_safe(task_index: int, args, agent_responder: AsyncAgentResponder, user_responder: AsyncTextResponder) -> dict[str, Any]:
    try:
        return await run_one_task(task_index, args, agent_responder, user_responder)
    except Exception as exc:
        return {
            "task_id": task_index,
            "instruction": None,
            "messages": [],
            "trajectory_actions": [],
            "ground_truth_actions": [],
            "outputs": [],
            "data_source": f"tau-bench/{args.env}/{args.task_split}",
            "model": args.model_path,
            "stop_reason": "error",
            "metrics": {
                "reward": 0.0,
                "r_actions": 0.0,
                "gt_data_hash": None,
                "current_data_hash": None,
                "r_outputs": None,
                "outputs": {},
            },
            "user_backend": args.user_backend,
            "agent_backend": args.agent_backend,
            "total_user_cost": 0.0,
            "error": {
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }

