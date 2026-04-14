from __future__ import annotations

import traceback
from copy import deepcopy
from datetime import datetime
from typing import Any

from prog_env.utils.chatvllm import ChatVLLM
from prog_env.utils.llm_server import AsyncLLMServer

from evaluate.tau2bench.env_adapter import (
    DEFAULT_FIRST_AGENT_MESSAGE,
    build_task_and_env,
    get_agent_tools,
    get_environment_constructor,
    get_tool_types,
    get_user_tools,
)
from evaluate.tau2bench.metrics import compute_reward_info
from evaluate.tau2bench.parser import build_tool_calls, parse_vllm_output


AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

AGENT_SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

USER_SYSTEM_PROMPT = """
{global_user_sim_guidelines}

<scenario>
{instructions}
</scenario>
""".strip()


class AsyncVLLMResponder:
    def __init__(
        self,
        chat_engine: ChatVLLM,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.chat_engine = chat_engine
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(self, messages: list[dict], tools: list[dict] | None) -> dict[str, Any]:
        assert isinstance(self.chat_engine.engine, AsyncLLMServer)
        raw_output = await self.chat_engine.engine.chat_one_async(
            messages,
            tools,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return parse_vllm_output(raw_output)


def _load_user_guidelines(tau2_root: str, use_tools: bool) -> str:
    filename = "simulation_guidelines_tools.md" if use_tools else "simulation_guidelines.md"
    path = f"{tau2_root}/data/tau2/user_simulator/{filename}"
    with open(path, "r", encoding="utf8") as f:
        return f.read().strip()


def _to_openai_tool_schema(tools: list) -> list[dict]:
    return [tool.openai_schema for tool in tools]


def _assistant_to_openai_message(message) -> dict[str, Any]:
    payload = {"role": "assistant", "content": message.content}
    if getattr(message, "tool_calls", None):
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": __import__("json").dumps(tool_call.arguments, ensure_ascii=False),
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def _tool_to_openai_message(message) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": message.id,
        "content": message.content,
    }


def _build_agent_messages(trajectory, system_prompt: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for message in trajectory:
        role = getattr(message, "role", None)
        if role == "assistant":
            messages.append(_assistant_to_openai_message(message))
        elif role == "user" and not message.is_tool_call():
            messages.append({"role": "user", "content": message.content})
        elif role == "tool" and getattr(message, "requestor", None) == "assistant":
            messages.append(_tool_to_openai_message(message))
    return messages


def _build_user_messages(trajectory, system_prompt: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for message in trajectory:
        role = getattr(message, "role", None)
        if role == "user":
            messages.append(_assistant_to_openai_message(message))
        elif role == "assistant" and not message.is_tool_call():
            messages.append({"role": "user", "content": message.content})
        elif role == "tool" and getattr(message, "requestor", None) == "user":
            messages.append(_tool_to_openai_message(message))
    return messages


def _get_initial_trajectory(task, message_classes):
    AssistantMessage, _UserMessage, _ToolMessage = message_classes
    if task.initial_state is not None and task.initial_state.message_history is not None:
        return deepcopy(task.initial_state.message_history)
    return [AssistantMessage(role="assistant", content=DEFAULT_FIRST_AGENT_MESSAGE)]


def _resolve_next_role(last_message, user_stop_fn) -> str:
    if last_message.role == "assistant":
        if getattr(last_message, "tool_calls", None):
            return "env"
        if last_message.content and "###STOP###" in last_message.content:
            return "done_agent"
        return "user"
    if last_message.role == "user":
        if user_stop_fn(last_message):
            return "done_user"
        if getattr(last_message, "tool_calls", None):
            return "env"
        return "agent"
    if last_message.role == "tool":
        return "agent" if last_message.requestor == "assistant" else "user"
    raise ValueError(f"Unsupported last message role: {last_message.role}")


def _serialize_messages(messages: list) -> list[dict]:
    payload = []
    for message in messages:
        payload.append(message.model_dump())
    return payload


async def run_one_task(task_index: int, args, agent_responder: AsyncVLLMResponder, user_responder: AsyncVLLMResponder) -> dict[str, Any]:
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage, ToolCall

    task, env = build_task_and_env(args.tau2_root, args.domain, args.task_split, task_index)
    tool_types = get_tool_types(env)
    environment_constructor = get_environment_constructor(args.tau2_root, args.domain)

    initial_state = task.initial_state
    initialization_data = initial_state.initialization_data if initial_state is not None else None
    initialization_actions = initial_state.initialization_actions if initial_state is not None else None
    message_history = deepcopy(initial_state.message_history) if initial_state is not None and initial_state.message_history is not None else []
    env.set_state(initialization_data, initialization_actions, message_history)

    trajectory = _get_initial_trajectory(task, (AssistantMessage, UserMessage, ToolMessage))
    agent_tools = _to_openai_tool_schema(get_agent_tools(env))
    user_tools = _to_openai_tool_schema(get_user_tools(env, task))
    agent_system_prompt = AGENT_SYSTEM_PROMPT.format(
        agent_instruction=AGENT_INSTRUCTION,
        domain_policy=env.get_policy(),
    )
    user_guidelines = _load_user_guidelines(args.tau2_root, use_tools=bool(user_tools))
    user_system_prompt = USER_SYSTEM_PROMPT.format(
        global_user_sim_guidelines=user_guidelines,
        instructions=str(task.user_scenario),
    )

    next_role = _resolve_next_role(trajectory[-1], user_stop_fn=lambda msg: "###STOP###" in (msg.content or "") or "###TRANSFER###" in (msg.content or "") or "###OUT-OF-SCOPE###" in (msg.content or ""))
    start_time = datetime.now()
    stop_reason = "max_steps"

    for _step in range(args.max_steps):
        if next_role == "done_agent":
            stop_reason = "agent_stop"
            break
        if next_role == "done_user":
            stop_reason = "user_stop"
            break

        if next_role == "env":
            last_message = trajectory[-1]
            tool_results = [env.get_response(tool_call) for tool_call in last_message.tool_calls]
            trajectory.extend(tool_results)
            next_role = "agent" if tool_results[0].requestor == "assistant" else "user"
            continue

        if next_role == "agent":
            agent_messages = _build_agent_messages(trajectory, agent_system_prompt)
            assistant_dict = await agent_responder.generate(agent_messages, agent_tools)
            assistant_tool_calls = build_tool_calls(assistant_dict, "assistant", ToolCall)
            assistant_message = AssistantMessage(
                role="assistant",
                content=assistant_dict.get("content"),
                tool_calls=assistant_tool_calls or None,
            )
            trajectory.append(assistant_message)
            next_role = _resolve_next_role(assistant_message, user_stop_fn=lambda _msg: False)
            continue

        if next_role == "user":
            user_messages = _build_user_messages(trajectory, user_system_prompt)
            user_dict = await user_responder.generate(user_messages, user_tools or None)
            user_tool_calls = build_tool_calls(user_dict, "user", ToolCall)
            user_message = UserMessage(
                role="user",
                content=user_dict.get("content"),
                tool_calls=user_tool_calls or None,
            )
            trajectory.append(user_message)
            next_role = _resolve_next_role(
                user_message,
                user_stop_fn=lambda msg: "###STOP###" in (msg.content or "") or "###TRANSFER###" in (msg.content or "") or "###OUT-OF-SCOPE###" in (msg.content or ""),
            )
            continue

        raise ValueError(f"Unsupported next role: {next_role}")

    reward_info = compute_reward_info(
        task=task,
        trajectory=trajectory,
        environment_constructor=environment_constructor,
        tool_types=tool_types,
    )
    end_time = datetime.now()

    metrics = {
        "reward": reward_info.reward,
        "reward_basis": [item.value if hasattr(item, "value") else str(item) for item in (reward_info.reward_basis or [])],
        "reward_breakdown": {
            (key.value if hasattr(key, "value") else str(key)): value
            for key, value in (reward_info.reward_breakdown or {}).items()
        },
        "db_match": reward_info.db_check.db_match if reward_info.db_check is not None else None,
        "db_reward": reward_info.db_check.db_reward if reward_info.db_check is not None else None,
        "num_action_checks": len(reward_info.action_checks or []),
        "num_communicate_checks": len(reward_info.communicate_checks or []),
        "num_env_assertions": len(reward_info.env_assertions or []),
    }

    return {
        "task_id": task.id,
        "task_index": task_index,
        "instruction": str(task.user_scenario),
        "messages": _serialize_messages(trajectory),
        "data_source": f"tau2-bench/{args.domain}/{args.task_split}",
        "model": args.model_path,
        "domain": args.domain,
        "task_split": args.task_split,
        "stop_reason": stop_reason,
        "metrics": metrics,
        "reward_info": reward_info.model_dump(),
        "agent_backend": "vllm",
        "user_backend": "vllm",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration": (end_time - start_time).total_seconds(),
    }


async def run_one_task_safe(task_index: int, args, agent_responder: AsyncVLLMResponder, user_responder: AsyncVLLMResponder) -> dict[str, Any]:
    try:
        return await run_one_task(task_index, args, agent_responder, user_responder)
    except Exception as exc:
        return {
            "task_id": None,
            "task_index": task_index,
            "instruction": None,
            "messages": [],
            "data_source": f"tau2-bench/{args.domain}/{args.task_split}",
            "model": args.model_path,
            "domain": args.domain,
            "task_split": args.task_split,
            "stop_reason": "error",
            "metrics": {
                "reward": 0.0,
                "reward_basis": None,
                "reward_breakdown": None,
                "db_match": None,
                "db_reward": None,
                "num_action_checks": 0,
                "num_communicate_checks": 0,
                "num_env_assertions": 0,
            },
            "reward_info": None,
            "agent_backend": "vllm",
            "user_backend": "vllm",
            "error": {
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }

