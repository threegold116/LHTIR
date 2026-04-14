from __future__ import annotations

import json
from typing import Any

from prog_env.utils.parse_output import parse_qwen


def _normalize_arguments(arguments: Any) -> dict:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        return json.loads(arguments)
    raise TypeError(f"Unsupported tool arguments type: {type(arguments)}")


def parse_vllm_output(raw_output: str) -> dict[str, Any]:
    return parse_qwen(raw_output, one_tool_only=False)


def build_tool_calls(message: dict[str, Any], requestor: str, tool_call_cls) -> list:
    tool_calls = []
    for idx, tool_call in enumerate(message.get("tool_calls") or []):
        function = tool_call["function"]
        tool_calls.append(
            tool_call_cls(
                id=tool_call.get("id", f"{requestor}_tool_{idx}"),
                name=function["name"],
                arguments=_normalize_arguments(function["arguments"]),
                requestor=requestor,
            )
        )
    return tool_calls


def normalize_openai_message(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    if isinstance(message, dict):
        return message
    raise TypeError(f"Unsupported message type: {type(message)}")

