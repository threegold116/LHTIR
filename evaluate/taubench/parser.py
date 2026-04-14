import json
from typing import Any, Dict

from prog_env.utils.parse_output import parse_qwen


def normalize_openai_message(message: Any) -> Dict[str, Any]:
    """把 OpenAI SDK 返回的 message 对象规范化成普通字典。

    Descriptions:
        OpenAI-compatible 接口返回的消息对象可能是 SDK 模型对象，也可能已经是
        Python 字典。该函数负责统一转换为普通字典，方便后续用同一套逻辑解析
        `content` 和 `tool_calls`。

    Args:
        message: OpenAI SDK 返回的消息对象或已规范化的字典。

    Returns:
        普通 Python 字典格式的 assistant message。
    """
    if hasattr(message, "model_dump"):
        return message.model_dump()
    if isinstance(message, dict):
        return message
    raise TypeError(f"Unsupported message type: {type(message)}")


def parse_vllm_agent_output(raw_output: str) -> Dict[str, Any]:
    """解析本地 vLLM 文本输出，转成统一的 assistant message 结构。

    Descriptions:
        当前 tau-bench 的 vLLM 路径沿用了项目里 Qwen 的 `<tool_call>...</tool_call>`
        解析逻辑。该函数会把原始文本解析成带 `role/content/tool_calls` 的标准字典，
        并限制一次只取一个工具调用。

    Args:
        raw_output: 模型返回的原始文本。

    Returns:
        统一格式的 assistant message 字典。
    """
    return parse_qwen(raw_output, one_tool_only=True)


def action_from_assistant_message(message: Dict[str, Any], action_cls, respond_action_name: str):
    """把 assistant message 转换为 tau-bench 的 `Action` 对象。

    Descriptions:
        如果 message 中包含 `tool_calls`，则提取第一个工具调用，解析参数后构造
        对应的工具动作；否则把 `content` 包装成普通回复动作，例如 `respond`。

    Args:
        message: 统一格式的 assistant message 字典。
        action_cls: tau-bench 中的 `Action` 类，用于实例化动作对象。
        respond_action_name: 普通回复动作的名称，通常是 `respond`。

    Returns:
        一个 tau-bench `Action` 实例。
    """
    tool_calls = message.get("tool_calls")
    if tool_calls:
        tool_call = tool_calls[0]
        arguments = tool_call["function"]["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return action_cls(name=tool_call["function"]["name"], kwargs=arguments)
    return action_cls(name=respond_action_name, kwargs={"content": message.get("content", "")})
