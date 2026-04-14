from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from evaluate.tau2bench.bootstrap import ensure_tau2_importable


DEFAULT_FIRST_AGENT_MESSAGE = "Hi! How can I help you today?"


def _load_domain_modules(tau2_root: str, domain: str):
    ensure_tau2_importable(tau2_root)

    if domain == "mock":
        from tau2.domains.mock.environment import get_environment, get_tasks

        return get_environment, get_tasks
    if domain == "airline":
        from tau2.domains.airline.environment import get_environment, get_tasks

        return get_environment, get_tasks
    if domain == "retail":
        from tau2.domains.retail.environment import get_environment, get_tasks

        return get_environment, get_tasks
    if domain == "telecom":
        from tau2.domains.telecom.environment import (
            get_environment_manual_policy as get_environment,
        )
        from tau2.domains.telecom.environment import get_tasks

        return get_environment, get_tasks
    raise ValueError(f"Unsupported tau2 domain: {domain}")


def build_task_and_env(tau2_root: str, domain: str, task_split: str, task_index: int):
    get_environment, get_tasks = _load_domain_modules(tau2_root, domain)
    tasks = get_tasks(task_split)
    task = deepcopy(tasks[task_index])
    env = get_environment()
    return task, env


def get_task_count(tau2_root: str, domain: str, task_split: str) -> int:
    _, get_tasks = _load_domain_modules(tau2_root, domain)
    return len(get_tasks(task_split))


def get_environment_constructor(tau2_root: str, domain: str) -> Callable[..., Any]:
    get_environment, _ = _load_domain_modules(tau2_root, domain)
    return get_environment


def get_agent_tools(env) -> list:
    return env.get_tools()


def get_user_tools(env, task) -> list:
    try:
        return env.get_user_tools(include=task.user_tools)
    except Exception:
        return []


def get_tool_types(env) -> dict[str, Any]:
    tool_types = {}
    if getattr(env, "tools", None) is not None:
        for name in env.tools.get_tools().keys():
            tool_types[name] = env.tools.tool_type(name)
    if getattr(env, "user_tools", None) is not None:
        for name in env.user_tools.get_tools().keys():
            tool_types[name] = env.user_tools.tool_type(name)
    return tool_types

