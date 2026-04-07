from __future__ import annotations

from typing import Any

from evaluate.taubench.bootstrap import ensure_tau_bench_importable


def build_env(tau_root: str, env_name: str, task_split: str, task_index: int | None = None):
    """构建 tau-bench 指定 domain 的环境实例。

    Descriptions:
        该函数负责按 domain 名称创建官方环境对象。当前只支持 `retail`
        和 `airline` 两个环境，并统一先用 `human` user strategy 初始化，
        以便后续由 LHTIR 自己的 user simulator 接管对话流程。

    Args:
        tau_root: tau-bench 仓库根目录。
        env_name: 环境名称，当前支持 `retail` 和 `airline`。
        task_split: 数据划分，当前通常为 `test`。
        task_index: 可选的任务索引；若传入则初始化到指定任务。

    Returns:
        对应 domain 的 tau-bench 环境实例。
    """
    ensure_tau_bench_importable(tau_root)
    if env_name == "retail":
        from tau_bench.envs.retail.env import MockRetailDomainEnv

        return MockRetailDomainEnv(
            user_strategy="human",
            user_model="human",
            task_split=task_split,
            task_index=task_index,
        )
    if env_name == "airline":
        from tau_bench.envs.airline.env import MockAirlineDomainEnv

        return MockAirlineDomainEnv(
            user_strategy="human",
            user_model="human",
            task_split=task_split,
            task_index=task_index,
        )
    raise ValueError(f"Unsupported tau-bench env: {env_name}")


def get_task_count(tau_root: str, env_name: str, task_split: str) -> int:
    """获取指定 tau-bench 环境与数据划分下的任务数量。

    Descriptions:
        该函数会先构造一个环境实例，再直接读取 `env.tasks` 的长度，用于
        evaluator 在主循环中计算 `start_id/end_id` 和切 batch。

    Args:
        tau_root: tau-bench 仓库根目录。
        env_name: 环境名称。
        task_split: 数据划分名称。

    Returns:
        当前环境下的任务总数。
    """
    env = build_env(tau_root, env_name, task_split, task_index=0)
    return len(env.tasks)


def stringify_observation(observation: Any) -> str:
    """把环境返回的 observation 统一转成字符串。

    Descriptions:
        tau-bench 的工具返回值大多数是字符串，但也可能出现其他 Python 对象。
        为了兼容后续统一写入 `messages`，这里对非字符串结果统一调用 `str(...)`。

    Args:
        observation: 环境 step 或工具调用返回的原始结果。

    Returns:
        字符串形式的 observation。
    """
    if isinstance(observation, str):
        return observation
    return str(observation)
