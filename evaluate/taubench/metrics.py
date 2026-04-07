from __future__ import annotations

from typing import Any


def compute_reward_metrics(env) -> dict[str, Any]:
    """复现 tau-bench 官方语义，计算最终 reward 与各子指标。

    Descriptions:
        该函数分两步计算分数：
        1. `r_actions`：把模型执行后的当前环境状态 hash，与按标准动作序列
           重放后的环境状态 hash 进行比较，一致则为 1。
        2. `r_outputs`：如果任务定义了 `outputs`，则检查模型所有 `respond`
           动作中是否都包含这些关键字符串。匹配时忽略大小写，并去掉逗号。

        最终：
        - `reward = 1` 仅当状态正确且（如果存在）输出也全部命中。
        - 否则 `reward = 0`。

    Args:
        env: 已经执行完当前任务的 tau-bench 环境实例。

    Returns:
        一个字典，包含：
        - `reward`
        - `r_actions`
        - `gt_data_hash`
        - `current_data_hash`
        - `r_outputs`
        - `outputs`
    """
    current_data_hash = env.get_data_hash()
    original_data = env.data
    original_actions = list(env.actions)

    env.data = env.data_load_func()
    try:
        for action in env.task.actions:
            if action.name in env.terminate_tools:
                continue
            if action.name in env.tools_map:
                try:
                    env.tools_map[action.name].invoke(data=env.data, **action.kwargs)
                except Exception:
                    pass
        gt_data_hash = env.get_data_hash()
    finally:
        env.data = original_data

    r_actions = float(current_data_hash == gt_data_hash)
    outputs_hit = {}
    r_outputs = None
    if env.task.outputs:
        r_outputs = 1.0
        for output in env.task.outputs:
            found = False
            for action in original_actions:
                if action.name == "respond" and output.lower() in str(action.kwargs.get("content", "")).lower().replace(",", ""):
                    found = True
                    break
            outputs_hit[output] = found
            if not found:
                r_outputs = 0.0

    reward = 1.0 if r_actions == 1.0 and (r_outputs is None or r_outputs == 1.0) else 0.0
    return {
        "reward": reward,
        "r_actions": r_actions,
        "gt_data_hash": gt_data_hash,
        "current_data_hash": current_data_hash,
        "r_outputs": r_outputs,
        "outputs": outputs_hit,
    }
