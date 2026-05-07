from __future__ import annotations

def compute_reward_info(task, trajectory, environment_constructor, tool_types, env_kwargs=None) -> RewardInfo:
    from tau2.data_model.simulation import RewardInfo
    from tau2.data_model.tasks import RewardType
    from tau2.evaluator.evaluator_action import ActionEvaluator
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    if env_kwargs is None:
        env_kwargs = {}
    if task.evaluation_criteria is None:
        return RewardInfo(reward=1.0, reward_basis=None, info={"note": "No evaluation criteria"})

    env_reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=environment_constructor,
        task=task,
        full_trajectory=trajectory,
        solo_mode=False,
        env_kwargs=env_kwargs,
    )
    action_reward_info = ActionEvaluator.calculate_reward(
        task=task,
        full_trajectory=trajectory,
        tool_types=tool_types,
    )
    communicate_reward_info = CommunicateEvaluator.calculate_reward(
        task=task,
        full_trajectory=trajectory,
    )

    reward = 1.0
    reward_breakdown = {}
    reward_basis = task.evaluation_criteria.reward_basis
    reward_basis_set = set(reward_basis)

    env_bases = {RewardType.DB, RewardType.ENV_ASSERTION}
    if reward_basis_set & env_bases:
        if env_reward_info.reward_breakdown is not None:
            reward_breakdown.update(env_reward_info.reward_breakdown)
        reward *= env_reward_info.reward

    if RewardType.ACTION in reward_basis_set:
        if action_reward_info.reward_breakdown is not None:
            reward_breakdown.update(action_reward_info.reward_breakdown)
        reward *= action_reward_info.reward

    if RewardType.COMMUNICATE in reward_basis_set:
        if communicate_reward_info.reward_breakdown is not None:
            reward_breakdown.update(communicate_reward_info.reward_breakdown)
        reward *= communicate_reward_info.reward

    if RewardType.NL_ASSERTION in reward_basis_set:
        raise ValueError("NL assertions are not supported in evaluation_tau2bench_agent.py")

    return RewardInfo(
        reward=reward,
        db_check=env_reward_info.db_check,
        env_assertions=env_reward_info.env_assertions,
        action_checks=action_reward_info.action_checks,
        communicate_checks=communicate_reward_info.communicate_checks,
        reward_basis=reward_basis,
        reward_breakdown=reward_breakdown,
        info={
            "env": env_reward_info.info,
            "action": action_reward_info.info,
            "communicate": communicate_reward_info.info,
        },
    )
