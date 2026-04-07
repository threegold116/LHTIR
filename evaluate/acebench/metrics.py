from typing import Any


def agent_checker(model_output: dict[str, Any], possible_answer: dict[str, Any]) -> dict[str, Any]:
    """比较单个类的预测状态与标准状态是否一致。

    Descriptions:
        该函数用于 ACEBench 的 end-to-end 判分。输入是某一个类的模型最终状态
        和对应的标准答案状态，函数会逐字段对比。如果字段缺失、子字段缺失、
        或字段值不一致，就会记录错误信息，并将 `valid` 设为 False。

    Args:
        model_output: 模型最终状态中某一个类对应的状态字典。
        possible_answer: 标准答案中同一个类对应的状态字典。

    Returns:
        一个包含校验结果的字典，主要字段包括：
        - `valid`: 是否完全匹配。
        - `error`: 错误信息列表。
        - `error_type`: 错误类型说明。
    """
    result = {
        "valid": True,
        "error": [],
        "error_type": "class attributes wrong",
    }
    scenario_name = list(possible_answer.keys())[0]
    possible_answer = list(possible_answer.values())[0]
    model_params = list(model_output.values())[0]

    for model_param, model_value in model_params.items():
        if model_param in possible_answer:
            possible_answer_value = possible_answer[model_param]
        else:
            result["valid"] = False
            result["error"].append(f"class({scenario_name}) attributes({model_param}) missing in possible_answer.")
            continue

        if isinstance(possible_answer_value, dict):
            for param, value in possible_answer_value.items():
                if param not in model_value:
                    result["valid"] = False
                    result["error"].append(
                        f"class({scenario_name}) attributes({model_param}) wrong, [expected: {possible_answer_value}, real: {model_value}]"
                    )
                elif value != model_value[param]:
                    result["valid"] = False
                    result["error"].append(
                        f"class({scenario_name}) attributes({model_param}.{param}) wrong, [expected: {value}, real: {model_value[param]}]"
                    )
        else:
            if possible_answer_value != model_value:
                result["valid"] = False
                result["error"].append(
                    f"class({scenario_name}) attributes({model_param}) wrong, [expected: {possible_answer_value}, real: {model_value}]"
                )
    return result


def compute_end_to_end_accuracy(final_state: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """根据最终环境状态与标准状态计算 end-to-end accuracy。

    Descriptions:
        该函数会先检查最终状态中的类数量是否与标准答案一致，然后逐个标准类
        在模型输出中寻找同名类，再调用 `agent_checker` 做字段级别比较。
        只要任意类缺失或字段不匹配，最终得分就是 0；全部匹配时得分为 1。

    Args:
        final_state: 模型执行结束后的环境状态快照列表。
        ground_truth: 数据集提供的标准最终状态列表。

    Returns:
        一个二元组：
        - 第一个元素是 end-to-end accuracy，取值为 0.0 或 1.0。
        - 第二个元素是错误信息列表。
    """
    errors = []
    is_valid = True
    if len(ground_truth) != len(final_state):
        return 0.0, ["wrong number of class"]

    for possible_item in ground_truth:
        possible_keys = set(possible_item.keys())
        matched_dict = None
        for model_dict in final_state:
            if set(model_dict.keys()) == possible_keys:
                matched_dict = model_dict
                break
        if matched_dict is None:
            is_valid = False
            errors.append(f"class missing: {list(possible_keys)}")
            continue
        checker_result = agent_checker(matched_dict, possible_item)
        if not checker_result["valid"]:
            is_valid = False
            errors.extend(checker_result["error"])
    return (1.0 if is_valid else 0.0), errors


def compute_process_accuracy(process: list[str], mile_stone: list[Any], end_to_end_accuracy: float) -> float:
    """根据执行轨迹和里程碑序列计算 process accuracy。

    Descriptions:
        如果 end-to-end 已经完全正确，则 process accuracy 直接记为 1。
        否则：
        - 若没有提供里程碑，也记为 1。
        - 若里程碑是多条可选路径，则取所有路径中得分最高的一条。
        - 若里程碑是单一路径，则直接按该路径计算。

    Args:
        process: 模型实际执行出的 API 调用轨迹。
        mile_stone: 数据集提供的里程碑列表，可能是单一路径，也可能是多条备选路径。
        end_to_end_accuracy: end-to-end accuracy 的结果。

    Returns:
        process accuracy 浮点数。
    """
    if end_to_end_accuracy == 1.0:
        return 1.0
    if not mile_stone:
        return 1.0
    if isinstance(mile_stone[0], list):
        return max(_single_process_accuracy(process, one_path) for one_path in mile_stone)
    return _single_process_accuracy(process, mile_stone)


def _single_process_accuracy(process: list[str], mile_stone: list[str]) -> float:
    """计算单条里程碑路径下的 process accuracy。

    Descriptions:
        该函数会按照顺序在 `process` 中查找每一个里程碑调用是否出现。
        匹配必须满足顺序一致，但不要求连续。最终分数是命中的里程碑数
        除以里程碑总数，并保留三位小数。

    Args:
        process: 模型执行出的 API 调用轨迹。
        mile_stone: 单条标准里程碑路径。

    Returns:
        当前里程碑路径下的过程得分。
    """
    if not mile_stone:
        return 1.0
    result_len = len(process)
    result_indices = []
    current_index = 0
    for stone in mile_stone:
        while current_index < result_len:
            if process[current_index].strip() == stone.strip():
                result_indices.append(current_index)
                current_index += 1
                break
            current_index += 1
    return round(len(result_indices) / len(mile_stone), 3)
