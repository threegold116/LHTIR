import copy
import importlib
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any


SAVED_CLASS = {
    "BaseApi": ["wifi", "logged_in"],
    "MessageApi": ["inbox"],
    "ReminderApi": ["reminder_list"],
    "ReminderAPI": ["reminder_list"],
    "FoodPlatform": ["users", "logged_in_users", "orders"],
    "Finance": ["user_accounts", "is_logged_in", "deposit_history", "withdrawal_history", "loan_history", "orders", "holdings"],
    "Travel": ["users", "reservations"],
}

STATELESS_CLASSES: list[str] = []

CLASS_FILE_PATH_MAPPING = {
    "agent_multi_turn": {
        "zh": {
            "BaseApi": "model_inference.multi_turn.scenarioszh.phone_platform.base_api",
            "MessageApi": "model_inference.multi_turn.scenarioszh.phone_platform.message",
            "ReminderApi": "model_inference.multi_turn.scenarioszh.phone_platform.reminder",
            "FoodPlatform": "model_inference.multi_turn.scenarioszh.phone_platform.food_services",
            "Travel": "model_inference.multi_turn.scenarioszh.travel",
        },
        "en": {
            "BaseApi": "model_inference.multi_turn.scenariosen.phone_platform.base_api",
            "MessageApi": "model_inference.multi_turn.scenariosen.phone_platform.message",
            "ReminderApi": "model_inference.multi_turn.scenariosen.phone_platform.reminder",
            "FoodPlatform": "model_inference.multi_turn.scenariosen.phone_platform.food_services",
            "Travel": "model_inference.multi_turn.scenariosen.travel",
        },
    },
    "agent_multi_step": {
        "zh": {
            "BaseApi": "model_inference.multi_step.scenarioszh.phone_platform.base_api",
            "MessageApi": "model_inference.multi_step.scenarioszh.phone_platform.message",
            "ReminderApi": "model_inference.multi_step.scenarioszh.phone_platform.reminder",
            "FoodPlatform": "model_inference.multi_step.scenarioszh.phone_platform.food_services",
            "Travel": "model_inference.multi_step.scenarioszh.travel",
        },
        "en": {
            "BaseApi": "model_inference.multi_step.scenariosen.phone_platform.base_api",
            "MessageApi": "model_inference.multi_step.scenariosen.phone_platform.message",
            "ReminderApi": "model_inference.multi_step.scenariosen.phone_platform.reminder",
            "FoodPlatform": "model_inference.multi_step.scenariosen.phone_platform.food_services",
            "Travel": "model_inference.multi_step.scenariosen.travel",
        },
    },
}

_INSTANCE_CACHE: dict[tuple[str, str, str, str], Any] = {}


def ensure_acebench_root(acebench_root: str) -> None:
    """将 ACEBench 仓库根目录加入 `sys.path`，确保后续可以按模块路径动态导入场景类。

    Descriptions:
        ACEBench 的执行逻辑依赖其原仓库中的场景模块。该函数负责把外部
        `ACEBench` 根目录注入 Python 导入路径，使 `importlib.import_module`
        可以解析 `model_inference.*` 这类模块路径。

    Args:
        acebench_root: ACEBench 仓库根目录的绝对路径或相对路径。

    Returns:
        None。
    """
    acebench_root = str(Path(acebench_root).resolve())
    if acebench_root not in sys.path:
        sys.path.insert(0, acebench_root)


def _sanitize_model_name(model_name: str) -> str:
    """将模型名转换为适合做缓存键和全局实例名的安全字符串。

    Descriptions:
        原始模型名里可能包含 `-`、`.`、`/` 等字符，不适合作为变量名的一部分。
        这里统一替换为下划线，避免后续拼接实例名时产生非法标识符。

    Args:
        model_name: 原始模型名称或路径标识。

    Returns:
        处理后的安全字符串。
    """
    return model_name.replace("-", "_").replace(".", "_").replace("/", "_")


def _instance_key(scenario: str, model_name: str, test_entry_id: str, class_name: str) -> tuple[str, str, str, str]:
    """构造用于实例缓存的唯一键。

    Descriptions:
        ACEBench 的同一条样本在多轮执行过程中需要复用同一组场景实例。
        该函数把场景模式、模型名、测试样本 id 和类名组合成一个稳定键，
        用于 `_INSTANCE_CACHE` 的读写。

    Args:
        scenario: 评测场景名，如 `agent_multi_step` 或 `agent_multi_turn`。
        model_name: 当前模型标识。
        test_entry_id: 当前测试样本的条目标识。
        class_name: 场景类名，如 `FoodPlatform`、`Travel`。

    Returns:
        一个四元组缓存键。
    """
    return (scenario, _sanitize_model_name(model_name), test_entry_id, class_name)


def _process_method_calls(function_call_string: str, instance_mapping: dict[str, list[str]]) -> list[str]:
    """把未绑定实例的方法调用字符串扩展成可直接执行的绑定调用字符串列表。

    Descriptions:
        模型生成的调用通常是 `foo(x=1)` 这种“只有方法名、没有实例名”的形式。
        该函数会先抽取方法名，再去 `instance_mapping` 中查找有哪些实例实现了该方法，
        最终把它改写成 `instance_name.foo(x=1)` 这样的可执行表达式。

    Args:
        function_call_string: 模型生成的单条函数调用字符串。
        instance_mapping: 方法名到实例名列表的映射表。

    Returns:
        所有可能的绑定调用字符串列表。若方法名不存在，则返回空列表。
    """
    compiled_pattern = re.compile(r"\b([a-zA-Z_]\w*)\s*(?=\()")
    match = compiled_pattern.search(function_call_string)
    processed_string_list = []
    if match:
        match_start, match_end = match.span()
        before_match = function_call_string[:match_start]
        after_match = function_call_string[match_end:]
        func_name = match.group(1)
        if func_name in instance_mapping:
            for name in instance_mapping[func_name]:
                processed_string = before_match + f"{name}.{func_name}" + after_match
                processed_string_list.append(processed_string)
    return processed_string_list


def _serialize_execution_result(item: Any) -> str:
    """把单条执行结果统一序列化为字符串，并优先保持 JSON 可逆性。

    Descriptions:
        `simulator.py` 后续会尝试对执行结果做 `json.loads`，因此这里需要保证
        结构化结果在返回时尽量是合法 JSON 字符串。对 `dict`、`list`、基础标量
        等 JSON 可序列化对象，统一走 `json.dumps`；仅当对象无法 JSON 序列化时，
        才退回 `str(item)`。

    Args:
        item: 单条 API 调用的返回结果。

    Returns:
        字符串形式的执行结果；若原结果可 JSON 序列化，则返回合法 JSON 字符串。
    """
    if isinstance(item, str):
        return item
    try:
        return json.dumps(item, ensure_ascii=False)
    except Exception:
        return str(item)


def execute_agent_func_call(
    func_call_list: list[str],
    initial_config: dict,
    involved_classes: list[str],
    model_name: str,
    test_entry_id: str,
    language: str,
    scenario: str,
    acebench_root: str,
) -> tuple[list[str], dict[str, Any]]:
    """执行模型生成的一组 ACEBench API 调用，并返回执行结果与实例状态。

    Descriptions:
        这是 ACEBench 评测里的核心执行器。函数会根据 `involved_classes`
        动态导入对应场景类、初始化或复用缓存实例、把模型生成的未绑定调用
        改写成绑定到具体实例的方法调用，然后逐条执行。

        执行完成后，返回两部分信息：
        1. `execution_results`：每条调用的结果，统一转成字符串；如果结果本身是
           `dict`、`list` 等结构化对象，会优先转成合法 JSON 字符串，方便后续
           `json.loads` 还原。
        2. `involved_instances`：当前样本涉及到的所有场景实例，供后续做状态快照。

    Args:
        func_call_list: 解析后的函数调用列表，每个元素都是可执行的调用字符串。
        initial_config: 数据集给出的初始环境配置。
        involved_classes: 当前样本涉及的类名列表。
        model_name: 当前模型标识，用于构造实例缓存键。
        test_entry_id: 当前测试样本 id。
        language: 语言标识，`zh` 或 `en`。
        scenario: 评测场景名。
        acebench_root: ACEBench 仓库根目录。

    Returns:
        一个二元组：
        - `execution_results`: 每条调用对应的执行结果字符串列表。
        - `involved_instances`: 本轮执行涉及到的实例字典。
    """
    ensure_acebench_root(acebench_root)

    class_method_name_mapping: dict[str, list[str]] = {}
    involved_instances: dict[str, Any] = {}

    for class_name in involved_classes:
        module_name = CLASS_FILE_PATH_MAPPING[scenario][language][class_name]
        cache_key = _instance_key(scenario, model_name, test_entry_id, class_name)
        instance_name = "_".join(cache_key)
        if cache_key not in _INSTANCE_CACHE:
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            class_instance = class_()
            if class_name not in STATELESS_CLASSES:
                class_initial_config = initial_config.get(class_name, {})
                class_instance._load_scenario(copy.deepcopy(class_initial_config), long_context=False)
                class_initial_baseconfig = initial_config.get("BaseApi", {})
                class_instance._load_scenario(copy.deepcopy(class_initial_baseconfig), long_context=False)
            _INSTANCE_CACHE[cache_key] = class_instance
        else:
            class_instance = _INSTANCE_CACHE[cache_key]

        involved_instances[class_name] = class_instance

        for method_name, method in inspect.getmembers(class_instance, predicate=inspect.ismethod):
            if method_name.startswith("_"):
                continue
            class_method_name_mapping.setdefault(method_name, []).append(instance_name)
            globals()[instance_name] = class_instance

    execution_results = []
    for func_call in func_call_list:
        func_calls = _process_method_calls(func_call, class_method_name_mapping)
        if not func_calls:
            execution_results.append(f"Error during execution: unknown function in call {func_call}")
            continue
        try:
            for bound_func_call in func_calls:
                func_call_result = eval(bound_func_call)
            execution_results.append(func_call_result)
        except Exception as e:
            execution_results.append(f"Error during execution: {str(e)}")

    for idx, item in enumerate(execution_results):
        execution_results[idx] = _serialize_execution_result(item)

    return execution_results, involved_instances


def snapshot_instances(involved_instances: dict[str, Any]) -> list[dict[str, Any]]:
    """从场景实例中提取需要持久化的关键状态字段，生成可落盘的快照。

    Descriptions:
        ACEBench 的最终判分依赖环境状态，因此这里会根据 `SAVED_CLASS` 中配置的
        白名单字段，从每个实例的 `__dict__` 中提取关键状态，并转成 JSON 兼容
        的普通 Python 对象。

    Args:
        involved_instances: 当前样本涉及到的实例字典，键是类名，值是实例对象。

    Returns:
        一个由字典组成的列表，每个元素对应一个类的状态快照。
    """
    result = []
    for name, instance in involved_instances.items():
        item_dict = {}
        save_keys = SAVED_CLASS.get(name, SAVED_CLASS.get(name.replace("API", "Api"), []))
        for item in instance.__dict__:
            if item in save_keys:
                item_dict[item] = instance.__dict__[item]
        item_dict = json.loads(json.dumps(item_dict, ensure_ascii=False))
        output_name = "ReminderAPI" if name == "ReminderApi" else name
        result.append({output_name: item_dict})
    return result
