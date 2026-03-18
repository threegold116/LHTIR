# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import builtins
import os
import json
import re

import datasets
import random
import string
import keyword
from typing import Set
# 设置随机种子
random.seed(42)

# 全局已使用集合（可选）
_USED_NAMES: Set[str] = set()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_train",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/data/ftrl/with_agent_name/train.parquet",
        help="Path to the raw train parquet file.",
    )
    parser.add_argument(
        "--input_test",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/data/ftrl/with_agent_name/test.parquet",
        help="Path to the raw test parquet file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/data/ftrl/with_agent_name_mask",
        help="Directory to save processed train/test parquet files.",
    )
    return parser.parse_args()


def _check_file_exists(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} parquet not found: {path}")


def add_agent_name(example: dict) -> dict:
    example["agent_name"] = "tool_agent"
    return example
def random_code(length: int = 6) -> str:
    """
    生成符合 Python 标识符规范的随机名字：
    - 第一个字符：小写字母
    - 后续字符：小写字母或数字
    """
    if length <= 0:
        raise ValueError("length must be greater than 0")
    first_chars = string.ascii_lowercase
    used_set = _USED_NAMES
    charset =  first_chars + string.digits + "_"
    builtins_names = set(dir(builtins))
    while True:
        # 生成随机名字
        first = random.choice(first_chars)
        if length == 1:
            name = first
        else:
            rest = ''.join(random.choices(charset, k=length - 1))
            name = first + rest

        # 过滤规则
        if keyword.iskeyword(name):
            continue

        if name in builtins_names:
            continue

        if name in used_set:
            continue

        # 记录唯一性
        used_set.add(name)
        return name
def mask_function_name(example: dict,mask_function_name:bool=True,mask_parameter_name:bool=False) -> dict:
    """
    结合论文中 “Concretely, we randomize tool names as well as their
    parameter names” 的思路，对单条样本中的工具名和参数名做脱敏 / 置换。

    - 输入样本需包含两列：
      - example["tools"]: JSON 字符串，类似 OpenAI tools schema 列表
      - example["codes"]: JSON 字符串，形如 {function_name: source_code_str}
    - 我们会：
      1）将每个 function.name 映射为 tool_{i}
      2）将 schema 中的参数名映射为 arg_{j}
      3）将 codes 里的键名从旧的 function name 映射到新的名字
    - 为了安全起见，不在源码字符串内部做复杂文本替换（避免误替换变量名、字符串等），
      因此这一步主要用于训练时的语义脱敏，而非真实执行环境。
    """
    tools_raw = example.get("tools")
    codes_raw = example.get("codes")

    if not tools_raw or not codes_raw:
        return example

    try:
        tools = json.loads(tools_raw)
        codes = json.loads(codes_raw)
    except Exception:
        # 若解析失败，保持原样
        return example

    if not isinstance(tools, list) or not isinstance(codes, dict):
        return example

    # 1. 为每个工具构造新的函数名，并同时重命名参数
    func_name_map = {}

    for idx, tool in enumerate(tools):
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue

        old_name = fn.get("name")
        if not isinstance(old_name, str) or not old_name:
            continue

        # new_name = f"tool_{idx}"
        new_name = random_code()
        func_name_map[old_name] = new_name
        fn["name"] = new_name

        # 处理参数名：properties / required
        params = fn.get("parameters")
        if not isinstance(params, dict):
            continue

        props = params.get("properties")
        if not isinstance(props, dict):
            continue
        if mask_parameter_name:
            new_props = {}
            param_name_map = {}
            for p_idx, (p_name, p_schema) in enumerate(props.items()):
                new_p_name = f"arg_{p_idx}"
                param_name_map[p_name] = new_p_name
                new_props[new_p_name] = p_schema
            params["properties"] = new_props

            required = params.get("required")
            if isinstance(required, list):
                params["required"] = [
                    param_name_map.get(r_name, r_name) for r_name in required
                ]

    # 2. 重命名 codes 中的键（函数名 -> 新函数名），并同步修改代码里的函数定义名
    new_codes = {}
    for old_name, src in codes.items():
        new_name = func_name_map.get(old_name, old_name)

        # 如果函数名有发生变化，在源码字符串中把 `def old_name(...)` 也改掉
        if new_name != old_name and isinstance(src, str):
            pattern = re.compile(rf"(\bdef\s+){re.escape(old_name)}(\s*\()")

            # 注意：不能在替换串里直接用 \1{new_name}\2，否则类似 \1a 会被解释成 group 1a 或 group 16。
            # 使用函数式替换，显式拼接分组内容，避免无效的 group 引用错误。
            def _repl(m):
                return m.group(1) + new_name + m.group(2)

            src = pattern.sub(_repl, src)

        new_codes[new_name] = src

    # 3. 对ground_truth做脱敏
    if "ground_truth" in example:
        ground_truth_raw = example.get("ground_truth")
        ground_truth = json.loads(ground_truth_raw)
        new_ground_truth = []
        for item in ground_truth:
            new_item = item.copy()
            for old_name, new_name in func_name_map.items():
                if old_name==new_item["name"]:
                    new_item["name"] = new_name
                    break
            new_ground_truth.append(new_item)
        example["ground_truth"] = json.dumps(new_ground_truth, ensure_ascii=False)
        
    # 4. 写回样本（保持为 JSON 字符串）
    example["tools"] = json.dumps(tools, ensure_ascii=False)
    example["codes"] = json.dumps(new_codes, ensure_ascii=False)

    return example


def main():
    args = parse_args()

    _check_file_exists(args.input_train, "Train")
    _check_file_exists(args.input_test, "Test")

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = datasets.load_dataset("parquet", data_files=args.input_train)["train"]
    test_dataset = datasets.load_dataset("parquet", data_files=args.input_test)["train"]

    # 先补充 agent_name，再做工具名 / 参数名脱敏
    train_dataset = train_dataset.map(mask_function_name, load_from_cache_file=False)
    test_dataset = test_dataset.map(mask_function_name, load_from_cache_file=False)

    train_output = os.path.join(args.output_dir, "train.parquet")
    test_output = os.path.join(args.output_dir, "test.parquet")

    train_dataset.to_parquet(train_output)
    test_dataset.to_parquet(test_output)


if __name__ == "__main__":
    main()
