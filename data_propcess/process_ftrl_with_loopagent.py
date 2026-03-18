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
import os

import datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_train",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/data/ftrl/train.parquet",
        help="Path to the raw train parquet file.",
    )
    parser.add_argument(
        "--input_test",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/data/ftrl/test.parquet",
        help="Path to the raw test parquet file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/share/home/sxjiang/myproject/LHTIR/data/ftrl/with_agent_name",
        help="Directory to save processed train/test parquet files.",
    )
    return parser.parse_args()


def _check_file_exists(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} parquet not found: {path}")


def add_agent_name(example: dict) -> dict:
    example["agent_name"] = "tool_agent"
    return example


def main():
    args = parse_args()

    _check_file_exists(args.input_train, "Train")
    _check_file_exists(args.input_test, "Test")

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = datasets.load_dataset("parquet", data_files=args.input_train)["train"]
    test_dataset = datasets.load_dataset("parquet", data_files=args.input_test)["train"]

    train_dataset = train_dataset.map(add_agent_name)
    test_dataset = test_dataset.map(add_agent_name)

    train_output = os.path.join(args.output_dir, "train.parquet")
    test_output = os.path.join(args.output_dir, "test.parquet")

    train_dataset.to_parquet(train_output)
    test_dataset.to_parquet(test_output)


if __name__ == "__main__":
    main()
