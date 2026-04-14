"""
 Copyright 2025 Bytedance Ltd. and/or its affiliates

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import re
# import json_repair
from hashlib import sha256
import datetime
import json
from logging import getLogger

logger = getLogger(__name__)

def parse_qwen(inputs: str, one_tool_only=False):
    output = {"role": "assistant", "content": inputs.strip(), "tool_calls": []}

    pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    matches = pattern.findall(inputs)

    if not matches:
        output["tool_calls"] = None
        return output

    if one_tool_only:
        matches = matches[:1]

    for m in matches:
        try:
            json_str = m.strip()
            tool_call = json.loads(json_str)

            if not isinstance(tool_call.get('arguments', {}), dict):
                tool_call['arguments'] = json.loads(tool_call['arguments'])

            output["tool_calls"].append({
                "type": "function",
                "function": {
                    "name": tool_call['name'],
                    "arguments": json.dumps(tool_call['arguments'], ensure_ascii=False)
                }
            })
        except Exception as e:
            logger.error("Parse error: %s", e)
            continue

    if not output["tool_calls"]:
        output["tool_calls"] = None

    return output


def get_parse_output(model):
    if model == "qwen":
        return parse_qwen
    else:
        raise NotImplementedError
