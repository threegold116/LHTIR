import ast
from typing import Any


def extract_outermost_bracket_content(text: str) -> str | None:
    stack: list[int] = []
    candidates: list[str] = []
    # 找所有的[]片段
    for i, char in enumerate(text):
        if char == "[":
            stack.append(i)
        elif char == "]" and stack:
            start = stack.pop()
            candidates.append(text[start : i + 1])

    # 从后往前找到最后一个，且在ast语义上是函数调用的
    for candidate in reversed(candidates):
        try:
            parsed = ast.parse(candidate, mode="eval")
        except Exception:
            continue
        body = parsed.body
        if isinstance(body, ast.Call):
            return candidate
        if isinstance(body, (ast.Tuple, ast.List)) and body.elts and all(
            isinstance(elem, ast.Call) for elem in body.elts
        ):
            return candidate
    return None


def resolve_ast_call(elem: ast.Call) -> dict[str, Any]:
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        args_dict[arg.arg] = resolve_ast_by_type(arg.value)
    return {func_name: args_dict}


def resolve_ast_by_type(value: ast.AST) -> Any:
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            return "..."
        return value.value
    if isinstance(value, ast.UnaryOp):
        return -value.operand.value
    if isinstance(value, ast.List):
        return [resolve_ast_by_type(v) for v in value.elts]
    if isinstance(value, ast.Dict):
        return {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    if isinstance(value, ast.NameConstant):
        return value.value
    if isinstance(value, ast.BinOp):
        return eval(ast.unparse(value))
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            return ast.unparse(value)
        return resolve_ast_call(value)
    if isinstance(value, ast.Tuple):
        return tuple(resolve_ast_by_type(v) for v in value.elts)
    if isinstance(value, ast.Lambda):
        return eval(ast.unparse(value.body[0].value))
    if isinstance(value, ast.Ellipsis):
        return "..."
    if isinstance(value, ast.Subscript):
        try:
            return ast.unparse(value.body[0].value)
        except Exception:
            return ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    raise Exception(f"Unsupported AST type: {type(value)}")


def ast_parse(input_str: str) -> list[dict[str, Any]]:
    parsed = ast.parse(input_str.strip("[]'"), mode="eval")
    extracted = []
    if isinstance(parsed.body, ast.Call):
        extracted.append(resolve_ast_call(parsed.body))
    elif isinstance(parsed.body, (ast.Tuple, ast.List)):
        for elem in parsed.body.elts:
            if not isinstance(elem, ast.Call):
                raise ValueError("Element is not a function call")
            extracted.append(resolve_ast_call(elem))
    return extracted


def parse_nested_value(value: Any) -> str:
    if isinstance(value, dict):
        func_name = list(value.keys())[0]
        args = value[func_name]
        args_str = ", ".join(f"{k}={parse_nested_value(v)}" for k, v in args.items())
        return f"{func_name}({args_str})"
    return repr(value)


def decoded_output_to_execution_list(decoded_output: list[dict[str, Any]]) -> list[str]:
    execution_list = []
    for function_call in decoded_output:
        for key, value in function_call.items():
            args_str = ", ".join(f"{k}={parse_nested_value(v)}" for k, v in value.items())
            execution_list.append(f"{key}({args_str})")
    return execution_list


def decode_function_list(result: str) -> list[str]:
    func = result.strip()
    if not func:
        return []
    if func[0] == " ":
        func = func[1:]
    if not func.startswith("["):
        func = "[" + func
    if not func.endswith("]"):
        func = func + "]"
    decoded_output = ast_parse(func)
    return decoded_output_to_execution_list(decoded_output)


def looks_like_function_call(text: str) -> bool:
    outer = extract_outermost_bracket_content(text)
    if outer is None:
        return False
    try:
        decode_function_list(outer)
        return True
    except Exception:
        return False
