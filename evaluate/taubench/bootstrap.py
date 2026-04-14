import sys
import types
from pathlib import Path


def ensure_tau_bench_importable(tau_root: str) -> Path:
    """确保外部 tau-bench 仓库可被当前评测代码导入。

    Descriptions:
        该函数会解析 `tau_root` 路径并检查其是否存在，然后把仓库根目录加入
        `sys.path`，使后续可以导入 `tau_bench.*` 模块。除此之外，它还会调用
        `ensure_litellm_shim`，为官方仓库中对 `litellm` 的导入提供一个最小兼容层。

    Args:
        tau_root: tau-bench 仓库根目录路径。

    Returns:
        规范化后的仓库根目录 `Path` 对象。
    """
    root = Path(tau_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"tau-bench root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    ensure_litellm_shim()
    return root


def ensure_litellm_shim() -> None:
    """向当前 Python 进程注入一个最小的 `litellm` 兼容模块。

    Descriptions:
        原始 tau-bench 在 import 链路中会依赖 `litellm`。为了复用它的环境与任务定义、
        同时避免真的走官方 `litellm` 推理路径，这里会在 `sys.modules` 中注册一个
        轻量 shim：
        - 提供 `completion` 占位函数，若被误调用则明确报错。
        - 提供 `provider_list`，满足部分导入时的校验需求。

    Args:
        None。

    Returns:
        None。
    """
    if "litellm" in sys.modules:
        return
    module = types.ModuleType("litellm")

    def completion(*args, **kwargs):
        raise RuntimeError(
            "tau-bench's litellm completion path is disabled in LHTIR. "
            "Use the isolated LHTIR tau-bench evaluator backends instead."
        )

    module.completion = completion
    module.provider_list = ["openai", "vllm", "local", "custom", "none"]
    sys.modules["litellm"] = module
