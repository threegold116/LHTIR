from __future__ import annotations

import os
import sys
import types
import logging


def ensure_tau2_importable(tau2_root: str) -> None:
    src_dir = os.path.join(tau2_root, "src")
    pkg_dir = os.path.join(src_dir, "tau2")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    os.environ.setdefault("TAU2_DATA_DIR", os.path.join(tau2_root, "data"))
    if "tau2" not in sys.modules:
        module = types.ModuleType("tau2")
        module.__path__ = [pkg_dir]
        module.__package__ = "tau2"
        sys.modules["tau2"] = module
    _ensure_basic_shims()


def _ensure_basic_shims() -> None:
    if "loguru" not in sys.modules:
        logger = logging.getLogger("tau2-shim")
        logging.basicConfig(level=logging.INFO)

        class _Logger:
            def debug(self, *args, **kwargs):
                logger.debug(*args, **kwargs)

            def info(self, *args, **kwargs):
                logger.info(*args, **kwargs)

            def warning(self, *args, **kwargs):
                logger.warning(*args, **kwargs)

            def error(self, *args, **kwargs):
                logger.error(*args, **kwargs)

            def success(self, *args, **kwargs):
                logger.info(*args, **kwargs)

            def exception(self, *args, **kwargs):
                logger.exception(*args, **kwargs)

        shim = types.ModuleType("loguru")
        shim.logger = _Logger()
        sys.modules["loguru"] = shim

    if "dotenv" not in sys.modules:
        shim = types.ModuleType("dotenv")

        def load_dotenv(*args, **kwargs):
            return False

        shim.load_dotenv = load_dotenv
        sys.modules["dotenv"] = shim

    if "deepdiff" not in sys.modules:
        shim = types.ModuleType("deepdiff")

        class DeepDiff(dict):
            def __init__(self, *args, **kwargs):
                super().__init__()

        shim.DeepDiff = DeepDiff
        sys.modules["deepdiff"] = shim
