import json
from pathlib import Path

from pydantic import BaseModel

CONFIG_FILE = Path(__file__).parent / "config.json"

PYBIND_DEFAULT = (
    "/home/ds/edge_llm/TensorRT-Edge-LLM/experimental/pybind/build"
)


class AppConfig(BaseModel):
    ENGINE_DIR: str = "/media/ds/DATA/engines/qwen25-vl-7b-4k-b1-10x448"
    PLUGIN_PATH: str = "/home/ds/edge_llm/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
    EDGELLM_PYBIND_DIR: str = PYBIND_DEFAULT


_state: AppConfig = AppConfig()


def get() -> AppConfig:
    return _state


def init() -> AppConfig:
    global _state
    if CONFIG_FILE.exists():
        _state = AppConfig.model_validate_json(CONFIG_FILE.read_text())
    else:
        _state = AppConfig()
        _save(_state)
    return _state


def _save(cfg: AppConfig) -> None:
    CONFIG_FILE.write_text(
        json.dumps(cfg.model_dump(), indent=2, ensure_ascii=False)
    )