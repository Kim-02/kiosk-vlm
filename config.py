import json
from pathlib import Path

from pydantic import BaseModel, Field

CONFIG_FILE = Path(__file__).parent / "config.json"

PYBIND_DEFAULT = (
    "/home/ds/edge_llm/TensorRT-Edge-LLM/experimental/pybind/build"
)


class AppConfig(BaseModel):
    ENGINE_DIR: str = "/media/ds/DATA/engines/qwen25-vl-7b-4k-b1-10x448"
    PLUGIN_PATH: str = "/home/ds/edge_llm/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
    EDGELLM_PYBIND_DIR: str = PYBIND_DEFAULT
    MAX_TOKENS: int = Field(default=128, ge=1, le=4096)
    TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    DEFAULT_PROMPT: str = (
        "다음 이미지들은 시간 순서대로 입력된 연속 프레임입니다. "
        "사용자가 요청한 기준에 따라 장면의 변화와 주요 행동을 간결하게 설명하세요."
    )
    NUM_WORKERS: int = Field(default=4, ge=1, le=16)


class ConfigUpdate(BaseModel):
    ENGINE_DIR: str | None = None
    PLUGIN_PATH: str | None = None
    EDGELLM_PYBIND_DIR: str | None = None
    MAX_TOKENS: int | None = Field(default=None, ge=1, le=4096)
    TEMPERATURE: float | None = Field(default=None, ge=0.0, le=2.0)
    DEFAULT_PROMPT: str | None = None
    NUM_WORKERS: int | None = Field(default=None, ge=1, le=16)


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


def apply(changes: dict) -> AppConfig:
    global _state
    _state = _state.model_copy(update=changes)
    _save(_state)
    return _state


def _save(cfg: AppConfig) -> None:
    CONFIG_FILE.write_text(
        json.dumps(cfg.model_dump(), indent=2, ensure_ascii=False)
    )
