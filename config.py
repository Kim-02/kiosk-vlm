import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

CONFIG_FILE = Path(__file__).parent / "config.json"

PYBIND_DEFAULT = (
    "/home/ds/edge_llm/TensorRT-Edge-LLM/experimental/pybind/build"
)


class AppConfig(BaseModel):
    # ── 런타임/엔진 경로 (변경 시 서버 재시작 필요) ──────────────────────────
    ENGINE_DIR: str = "/media/ds/DATA/engines/qwen25-vl-7b-4k-b5"
    PLUGIN_PATH: str = "/home/ds/edge_llm/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
    EDGELLM_PYBIND_DIR: str = PYBIND_DEFAULT

    # ── 프레임 처리 ──────────────────────────────────────────────────────────
    NUM_FRAMES: int = 4
    FRAME_SIZE: int = 1024
    JPEG_QUALITY: int = 95
    RESIZE_TMP_DIR: str = "/tmp/vlm_resized"

    # ── 샘플링 파라미터 ──────────────────────────────────────────────────────
    # TEMPERATURE=0.0(그리디)일 때는 TOP_P=1.0, TOP_K=1 이어야 경고/수치 불안정이 없다.
    MAX_TOKENS: int = 384
    TEMPERATURE: float = 0.0
    TOP_P: float = 1.0
    TOP_K: int = 1

    # ── 서버 ─────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000


_state: AppConfig = AppConfig()


def get() -> AppConfig:
    return _state


def init() -> AppConfig:
    global _state
    if CONFIG_FILE.exists():
        _state = AppConfig.model_validate_json(CONFIG_FILE.read_text())
        # 새 필드가 추가됐을 때 저장본을 최신 스키마로 보강
        _save(_state)
    else:
        _state = AppConfig()
        _save(_state)
    return _state


def update(patch: dict[str, Any]) -> AppConfig:
    """부분 업데이트. 알 수 없는 키는 거부한다."""
    global _state
    unknown = set(patch) - set(AppConfig.model_fields)
    if unknown:
        raise KeyError(f"알 수 없는 설정 키: {sorted(unknown)}")

    merged = _state.model_dump()
    for key, value in patch.items():
        if value is None:
            continue
        merged[key] = value

    _state = AppConfig.model_validate(merged)
    _save(_state)
    return _state


def _save(cfg: AppConfig) -> None:
    CONFIG_FILE.write_text(
        json.dumps(cfg.model_dump(), indent=2, ensure_ascii=False)
    )
