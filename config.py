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
    ENGINE_DIR: str = "/media/ds/DATA/engines/qwen25-vl-7b-4k-b1-10x448"
    PLUGIN_PATH: str = "/home/ds/edge_llm/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
    EDGELLM_PYBIND_DIR: str = PYBIND_DEFAULT
    # 참조(few-shot) 이미지 경로. DANGER/SAFE/HARD CASE 3열 콜라주. (변경 시 재시작 필요)
    REFERENCE_IMAGE_PATH: str = (
        "/media/ds/DATA/image/tune_image.png"
    )

    # ── 프레임 처리 ──────────────────────────────────────────────────────────
    NUM_FRAMES: int = 15
    FRAME_SIZE: int = 448
    JPEG_QUALITY: int = 95
    RESIZE_TMP_DIR: str = "/tmp/vlm_resized"

    # ── 샘플링 파라미터 ──────────────────────────────────────────────────────
    MAX_TOKENS: int = 384
    TEMPERATURE: float = 0.0
    TOP_P: float = 0.9
    TOP_K: int = 50

    # ── 행동별 알림 임계값. 오탐 잦으면 올리고 미탐 잦으면 내린다. ───────────
    CONFIDENCE_THRESHOLD: dict[str, float] = Field(
        default_factory=lambda: {
            "helmet_off": 0.6,
            "cone_touch": 0.6,
            "fence_crossing": 0.6,
            "ladder_alone": 0.6,
        }
    )

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
    """부분 업데이트. 알 수 없는 키는 거부한다.

    CONFIDENCE_THRESHOLD는 dict이므로 기존 값에 병합한다(라벨 단위 부분 수정 지원).
    """
    global _state
    unknown = set(patch) - set(AppConfig.model_fields)
    if unknown:
        raise KeyError(f"알 수 없는 설정 키: {sorted(unknown)}")

    merged = _state.model_dump()
    for key, value in patch.items():
        if value is None:
            continue
        if key == "CONFIDENCE_THRESHOLD" and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value

    _state = AppConfig.model_validate(merged)
    _save(_state)
    return _state


def _save(cfg: AppConfig) -> None:
    CONFIG_FILE.write_text(
        json.dumps(cfg.model_dump(), indent=2, ensure_ascii=False)
    )
