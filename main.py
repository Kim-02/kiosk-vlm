import asyncio
import importlib
import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config as cfg_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

NUM_FRAMES = 15
FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_FRAME_RE = re.compile(r"^frame_(\d+)$|^frame(\d+)$")

# ── 프롬프트 / 샘플링 ────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a vision-language model for CCTV safety monitoring. Answer in Korean."
)
VLM_PROMPT_BASE = (
    "다음 15장의 CCTV 프레임은 시간 순서대로 촬영된 공장 작업 현장 연속 장면이다. "
    "각 프레임을 개별로 보지 말고 전체 흐름을 하나의 장면으로 분석하라. "
    "장면에 보이는 환경, 사람, 장비, 물체를 묘사하고 "
    "사람이 어떤 행동을 하고 있는지 구체적으로 서술하라."
)
VLM_FOCUS_PREFIX = " 특히 다음 항목에 주의하여 관찰하라: "

VLM_MAX_TOKENS = 256
LLM_MAX_TOKENS = 200
TEMPERATURE = 0.2
TOP_P = 0.9
TOP_K = 50

LLM_PROMPT_TEMPLATE = (
    "현장 설명:\n{vlm_output}\n\n"
    "위 설명에서 아래 위험 행동을 감지하라.\n"
    "{detect_items}\n\n"
    "JSON만 출력하라.\n"
    '{{"action":"감지키를_쉼표구분","tts_message":"경고문"}}\n'
    "action: 감지된 키만 나열. 없으면 빈 문자열.\n"
    "tts_message: 위험 사유와 함께 행동을 중단하십시오로 끝나는 문장. 없으면 빈 문자열."
)

DEFAULT_DETECT_ACTIONS: list[dict] = [
    {"key": "hat_action", "label": "안전모 미착용 또는 벗는 행동"},
    {"key": "touch_action", "label": "스피커를 만지는 행동"},
    {"key": "dangerInOut_action", "label": "금지 구역 출입"},
    {"key": "ladder_action", "label": "사다리를 올라가거나 단독 사다리 작업"},
]

# ── 런타임 ────────────────────────────────────────────────────────────────────
_runtime: Any = None
_edgellm: Any = None
_lock = asyncio.Lock()


def _load_edgellm_module(pybind_dir: str) -> Any:
    if pybind_dir not in sys.path:
        sys.path.insert(0, pybind_dir)
    return importlib.import_module("_edgellm_runtime")


# ── 앱 수명주기 ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI):
    global _runtime, _edgellm

    cfg = cfg_module.init()
    os.environ["EDGELLM_PLUGIN_PATH"] = cfg.PLUGIN_PATH

    log.info("_edgellm_runtime 모듈 로드 중... (%s)", cfg.EDGELLM_PYBIND_DIR)
    _edgellm = _load_edgellm_module(cfg.EDGELLM_PYBIND_DIR)

    log.info("LLMRuntime 엔진 로드 중... (%s)", cfg.ENGINE_DIR)
    _runtime = _edgellm.LLMRuntime(
        engine_dir=cfg.ENGINE_DIR,
        multimodal_engine_dir=cfg.ENGINE_DIR,
    )
    log.info("LLMRuntime 로드 완료")

    yield

    _runtime = None


app = FastAPI(title="DueGo VLM Server", lifespan=lifespan)


# ── 스키마 ────────────────────────────────────────────────────────────────────
class DetectAction(BaseModel):
    key: str
    label: str


class AnalyzeRequest(BaseModel):
    dir_path: str
    focus: Optional[str] = None
    detect_actions: Optional[list[DetectAction]] = None


class AnalyzeResponse(BaseModel):
    request_id: str
    action: str
    tts_message: str
    vlm_description: str
    elapsed_sec: float


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def collect_frames(folder: Path) -> list[Path]:
    candidates: list[tuple[int, Path]] = []
    for f in folder.iterdir():
        if f.suffix.lower() not in FRAME_EXTS:
            continue
        m = _FRAME_RE.fullmatch(f.stem)
        if m:
            num = int(m.group(1) or m.group(2))
            candidates.append((num, f))
    candidates.sort(key=lambda x: x[0])
    return [p for _, p in candidates]


def select_frames(frames: list[Path], target: int = NUM_FRAMES) -> list[Path]:
    n = len(frames)
    if n <= target:
        return frames
    indices = [round(i * (n - 1) / (target - 1)) for i in range(target)]
    return [frames[i] for i in indices]


def _build_gen_kwargs(
    batch_messages: list, max_tokens: int,
    temperature: float, top_p: float, top_k: int,
) -> dict:
    return dict(
        batch_messages=batch_messages,
        temperature=temperature,
        max_generate_length=max_tokens,
        top_p=top_p,
        top_k=top_k,
        apply_chat_template=True,
        add_generation_prompt=True,
    )


def _sync_vlm_infer(frame_paths: list[Path], prompt: str, max_tokens: int) -> str:
    messages = [
        _edgellm.Message("system", [_edgellm.MessageContent("text", SYSTEM_PROMPT)]),
    ]
    contents = [_edgellm.MessageContent("image", str(p)) for p in frame_paths]
    contents.append(_edgellm.MessageContent("text", prompt))
    messages.append(_edgellm.Message("user", contents))

    kwargs = _build_gen_kwargs([messages], max_tokens, TEMPERATURE, TOP_P, TOP_K)

    log.info("VLM 추론 시작 | 이미지=%d장 | max_tokens=%d", len(frame_paths), max_tokens)
    response = _runtime.handle_request(_edgellm.create_generation_request(**kwargs))
    raw = response.output_texts[0] if response.output_texts else ""
    log.info("VLM 추론 완료 | 출력길이=%d | 원문=%s", len(raw), raw[:200])
    return raw


def _sync_llm_infer(prompt: str, max_tokens: int) -> str:
    messages = [
        _edgellm.Message("system", [_edgellm.MessageContent("text", SYSTEM_PROMPT)]),
        _edgellm.Message("user", [_edgellm.MessageContent("text", prompt)]),
    ]
    kwargs = _build_gen_kwargs([messages], max_tokens, TEMPERATURE, TOP_P, TOP_K)

    log.info("LLM 추론 시작 | max_tokens=%d", max_tokens)
    response = _runtime.handle_request(_edgellm.create_generation_request(**kwargs))
    raw = response.output_texts[0] if response.output_texts else ""
    log.info("LLM 추론 완료 | 출력길이=%d | 원문=%s", len(raw), raw[:300])
    return raw


def _parse_llm_json(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {"action": "", "tts_message": ""}


# ── 엔드포인트 ───────────────────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    folder = Path(req.dir_path)
    if not folder.is_dir():
        raise HTTPException(
            status_code=400, detail=f"폴더가 존재하지 않습니다: {req.dir_path}"
        )

    all_frames = collect_frames(folder)
    if len(all_frames) < NUM_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"프레임이 {NUM_FRAMES}장 미만입니다 (발견: {len(all_frames)}장)",
        )

    frames = select_frames(all_frames, NUM_FRAMES)

    vlm_prompt = VLM_PROMPT_BASE
    if req.focus:
        vlm_prompt += VLM_FOCUS_PREFIX + req.focus

    actions = (
        [a.model_dump() for a in req.detect_actions]
        if req.detect_actions
        else DEFAULT_DETECT_ACTIONS
    )
    detect_items = "\n".join(f'- {a["key"]}: {a["label"]}' for a in actions)

    request_id = str(uuid.uuid4())[:8]
    log.info(
        "analyze 시작 | req=%s | 폴더=%s | 전체=%d → 선택=%d | focus=%s",
        request_id, req.dir_path, len(all_frames), len(frames),
        req.focus or "(없음)",
    )
    t0 = time.perf_counter()

    async with _lock:
        vlm_output = await asyncio.to_thread(
            _sync_vlm_infer, frames, vlm_prompt, VLM_MAX_TOKENS,
        )
        log.info("analyze VLM 완료 | req=%s | 설명길이=%d", request_id, len(vlm_output))

        llm_prompt = LLM_PROMPT_TEMPLATE.format(
            vlm_output=vlm_output, detect_items=detect_items,
        )
        llm_output = await asyncio.to_thread(
            _sync_llm_infer, llm_prompt, LLM_MAX_TOKENS,
        )
    log.info("analyze LLM 완료 | req=%s | 응답=%s", request_id, llm_output.strip())

    elapsed = time.perf_counter() - t0
    result = _parse_llm_json(llm_output)

    return AnalyzeResponse(
        request_id=request_id,
        action=result.get("action", ""),
        tts_message=result.get("tts_message", ""),
        vlm_description=vlm_output,
        elapsed_sec=round(elapsed, 3),
    )


@app.post("/debug/analyze_raw")
async def debug_analyze_raw(req: AnalyzeRequest):
    """추론 없이 런타임에 전달될 요청 구조만 반환. 직접 실행 JSON과 diff 비교용."""
    folder = Path(req.dir_path)
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"폴더 없음: {req.dir_path}")

    all_frames = collect_frames(folder)
    if len(all_frames) < NUM_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"프레임 {NUM_FRAMES}장 미만 (발견: {len(all_frames)}장)",
        )

    frames = select_frames(all_frames, NUM_FRAMES)

    vlm_prompt = VLM_PROMPT_BASE
    if req.focus:
        vlm_prompt += VLM_FOCUS_PREFIX + req.focus

    actions = (
        [a.model_dump() for a in req.detect_actions]
        if req.detect_actions
        else DEFAULT_DETECT_ACTIONS
    )
    detect_items = "\n".join(f'- {a["key"]}: {a["label"]}' for a in actions)
    llm_prompt_preview = LLM_PROMPT_TEMPLATE.format(
        vlm_output="<VLM 출력이 여기에 들어감>", detect_items=detect_items,
    )

    vlm_request = {
        "batch_size": 1,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_generate_length": VLM_MAX_TOKENS,
        "requests": [{
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    *[{"type": "image", "image": str(p)} for p in frames],
                    {"type": "text", "text": vlm_prompt},
                ]},
            ]
        }],
    }

    llm_request = {
        "batch_size": 1,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_generate_length": LLM_MAX_TOKENS,
        "requests": [{
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": llm_prompt_preview},
            ]
        }],
    }

    return {
        "frames_found": len(all_frames),
        "frames_selected": [str(f) for f in frames],
        "vlm_request": vlm_request,
        "llm_request": llm_request,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)