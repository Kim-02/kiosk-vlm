import asyncio
import importlib
import logging
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import config as cfg_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

NUM_FRAMES = 15
FRAME_SIZE = 448
FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_FRAME_RE = re.compile(r"^frame_(\d+)$|^frame(\d+)$")

# ── 샘플링 파라미터 ──────────────────────────────────────────────────────────
MAX_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9
TOP_K = 50

# ── 프롬프트 ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are a factory safety inspector analyzing CCTV footage. Only describe what you actually see. Answer in Korean."

PROMPT_GENERAL = (
    "이 연속 프레임은 작업 현장 CCTV 영상이다. 안전 관점에서 현재 상황을 자세히 분석하라.\n"
    "- 사람이 몇 명 보이는지\n"
    "- 각 사람이 무엇을 하고 있는지\n"
    "- 헬멧, 안전장비 착용 여부\n"
    "- 사다리, 위험 구역, 장비 접촉 등 위험 요소가 있는지\n"
    "특히 안전모 미착용, 스피커 접촉, 혼자 사다리 작업, 위험 테이프 넘기에 주의하라.\n"
    "보이는 것만 서술하라. 추측하지 마라."
)

PROMPT_HAT = (
    "이 연속 프레임은 작업 현장 CCTV 영상이다.\n"
    "프레임에 보이는 사람들의 머리 부분을 집중해서 관찰하라.\n"
    "안전모(헬멧)를 착용하고 있는 사람과 착용하지 않은 사람을 구분하라.\n"
    "안전모를 벗고 있는 중인 사람도 포함하라.\n"
    "보이는 것만 서술하라. 추측하지 마라."
)

PROMPT_SPEAKER = (
    "이 연속 프레임은 작업 현장 CCTV 영상이다.\n"
    "프레임에 스피커 또는 음향 장비가 보이는지 확인하라.\n"
    "사람이 스피커를 만지거나 손을 대고 있는지 관찰하라.\n"
    "보이는 것만 서술하라. 추측하지 마라."
)

PROMPT_LADDER = (
    "이 연속 프레임은 작업 현장 CCTV 영상이다.\n"
    "프레임에 사다리가 보이는지 확인하라.\n"
    "사다리 위에 올라간 사람이 있는지, 아래에서 잡아주는 사람이 있는지 관찰하라.\n"
    "혼자 사다리를 타고 있는 경우를 특히 주의하라.\n"
    "보이는 것만 서술하라. 추측하지 마라."
)

PROMPT_TAPE = (
    "이 연속 프레임은 작업 현장 CCTV 영상이다.\n"
    "바닥에 위험 구역을 표시하는 테이프, 라인, 펜스가 보이는지 확인하라.\n"
    "사람이 그 테이프나 라인을 넘어가고 있는지 관찰하라.\n"
    "보이는 것만 서술하라. 추측하지 마라."
)

PROMPT_FIST = (
    "이 연속 프레임은 작업 현장 CCTV 영상이다.\n"
    "프레임에 보이는 사람들의 손을 집중해서 관찰하라.\n"
    "주먹을 쥐고 있는 사람이 있는지 확인하라.\n"
    "보이는 것만 서술하라. 추측하지 마라."
)

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
class AnalyzeRequest(BaseModel):
    dir_path: str


class AnalyzeResponse(BaseModel):
    request_id: str
    description: str
    elapsed_sec: float


class FistResponse(BaseModel):
    request_id: str
    description: str
    action: str
    tts_message: str
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


def _resize_frame(src: Path) -> str:
    import cv2
    img = cv2.imread(str(src))
    if img is None:
        return str(src)
    h, w = img.shape[:2]
    if max(h, w) <= FRAME_SIZE:
        return str(src)
    scale = FRAME_SIZE / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    tmp_dir = Path("/tmp/vlm_resized")
    tmp_dir.mkdir(exist_ok=True)
    out = tmp_dir / f"{src.stem}.jpg"
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(out)


def _run_vlm(
    frame_paths: list[Path], prompt: str, max_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    resized = [_resize_frame(p) for p in frame_paths]

    image_buffers = []
    for rp in resized:
        image_buffers.append(_edgellm.load_image_from_path(rp))

    messages = [
        _edgellm.Message("system", [_edgellm.MessageContent("text", system_prompt)]),
    ]
    contents = [_edgellm.MessageContent("image", rp) for rp in resized]
    contents.append(_edgellm.MessageContent("text", prompt))
    messages.append(_edgellm.Message("user", contents))

    gen_req = _edgellm.create_generation_request(
        batch_messages=[messages],
        temperature=TEMPERATURE,
        max_generate_length=max_tokens,
        top_p=TOP_P,
        top_k=TOP_K,
        apply_chat_template=True,
        add_generation_prompt=True,
    )
    gen_req.requests[0].image_buffers = image_buffers

    log.info("추론 시작 | 이미지=%d장 | max_tokens=%d", len(frame_paths), max_tokens)
    response = _runtime.handle_request(gen_req)
    raw = response.output_texts[0] if response.output_texts else ""
    log.info("추론 완료 | 출력길이=%d | 원문=%s", len(raw), raw[:300])
    return raw


def _validate_and_select(dir_path: str) -> list[Path]:
    folder = Path(dir_path)
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"폴더가 존재하지 않습니다: {dir_path}")
    all_frames = collect_frames(folder)
    if len(all_frames) < NUM_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"프레임이 {NUM_FRAMES}장 미만입니다 (발견: {len(all_frames)}장)",
        )
    return select_frames(all_frames, NUM_FRAMES)


async def _analyze(req: AnalyzeRequest, prompt: str, tag: str) -> AnalyzeResponse:
    frames = _validate_and_select(req.dir_path)
    request_id = str(uuid.uuid4())[:8]

    log.info("%s 시작 | req=%s | 폴더=%s | 프레임=%d", tag, request_id, req.dir_path, len(frames))
    t0 = time.perf_counter()

    async with _lock:
        raw = await asyncio.to_thread(_run_vlm, frames, prompt, MAX_TOKENS)

    elapsed = time.perf_counter() - t0
    log.info("%s 완료 | req=%s | %.2fs | 응답=%s", tag, request_id, elapsed, raw.strip())

    return AnalyzeResponse(
        request_id=request_id,
        description=raw.strip(),
        elapsed_sec=round(elapsed, 3),
    )


# ── 엔드포인트 ───────────────────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """종합 안전 분석"""
    return await _analyze(req, PROMPT_GENERAL, "analyze")


@app.post("/analyze/hat", response_model=AnalyzeResponse)
async def analyze_hat(req: AnalyzeRequest):
    """안전모 미착용 탐지"""
    return await _analyze(req, PROMPT_HAT, "analyze/hat")


@app.post("/analyze/speaker", response_model=AnalyzeResponse)
async def analyze_speaker(req: AnalyzeRequest):
    """스피커 접촉 탐지"""
    return await _analyze(req, PROMPT_SPEAKER, "analyze/speaker")


@app.post("/analyze/ladder", response_model=AnalyzeResponse)
async def analyze_ladder(req: AnalyzeRequest):
    """혼자 사다리 작업 탐지"""
    return await _analyze(req, PROMPT_LADDER, "analyze/ladder")


@app.post("/analyze/tape", response_model=AnalyzeResponse)
async def analyze_tape(req: AnalyzeRequest):
    """위험 테이프 넘기 탐지"""
    return await _analyze(req, PROMPT_TAPE, "analyze/tape")


@app.post("/analyze/fist", response_model=FistResponse)
async def analyze_fist(req: AnalyzeRequest):
    """주먹 쥔 사람 탐지 → 감지 시 action: hat_action"""
    frames = _validate_and_select(req.dir_path)
    request_id = str(uuid.uuid4())[:8]

    log.info("analyze/fist 시작 | req=%s | 폴더=%s", request_id, req.dir_path)
    t0 = time.perf_counter()

    async with _lock:
        raw = await asyncio.to_thread(_run_vlm, frames, PROMPT_FIST, MAX_TOKENS)

    elapsed = time.perf_counter() - t0
    desc = raw.strip()
    log.info("analyze/fist 완료 | req=%s | %.2fs | 응답=%s", request_id, elapsed, desc[:200])

    detected = "주먹" in desc
    return FistResponse(
        request_id=request_id,
        description=desc,
        action="hat_action" if detected else "",
        tts_message="안전 이상이 발생되었습니다." if detected else "",
        elapsed_sec=round(elapsed, 3),
    )


# ── 디버그 ───────────────────────────────────────────────────────────────────
@app.post("/v1/debug")
async def v1_debug(req: AnalyzeRequest):
    """프롬프트 없이 장면 설명만 요청."""
    frames = _validate_and_select(req.dir_path)

    t0 = time.perf_counter()
    async with _lock:
        raw = await asyncio.to_thread(
            _run_vlm, frames, "Describe what you see.", 256,
            system_prompt="Describe the images. Answer in Korean.",
        )
    elapsed = time.perf_counter() - t0

    return {
        "description": raw,
        "frames_used": [str(f) for f in frames],
        "elapsed_sec": round(elapsed, 3),
    }


@app.get("/v1/debug/resize")
async def v1_debug_resize(dir_path: str):
    """브라우저에서 리사이즈된 15장 확인."""
    import base64

    frames = _validate_and_select(dir_path)
    resized = [_resize_frame(p) for p in frames]

    imgs_html = ""
    for after in resized:
        with open(after, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        imgs_html += (
            '<div style="display:inline-block;margin:4px;text-align:center;">'
            f'<img src="data:image/jpeg;base64,{b64}" style="width:200px;height:200px;object-fit:contain;border:1px solid #444;">'
            f'<div style="color:#aaa;font-size:11px;">{Path(after).name}</div>'
            '</div>\n'
        )

    return HTMLResponse(
        f'<!DOCTYPE html><html><head><meta charset="utf-8"></head>'
        f'<body style="background:#111;margin:20px;font-family:sans-serif;">'
        f'<h2 style="color:#fff;">리사이즈된 VLM 입력 프레임 ({len(resized)}장)</h2>'
        f'<p style="color:#888;">원본: {dir_path} | 타겟: {FRAME_SIZE}px</p>'
        f'<div style="display:flex;flex-wrap:wrap;">{imgs_html}</div>'
        f'</body></html>'
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
