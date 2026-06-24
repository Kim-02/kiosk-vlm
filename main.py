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

# ── 프롬프트 / 샘플링 ────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are a factory safety inspector. Only report what you actually see. Answer in JSON."

MAX_TOKENS = 128
TEMPERATURE = 0.2
TOP_P = 0.9
TOP_K = 50

DETECT_PROMPT = "Describe what the people in these frames are doing. Answer in Korean."

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


class DebugRequest(BaseModel):
    dir_path: str


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


ACTION_KEYS = ["hat_action", "touch_action", "dangerInOut_action", "ladder_action"]

ACTION_TTS = {
    "hat_action": "헬멧을 착용하십시오",
    "touch_action": "스피커에서 손을 떼십시오",
    "dangerInOut_action": "위험 구역에서 벗어나십시오",
    "ladder_action": "사다리에서 내려오십시오. 혼자 사다리 작업은 금지입니다",
}

EMPTY_RESULT = {
    "person_count": 0,
    "hat_action": 0, "touch_action": 0, "dangerInOut_action": 0, "ladder_action": 0,
    "description": "",
    "tts_message": "",
}


def _parse_and_validate(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return EMPTY_RESULT.copy()

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return EMPTY_RESULT.copy()

    person_count = max(int(data.get("person_count", 0)), 0)

    counts = {}
    for k in ACTION_KEYS:
        v = data.get(k, 0)
        counts[k] = max(int(v), 0) if isinstance(v, (int, float, str)) and str(v).isdigit() else 0

    tts_parts = []
    for k in ACTION_KEYS:
        if counts[k] > 0:
            tts_parts.append(f"{ACTION_TTS[k]} ({counts[k]}명)")

    return {
        "person_count": person_count,
        **counts,
        "description": str(data.get("description", "")),
        "tts_message": ", ".join(tts_parts),
    }


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
    request_id = str(uuid.uuid4())[:8]

    log.info("analyze 시작 | req=%s | 폴더=%s | 프레임=%d", request_id, req.dir_path, len(frames))
    t0 = time.perf_counter()

    async with _lock:
        raw = await asyncio.to_thread(_run_vlm, frames, DETECT_PROMPT, 256)

    elapsed = time.perf_counter() - t0
    log.info("analyze 완료 | req=%s | %.2fs | 응답=%s", request_id, elapsed, raw.strip())

    return AnalyzeResponse(
        request_id=request_id,
        description=raw.strip(),
        elapsed_sec=round(elapsed, 3),
    )


@app.post("/v1/debug")
async def v1_debug(req: DebugRequest):
    """프롬프트 없이 장면 설명만 요청."""
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

    folder = Path(dir_path)
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"폴더 없음: {dir_path}")

    all_frames = collect_frames(folder)
    if len(all_frames) < NUM_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"프레임 {NUM_FRAMES}장 미만 (발견: {len(all_frames)}장)",
        )

    frames = select_frames(all_frames, NUM_FRAMES)
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

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VLM Resized Frames</title></head>
<body style="background:#111;margin:20px;font-family:sans-serif;">
<h2 style="color:#fff;">리사이즈된 VLM 입력 프레임 ({len(resized)}장)</h2>
<p style="color:#888;">원본: {dir_path} | 타겟: {FRAME_SIZE}px</p>
<div style="display:flex;flex-wrap:wrap;">{imgs_html}</div>
</body></html>"""

    return HTMLResponse(content=html)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
