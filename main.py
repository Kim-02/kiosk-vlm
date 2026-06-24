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

MAX_TOKENS = 128
TEMPERATURE = 0.2
TOP_P = 0.9
TOP_K = 50

PROMPT_TEMPLATE = (
    "공장 CCTV 프레임이다."
    "아래 행동이 확실히 보일 때만 해당 키를 출력하라. 안 보이면 빈 문자열로 둬라.\n"
    "{detect_items}\n"
    "JSON만 출력.\n"
    '{{"action":"","tts_message":""}}\n'
    "{example_single}\n"
    "{example_multi}"
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
class DetectAction(BaseModel):
    key: str
    label: str


class AnalyzeRequest(BaseModel):
    dir_path: str
    focus: Optional[str] = None
    detect_actions: list[DetectAction]


class AnalyzeResponse(BaseModel):
    request_id: str
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


def _sync_infer(frame_paths: list[Path], prompt: str, max_tokens: int) -> str:
    messages = [
        _edgellm.Message("system", [_edgellm.MessageContent("text", SYSTEM_PROMPT)]),
    ]
    contents = [_edgellm.MessageContent("image", str(p)) for p in frame_paths]
    contents.append(_edgellm.MessageContent("text", prompt))
    messages.append(_edgellm.Message("user", contents))

    kwargs = dict(
        batch_messages=[messages],
        temperature=TEMPERATURE,
        max_generate_length=max_tokens,
        top_p=TOP_P,
        top_k=TOP_K,
        apply_chat_template=True,
        add_generation_prompt=True,
    )

    log.info("추론 시작 | 이미지=%d장 | max_tokens=%d", len(frame_paths), max_tokens)
    response = _runtime.handle_request(_edgellm.create_generation_request(**kwargs))
    raw = response.output_texts[0] if response.output_texts else ""
    log.info("추론 완료 | 출력길이=%d | 원문=%s", len(raw), raw[:300])
    return raw


def _parse_json(text: str) -> dict:
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

    actions = [a.model_dump() for a in req.detect_actions]
    detect_items = "\n".join(f'- {a["key"]}: {a["label"]}' for a in actions)

    first = actions[0]
    example_single = json.dumps(
        {"action": first["key"], "tts_message": first["label"] + "이 감지되었습니다. 즉시 중단하십시오"},
        ensure_ascii=False,
    )
    if len(actions) >= 2:
        second = actions[1]
        example_multi = json.dumps(
            {
                "action": f'{first["key"]},{second["key"]}',
                "tts_message": f'{first["label"]}이 감지되고, {second["label"]}이 감지되었습니다. 즉시 중단하십시오',
            },
            ensure_ascii=False,
        )
    else:
        example_multi = example_single

    prompt = PROMPT_TEMPLATE.format(
        detect_items=detect_items,
        example_single=example_single,
        example_multi=example_multi,
    )
    if req.focus:
        prompt += "\n관찰 영역: " + req.focus

    request_id = str(uuid.uuid4())[:8]
    log.info(
        "analyze 시작 | req=%s | 폴더=%s | 프레임=%d | focus=%s",
        request_id, req.dir_path, len(frames), req.focus or "(없음)",
    )
    t0 = time.perf_counter()

    async with _lock:
        raw = await asyncio.to_thread(
            _sync_infer, frames, prompt, MAX_TOKENS,
        )

    elapsed = time.perf_counter() - t0
    log.info("analyze 완료 | req=%s | %.2fs | 응답=%s", request_id, elapsed, raw.strip())

    result = _parse_json(raw)
    return AnalyzeResponse(
        request_id=request_id,
        action=result.get("action", ""),
        tts_message=result.get("tts_message", ""),
        elapsed_sec=round(elapsed, 3),
    )


@app.post("/debug/analyze_raw")
async def debug_analyze_raw(req: AnalyzeRequest):
    """추론 없이 런타임에 전달될 요청 구조만 반환."""
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

    actions = [a.model_dump() for a in req.detect_actions]
    detect_items = "\n".join(f'- {a["key"]}: {a["label"]}' for a in actions)

    first = actions[0]
    example_single = json.dumps(
        {"action": first["key"], "tts_message": first["label"] + "이 감지되었습니다. 즉시 중단하십시오"},
        ensure_ascii=False,
    )
    if len(actions) >= 2:
        second = actions[1]
        example_multi = json.dumps(
            {
                "action": f'{first["key"]},{second["key"]}',
                "tts_message": f'{first["label"]}이 감지되고, {second["label"]}이 감지되었습니다. 즉시 중단하십시오',
            },
            ensure_ascii=False,
        )
    else:
        example_multi = example_single

    prompt = PROMPT_TEMPLATE.format(
        detect_items=detect_items,
        example_single=example_single,
        example_multi=example_multi,
    )
    if req.focus:
        prompt += "\n관찰 영역: " + req.focus

    return {
        "frames_found": len(all_frames),
        "frames_selected": [str(f) for f in frames],
        "request": {
            "batch_size": 1,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_generate_length": MAX_TOKENS,
            "requests": [{
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        *[{"type": "image", "image": str(p)} for p in frames],
                        {"type": "text", "text": prompt},
                    ]},
                ]
            }],
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
