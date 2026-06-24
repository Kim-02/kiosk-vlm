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
from fastapi.responses import FileResponse
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
SYSTEM_PROMPT = (
    "You are a vision-language model for CCTV safety monitoring. Answer in Korean."
)

MAX_TOKENS = 128
TEMPERATURE = 0.2
TOP_P = 0.9
TOP_K = 50

PROMPT_TEMPLATE = (
    "공장 작업 현장 CCTV 연속 프레임이다. "
    "아래 행동이 프레임에서 보이는지 판단하라. "
    "보이면 해당 키를 출력하고, 보이지 않으면 빈 문자열로 둬라.\n"
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
    prompt: str
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


def _resize_frames(frame_paths: list[Path]) -> list[str]:
    """448x448이 아닌 이미지를 리사이즈하여 임시 파일로 저장. 비율 유지 + 검정 패딩."""
    import cv2
    import numpy as np
    resized = []
    tmp_dir = Path("/tmp/vlm_resized")
    tmp_dir.mkdir(exist_ok=True)

    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            log.warning("이미지 읽기 실패: %s", p)
            resized.append(str(p))
            continue

        h, w = img.shape[:2]
        if h == FRAME_SIZE and w == FRAME_SIZE:
            resized.append(str(p))
            continue

        scale = min(FRAME_SIZE / w, FRAME_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        y_off = (FRAME_SIZE - new_h) // 2
        x_off = (FRAME_SIZE - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = img

        out = tmp_dir / f"{p.stem}.jpg"
        cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        resized.append(str(out))

    return resized


def _sync_infer(
    frame_paths: list[Path], prompt: str, max_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    image_paths = _resize_frames(frame_paths)

    messages = [
        _edgellm.Message("system", [_edgellm.MessageContent("text", system_prompt)]),
    ]
    contents = [_edgellm.MessageContent("image", p) for p in image_paths]
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
        prompt=prompt,
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


class DebugRequest(BaseModel):
    dir_path: str


@app.post("/v1/debug")
async def v1_debug(req: DebugRequest):
    """이미지 1장만, 리사이즈 없이, 원본 경로 그대로 pybind에 전달. 최소 테스트."""
    folder = Path(req.dir_path)
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"폴더 없음: {req.dir_path}")

    all_frames = collect_frames(folder)
    if not all_frames:
        raise HTTPException(status_code=400, detail="프레임 없음")

    test_image = str(all_frames[0])

    contents = [
        _edgellm.MessageContent("image", test_image),
        _edgellm.MessageContent("text", "Describe what you see."),
    ]

    request = _edgellm.create_generation_request(
        batch_messages=[[_edgellm.Message("user", contents)]],
        temperature=0.2,
        max_generate_length=256,
        top_p=0.9,
        top_k=50,
        apply_chat_template=True,
        add_generation_prompt=True,
    )

    t0 = time.perf_counter()
    async with _lock:
        response = await asyncio.to_thread(_runtime.handle_request, request)
    elapsed = time.perf_counter() - t0

    raw = response.output_texts[0] if response.output_texts else ""

    return {
        "description": raw,
        "test_image": test_image,
        "elapsed_sec": round(elapsed, 3),
    }


@app.get("/v1/debug/pybind")
async def v1_debug_pybind():
    """pybind 모듈의 클래스/메서드/속성 전체 목록을 반환."""
    import inspect
    result = {}

    for name in sorted(dir(_edgellm)):
        if name.startswith("_"):
            continue
        obj = getattr(_edgellm, name)
        info = {"type": type(obj).__name__}
        if inspect.isclass(obj):
            info["methods"] = [m for m in dir(obj) if not m.startswith("_")]
            try:
                sig = inspect.signature(obj.__init__)
                info["init_params"] = str(sig)
            except (ValueError, TypeError):
                info["init_params"] = "확인 불가"
        result[name] = info

    runtime_info = {}
    for name in sorted(dir(_runtime)):
        if name.startswith("_"):
            continue
        obj = getattr(_runtime, name)
        runtime_info[name] = type(obj).__name__

    mc_info = {}
    try:
        mc = _edgellm.MessageContent("text", "test")
        for name in sorted(dir(mc)):
            if not name.startswith("_"):
                mc_info[name] = type(getattr(mc, name)).__name__
    except Exception as e:
        mc_info["error"] = str(e)

    return {
        "module_members": result,
        "runtime_members": runtime_info,
        "message_content_members": mc_info,
    }


@app.get("/v1/debug/resize")
async def v1_debug_resize(dir_path: str):
    """GET으로 접근 — 브라우저에서 리사이즈된 15장을 한눈에 확인."""
    from fastapi.responses import HTMLResponse
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
    resized = _resize_frames(frames)

    imgs_html = ""
    for orig, after in zip(frames, resized):
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
<p style="color:#888;">원본: {dir_path} | 크기: {FRAME_SIZE}x{FRAME_SIZE}</p>
<div style="display:flex;flex-wrap:wrap;">{imgs_html}</div>
</body></html>"""

    return HTMLResponse(content=html)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
