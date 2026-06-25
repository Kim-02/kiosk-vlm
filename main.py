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

FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_FRAME_RE = re.compile(r"^frame_(\d+)$|^frame(\d+)$")

# ── 라벨 정의 (후처리용 고정 키) ─────────────────────────────────────────────
LABELS = ["helmet_off", "cone_touch", "fence_crossing", "ladder_alone"]

LABEL_KO = {
    "helmet_off": "안전모 미착용",
    "cone_touch": "라바콘 접촉",
    "fence_crossing": "위험 펜스 넘음",
    "ladder_alone": "사다리 단독 이용",
}

# 행동별 제지 TTS 문구. 여러 개 감지되면 이어붙인다.
TTS_PHRASE = {
    "helmet_off": "안전모를 착용하세요.",
    "cone_touch": "라바콘에서 손을 떼세요.",
    "fence_crossing": "위험 펜스를 넘지 마세요.",
    "ladder_alone": "사다리를 혼자 사용하지 마세요. 보조자를 배치하세요.",
}

# ── 시스템 프롬프트 ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "당신은 공장 안전 감시 시스템입니다. 첫 번째 이미지는 위험 행동 4종의 "
    "판정 기준을 보여주는 참조표입니다(DANGER=탐지대상, SAFE=정상, HARD CASE=헷갈리지만 정상). "
    "이후 이미지들은 실제 CCTV 연속 프레임입니다. "
    "참조표 기준에 따라, 연속 프레임에서 실제로 발생한 위험 행동만 판단하세요. "
    "추측하지 말고 프레임에서 명확히 보이는 근거로만 판단하며, 지정된 JSON 형식으로만 답하세요."
)

# ── 통합 판정 프롬프트 ───────────────────────────────────────────────────────
DETECT_PROMPT = (
    "첫 번째 이미지는 위험 행동 4종의 판정 기준 참조표다.\n"
    "왼쪽 열(DANGER)은 탐지해야 할 위험 상태, 가운데 열(SAFE)은 정상,\n"
    "오른쪽 열(HARD CASE)은 위험처럼 보이지만 정상으로 판단해야 하는 경우다.\n\n"
    "나머지 이미지는 실제 작업 현장 CCTV 연속 프레임이다.\n"
    "아래 4가지 위험 행동 각각에 대해, 연속 프레임에서 실제 발생했는지 단계적으로 판단하라.\n\n"
    "1) helmet_off (안전모 미착용): 머리에 단단한 반구형 안전모가 없거나 벗는 중.\n"
    "   - 안전모를 명확히 쓰고 있으면 정상. 모자/머리카락을 안전모로 착각하지 마라.\n"
    "2) cone_touch (라바콘 접촉): 손이 주황색 라바콘에 실제로 닿아 있거나 잡고 있음.\n"
    "   - 근처를 지나가거나 옆에 서 있을 뿐이면 정상(HARD CASE).\n"
    "3) fence_crossing (위험 펜스 넘음): 다리나 몸이 노랑/검정 펜스를 넘어가거나 넘는 중.\n"
    "   - 펜스 앞에 서 있거나 기대기만 했으면 정상(HARD CASE).\n"
    "4) ladder_alone (사다리 단독 이용): 사다리에 올라간 사람이 있고, 잡아주거나 옆에서 보조하는 사람이 없음.\n"
    "   - 사다리를 잡아주는 보조자가 있으면 정상(HARD CASE). 멀리 지나가는 사람은 보조자가 아니다.\n\n"
    "발생한 행동만 labels 배열에 넣어라. 아무것도 발생하지 않았으면 빈 배열로 둬라.\n"
    "[출력 형식] 아래 JSON 객체 하나만 출력하라. 마크다운/코드펜스/추가설명 금지.\n"
    '{"detections": ['
    '{"label": "helmet_off|cone_touch|fence_crossing|ladder_alone", '
    '"confidence": 0.0~1.0, "evidence": "관찰 근거 한 문장"}'
    ']}\n'
    "확실하지 않은 행동은 detections에 넣지 마라."
)


# ── 런타임 ────────────────────────────────────────────────────────────────────
_runtime: Any = None
_edgellm: Any = None
_lock = asyncio.Lock()
_reference_resized: str | None = None  # 참조 이미지는 한 번만 리사이즈해 캐시


def _load_edgellm_module(pybind_dir: str) -> Any:
    if pybind_dir not in sys.path:
        sys.path.insert(0, pybind_dir)
    return importlib.import_module("_edgellm_runtime")


# ── 앱 수명주기 ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI):
    global _runtime, _edgellm, _reference_resized

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

    # 참조 이미지 사전 리사이즈
    ref = Path(cfg.REFERENCE_IMAGE_PATH)
    if ref.is_file():
        _reference_resized = _resize_frame(ref, "reference")
        log.info("참조 이미지 로드 완료: %s", _reference_resized)
    else:
        log.warning("참조 이미지를 찾을 수 없습니다: %s (텍스트 기준만으로 동작)", cfg.REFERENCE_IMAGE_PATH)
        _reference_resized = None

    yield

    _runtime = None


app = FastAPI(title="DueGo VLM Server", lifespan=lifespan)


# ── 스키마 ────────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    dir_path: str


class Detection(BaseModel):
    label: str
    confidence: float
    evidence: str


class DetectResponse(BaseModel):
    request_id: str
    detected: bool
    labels: list[str]
    tts_message: str
    details: list[Detection]
    raw: str
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


def select_frames(frames: list[Path], target: int | None = None) -> list[Path]:
    if target is None:
        target = cfg_module.get().NUM_FRAMES
    n = len(frames)
    if n <= target:
        return frames
    indices = [round(i * (n - 1) / (target - 1)) for i in range(target)]
    return [frames[i] for i in indices]


def _resize_frame(src: Path, sub: str = "default") -> str:
    import cv2
    cfg = cfg_module.get()
    img = cv2.imread(str(src))
    if img is None:
        return str(src)
    h, w = img.shape[:2]
    if max(h, w) <= cfg.FRAME_SIZE:
        return str(src)
    scale = cfg.FRAME_SIZE / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    tmp_dir = Path(cfg.RESIZE_TMP_DIR) / sub
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / f"{src.stem}.jpg"
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, cfg.JPEG_QUALITY])
    return str(out)


def _run_vlm(frame_paths: list[Path], prompt: str, sub: str) -> str:
    """참조 이미지(있으면) + 판정 프레임을 함께 입력해 추론."""
    cfg = cfg_module.get()
    resized = [_resize_frame(p, sub) for p in frame_paths]

    # 참조 이미지를 맨 앞에 붙인다 (시스템/프롬프트에서 '첫 번째 이미지'로 지칭)
    all_image_paths: list[str] = []
    if _reference_resized:
        all_image_paths.append(_reference_resized)
    all_image_paths.extend(resized)

    image_buffers = [_edgellm.load_image_from_path(rp) for rp in all_image_paths]

    messages = [
        _edgellm.Message("system", [_edgellm.MessageContent("text", SYSTEM_PROMPT)]),
    ]
    contents = [_edgellm.MessageContent("image", rp) for rp in all_image_paths]
    contents.append(_edgellm.MessageContent("text", prompt))
    messages.append(_edgellm.Message("user", contents))

    gen_req = _edgellm.create_generation_request(
        batch_messages=[messages],
        temperature=cfg.TEMPERATURE,
        max_generate_length=cfg.MAX_TOKENS,
        top_p=cfg.TOP_P,
        top_k=cfg.TOP_K,
        apply_chat_template=True,
        add_generation_prompt=True,
    )
    gen_req.requests[0].image_buffers = image_buffers

    log.info("추론 시작 | 참조=%s | 프레임=%d장 | 총이미지=%d장 | max_tokens=%d",
             bool(_reference_resized), len(frame_paths), len(all_image_paths), cfg.MAX_TOKENS)
    response = _runtime.handle_request(gen_req)
    raw = response.output_texts[0] if response.output_texts else ""
    log.info("추론 완료 | 출력길이=%d | 원문=%s", len(raw), raw[:300])
    return raw


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_detections(raw: str) -> list[dict]:
    """VLM 출력에서 detections 배열을 안전하게 추출."""
    text = re.sub(r"```(?:json)?", "", raw.strip()).strip()
    m = _JSON_RE.search(text)
    if not m:
        log.warning("JSON 파싱 실패, 원문=%s", raw[:200])
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        log.warning("JSON 디코드 실패, 원문=%s", raw[:200])
        return []

    raw_dets = obj.get("detections", [])
    if not isinstance(raw_dets, list):
        return []

    out: list[dict] = []
    seen: set[str] = set()
    for d in raw_dets:
        if not isinstance(d, dict):
            continue
        label = str(d.get("label", "")).strip()
        if label not in LABELS or label in seen:
            continue
        try:
            conf = float(d.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        out.append({"label": label, "confidence": conf, "evidence": str(d.get("evidence", ""))})
        seen.add(label)
    return out


def _build_tts(labels: list[str]) -> str:
    """감지된 행동들의 제지 문구를 LABELS 순서대로 이어붙인다."""
    if not labels:
        return ""
    ordered = [lb for lb in LABELS if lb in labels]
    phrases = [TTS_PHRASE[lb] for lb in ordered]
    return "안전 이상이 발생했습니다. " + " ".join(phrases)


def _validate_and_select(dir_path: str) -> list[Path]:
    num_frames = cfg_module.get().NUM_FRAMES
    folder = Path(dir_path)
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"폴더가 존재하지 않습니다: {dir_path}")
    all_frames = collect_frames(folder)
    if len(all_frames) < num_frames:
        raise HTTPException(
            status_code=400,
            detail=f"프레임이 {num_frames}장 미만입니다 (발견: {len(all_frames)}장)",
        )
    return select_frames(all_frames, num_frames)


# ── 엔드포인트 ───────────────────────────────────────────────────────────────
@app.post("/analyze", response_model=DetectResponse)
async def analyze(req: AnalyzeRequest):
    """통합 안전 탐지: 4종 위험 행동을 한 번에 판정."""
    frames = _validate_and_select(req.dir_path)
    request_id = str(uuid.uuid4())[:8]

    log.info("analyze 시작 | req=%s | 폴더=%s | 프레임=%d", request_id, req.dir_path, len(frames))
    t0 = time.perf_counter()

    async with _lock:
        raw = await asyncio.to_thread(_run_vlm, frames, DETECT_PROMPT, request_id)

    elapsed = time.perf_counter() - t0
    dets = _parse_detections(raw)

    # 임계값 통과한 것만 최종 채택
    thresholds = cfg_module.get().CONFIDENCE_THRESHOLD
    kept = [d for d in dets if d["confidence"] >= thresholds.get(d["label"], 0.6)]
    labels = [lb for lb in LABELS if lb in {d["label"] for d in kept}]  # LABELS 순서 유지
    tts = _build_tts(labels)

    log.info("analyze 완료 | req=%s | %.2fs | labels=%s", request_id, elapsed, labels)

    return DetectResponse(
        request_id=request_id,
        detected=bool(labels),
        labels=labels,
        tts_message=tts,
        details=[Detection(**d) for d in kept],
        raw=raw.strip(),
        elapsed_sec=round(elapsed, 3),
    )


# ── 디버그 ───────────────────────────────────────────────────────────────────
@app.post("/v1/debug")
async def v1_debug(req: AnalyzeRequest):
    """판정 프롬프트 그대로, 파싱 전 원문 확인용."""
    frames = _validate_and_select(req.dir_path)
    request_id = str(uuid.uuid4())[:8]

    t0 = time.perf_counter()
    async with _lock:
        raw = await asyncio.to_thread(_run_vlm, frames, DETECT_PROMPT, request_id)
    elapsed = time.perf_counter() - t0

    return {
        "raw": raw,
        "parsed": _parse_detections(raw),
        "frames_used": [str(f) for f in frames],
        "reference_used": bool(_reference_resized),
        "elapsed_sec": round(elapsed, 3),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "reference_loaded": bool(_reference_resized)}


# ── 설정 API ─────────────────────────────────────────────────────────────────
@app.get("/config", response_model=cfg_module.AppConfig)
async def get_config():
    """현재 설정 조회."""
    return cfg_module.get()


@app.put("/config", response_model=cfg_module.AppConfig)
async def update_config(patch: dict[str, Any]):
    """설정 부분 업데이트 후 config.json에 저장.

    샘플링 파라미터·임계값·프레임 설정은 즉시 반영된다.
    경로(ENGINE_DIR/PLUGIN_PATH/EDGELLM_PYBIND_DIR/REFERENCE_IMAGE_PATH)·HOST·PORT는
    서버 재시작 후에 적용된다.
    """
    try:
        updated = cfg_module.update(patch)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # pydantic 검증 오류 등
        raise HTTPException(status_code=400, detail=f"설정 검증 실패: {e}")
    log.info("설정 업데이트 | 변경키=%s", sorted(patch))
    return updated


if __name__ == "__main__":
    cfg = cfg_module.init()
    uvicorn.run(app, host=cfg.HOST, port=cfg.PORT)