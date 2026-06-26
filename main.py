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
# 현재는 '라바콘 접촉'만 탐지. (다른 라벨이 모델에서 나와도 후처리에서 걸러진다)
LABELS = ["cone_touch", "helmet_off", "fence_crossing", "ladder_alone", "safety_vest"]

LABEL_KO = {
    "helmet_off": "안전모 미착용",
    "cone_touch": "라바콘 접촉",
    "fence_crossing": "위험 펜스 넘음",
    "ladder_alone": "사다리 단독 이용",
    "safety_vest": "안전 고리 미착용",
}

# 행동별 제지 TTS 문구. 여러 개 감지되면 이어붙인다.
TTS_PHRASE = {
    "helmet_off": "안전모를 착용하세요.",
    "cone_touch": "라바콘에서 손을 떼세요.",
    "fence_crossing": "위험 펜스를 넘지 마세요.",
    "ladder_alone": "사다리를 혼자 사용하지 마세요. 보조자를 배치하세요.",
    "safety_vest": "안전 고리를 착용하세요.",
}

# 라벨별 단일 판정 기준. 한 요청당 라벨마다 개별 추론을 돌릴 때 사용한다.
LABEL_CRITERIA = {
    "helmet_off": "a worker whose head is clearly visible and is not wearing a red helmet",
    "cone_touch": "a worker whose hand or body is clearly touching an orange cone",
    "fence_crossing": "a person clearly standing on, stepping on, or crossing an expandable safety rail",
    "ladder_alone": "a worker clearly using or climbing a green ladder with no helper visible nearby",
    "safety_vest": "a red helmet and upper body are clearly visible, but the safety hook is not above/on the helmet",
}

# ── 시스템 프롬프트 ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a factory CCTV safety checker. "
    "Images are consecutive CCTV frames. Analyze them as one sequence. "
    "Detect all clearly visible safety violations. Multiple labels may appear at once. "
    "Check only these labels: "
    "helmet_off=worker head is clearly visible and no red helmet is worn; "
    "cone_touch=worker hand/body is clearly touching orange cone; "
    "fence_crossing=person is clearly standing on, stepping on, or crossing expandable safety rail; "
    "ladder_alone=worker is clearly using/climbing green ladder and no helper is visible nearby; "
    "safety_vest=red helmet and upper area are clearly visible, but safety hook is not above/on helmet. "
    "Report a label only when visual evidence is clear. "
    "Do not infer hidden objects, unseen people, colors, or actions. "
    "Do not report if blurry, occluded, cropped, or uncertain. "
    "If a label appears in any frame, report it once. "
    "Return JSON only, no markdown, no extra text. "
    '{"detections":[{"label":"","evidence":""}]} '
    "label must be one listed key. "
    "Each item must have only label and evidence. "
    "Evidence must be short and visual. "
    'If none or uncertain, return {"detections":[]}. '
    "Evidence max 8 words."
    "Do not guess."
)

# ── 판정 프롬프트 (라벨별 단일 판정의 유저 프롬프트) ─────────────────────────
DETECT_PROMPT = (
    "Check all frames and decide if the specified situation is present."
)

CHECK_PROMPT = (
    "화면에 보이는걸 모두 설명해라."
)

# ── 설명 전용(디버그) 시스템 프롬프트 ────────────────────────────────────────
# 안전판정/참조표와 무관하게, 보이는 장면을 있는 그대로만 묘사하도록 한다.
DESCRIBE_SYSTEM_PROMPT = (
    "당신은 영상 장면을 객관적으로 묘사하는 도우미입니다. "
    "이미지들은 시간 순서대로 이어진 연속 프레임입니다. "
    "위험/안전을 판정하지 말고, 화면에 실제로 보이는 것만 한국어로 설명하세요."
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


class VLMRequest(BaseModel):
    image_path: str
    prompt: str
    system_prompt: str | None = None


class Detection(BaseModel):
    label: str
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


def _resize_frames(frame_paths: list[Path], sub: str) -> list[str]:
    """프레임들을 한 번만 리사이즈해 경로 목록을 반환."""
    return [_resize_frame(p, sub) for p in frame_paths]


def _build_messages(image_paths: list[str], system_prompt: str, prompt: str) -> list:
    """system + (이미지들 + 텍스트) user 메시지 한 묶음을 구성."""
    messages = [
        _edgellm.Message("system", [_edgellm.MessageContent("text", system_prompt)]),
    ]
    contents = [_edgellm.MessageContent("image", rp) for rp in image_paths]
    contents.append(_edgellm.MessageContent("text", prompt))
    messages.append(_edgellm.Message("user", contents))
    return messages


def _run_vlm_batch(
    image_paths: list[str],
    items: list[tuple[str, str]],
) -> list[str]:
    """같은 이미지에 대해 여러 (system_prompt, prompt) 질의를 한 번의 배치로 추론.

    items[i] = (system_prompt, user_prompt). 반환은 items와 같은 순서의 raw 출력 목록.
    리사이즈는 호출자가 _resize_frames로 미리 수행한다.
    """
    cfg = cfg_module.get()
    batch_messages = [_build_messages(image_paths, sp, pr) for sp, pr in items]

    gen_req = _edgellm.create_generation_request(
        batch_messages=batch_messages,
        temperature=cfg.TEMPERATURE,
        max_generate_length=cfg.MAX_TOKENS,
        top_p=cfg.TOP_P,
        top_k=cfg.TOP_K,
        apply_chat_template=True,
        add_generation_prompt=True,
    )
    # 이미지는 한 번만 로드한다.
    image_buffers = [_edgellm.load_image_from_path(rp) for rp in image_paths]

    # 각 batch request에 동일한 이미지 버퍼 리스트를 연결한다.
    # 리스트 객체는 분리하고, buffer 객체는 재사용한다.
    for req in gen_req.requests:
        req.image_buffers = list(image_buffers)

    log.info(
        "배치 추론 시작 | 배치=%d | 이미지=%d장 | image_buffers=%d | max_tokens=%d",
        len(batch_messages), len(image_paths), len(image_buffers), cfg.MAX_TOKENS,
    )

    response = _runtime.handle_request(gen_req)
    outs = list(response.output_texts) if response.output_texts else []

    # 출력 개수는 항상 batch_messages 개수와 맞춘다.
    # 많으면 자르고, 부족하면 빈 문자열로 채운다.
    expected = len(batch_messages)
    if len(outs) > expected:
        log.warning("배치 출력 초과 | expected=%d | actual=%d | 초과분 제거", expected, len(outs))
        outs = outs[:expected]
    elif len(outs) < expected:
        log.warning("배치 출력 부족 | expected=%d | actual=%d | 빈 문자열 보정", expected, len(outs))
        outs += [""] * (expected - len(outs))

    log.info("배치 추론 완료 | 출력=%d개", len(outs))
    return outs


def _run_vlm(
    image_paths: list[str],
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """단일 질의 추론. _run_vlm_batch의 1개짜리 래퍼."""
    return _run_vlm_batch(image_paths, [(system_prompt, prompt)])[0]


# 최상위가 객체({...})든 배열([...])이든 잡아낸다.
_JSON_RE = re.compile(r"[\[{].*[\]}]", re.DOTALL)


def _single_label_system_prompt(label: str) -> str:
    """라벨 하나만 판정하도록 좁힌 시스템 프롬프트."""
    return (
        "You are a factory CCTV safety checker. "
        "Images are consecutive CCTV frames; analyze them as one sequence. "
        f"Decide ONLY whether this specific situation appears in any frame: {LABEL_CRITERIA[label]}. "
        "Answer present=true only when the visual evidence is clear. "
        "Do not infer hidden objects, unseen people, colors, or actions. "
        "If blurry, occluded, cropped, or uncertain, answer present=false. "
        "Return JSON only, no markdown, no extra text: "
        '{"present": false, "evidence": ""} '
        "Evidence must be short and visual, max 8 words. Do not guess."
    )


def _parse_single(raw: str) -> tuple[bool, str]:
    """단일 라벨 판정 응답에서 (present, evidence)를 추출.

    {"present": bool, "evidence": ""} 형식을 기대하되, 배열로 감싸 와도 첫 객체를 본다.
    파싱 실패 시 (False, "")를 반환한다.
    """
    text = re.sub(r"```(?:json)?", "", raw.strip()).strip()
    m = _JSON_RE.search(text)
    if not m:
        log.warning("JSON 파싱 실패, 원문=%s", raw[:200])
        return False, ""
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        log.warning("JSON 디코드 실패, 원문=%s", raw[:200])
        return False, ""

    if isinstance(obj, list):
        obj = next((d for d in obj if isinstance(d, dict)), {})
    if not isinstance(obj, dict):
        return False, ""

    present = bool(obj.get("present", obj.get("detected", False)))
    return present, str(obj.get("evidence", ""))


def _inspect_images(paths: list[str]) -> list[dict]:
    """모델에 실제 입력된 이미지들의 로드 가능 여부와 크기를 점검."""
    import cv2
    info: list[dict] = []
    for p in paths:
        img = cv2.imread(p)
        ok = img is not None
        info.append({
            "path": p,
            "exists": Path(p).is_file(),
            "loadable": ok,
            "width": int(img.shape[1]) if ok else None,
            "height": int(img.shape[0]) if ok else None,
        })
    return info


def _build_tts(labels: list[str]) -> str:
    """감지된 행동들의 제지 문구를 LABELS 순서대로 이어붙인다."""
    if not labels:
        return "안전 이상이 없습니다."
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
    """라벨별 판정을 한 번의 배치 추론으로 처리한 뒤, 탐지 결과를 합쳐 TTS를 만든다."""
    frames = _validate_and_select(req.dir_path)
    request_id = str(uuid.uuid4())[:8]

    log.info("analyze 시작 | req=%s | 폴더=%s | 프레임=%d | 라벨=%d종",
             request_id, req.dir_path, len(frames), len(LABELS))
    t0 = time.perf_counter()

    details: list[Detection] = []
    raw_by_label: dict[str, str] = {}

    # 라벨 5개를 한 번의 배치로 모델에 넣는다(리사이즈는 1회).
    async with _lock:
        image_paths = await asyncio.to_thread(_resize_frames, frames, request_id)
        items = [(_single_label_system_prompt(label), DETECT_PROMPT) for label in LABELS]
        raws = await asyncio.to_thread(_run_vlm_batch, image_paths, items)

    for label, raw in zip(LABELS, raws):
        raw_by_label[label] = raw.strip()
        present, evidence = _parse_single(raw)
        log.info("  라벨=%s | present=%s | evidence=%s", label, present, evidence[:80])
        if present:
            details.append(Detection(label=label, evidence=evidence))

    elapsed = time.perf_counter() - t0
    labels = [det.label for det in details]  # LABELS 순서로 누적됨
    tts = _build_tts(labels)

    log.info("analyze 완료 | req=%s | %.2fs | labels=%s", request_id, elapsed, labels)

    return DetectResponse(
        request_id=request_id,
        detected=bool(details),
        labels=labels,
        tts_message=tts,
        details=details,
        raw=json.dumps(raw_by_label, ensure_ascii=False),
        elapsed_sec=round(elapsed, 3),
    )


# ── 디버그 ───────────────────────────────────────────────────────────────────
@app.post("/v1/debug")
async def v1_debug(req: AnalyzeRequest):
    """안전판정 없이, 프레임에 보이는 장면을 그대로 설명만 한다."""
    frames = _validate_and_select(req.dir_path)
    request_id = str(uuid.uuid4())[:8]

    t0 = time.perf_counter()
    async with _lock:
        image_paths = await asyncio.to_thread(_resize_frames, frames, request_id)
        # 안전판정 시스템 프롬프트를 쓰지 않고, 중립적 설명만 수행
        raw = await asyncio.to_thread(
            _run_vlm, image_paths, CHECK_PROMPT, DESCRIBE_SYSTEM_PROMPT,
        )
    elapsed = time.perf_counter() - t0

    images = await asyncio.to_thread(_inspect_images, image_paths)

    return {
        "request_id": request_id,
        "description": raw.strip(),
        "frames_used": [str(f) for f in frames],
        # 모델에 실제로 입력된 프레임들(리사이즈 후 경로/크기/로드여부).
        "images_fed": images,
        "elapsed_sec": round(elapsed, 3),
    }


@app.post("/vlm")
async def vlm(req: VLMRequest):
    """사진 한 장 경로 + 프롬프트를 받아 VLM 응답을 그대로 반환한다.

    참조표/안전판정 로직을 거치지 않고, 입력 이미지와 프롬프트만으로 추론한다.
    system_prompt를 지정하지 않으면 중립적 설명 프롬프트를 사용한다.
    """
    path = Path(req.image_path)
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"이미지가 존재하지 않습니다: {req.image_path}")
    if path.suffix.lower() not in FRAME_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 이미지 형식입니다: {path.suffix} (지원: {sorted(FRAME_EXTS)})",
        )

    request_id = str(uuid.uuid4())[:8]
    system_prompt = req.system_prompt or DESCRIBE_SYSTEM_PROMPT

    log.info("vlm 시작 | req=%s | 이미지=%s", request_id, req.image_path)
    t0 = time.perf_counter()

    async with _lock:
        image_paths = await asyncio.to_thread(_resize_frames, [path], request_id)
        raw = await asyncio.to_thread(
            _run_vlm, image_paths, req.prompt, system_prompt,
        )
    elapsed = time.perf_counter() - t0

    log.info("vlm 완료 | req=%s | %.2fs", request_id, elapsed)

    return {
        "request_id": request_id,
        "response": raw.strip(),
        "image": str(path),
        "image_fed": image_paths[0] if image_paths else None,
        "elapsed_sec": round(elapsed, 3),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── 설정 API ─────────────────────────────────────────────────────────────────
@app.get("/config", response_model=cfg_module.AppConfig)
async def get_config():
    """현재 설정 조회."""
    return cfg_module.get()


@app.put("/config", response_model=cfg_module.AppConfig)
async def update_config(patch: dict[str, Any]):
    """설정 부분 업데이트 후 config.json에 저장.

    샘플링 파라미터·임계값·프레임 설정은 즉시 반영된다.
    경로(ENGINE_DIR/PLUGIN_PATH/EDGELLM_PYBIND_DIR)·HOST·PORT는
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