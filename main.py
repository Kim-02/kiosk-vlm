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
LABELS = ["cone_touch", "helmet_off", "fence_crossing", "ladder_alone", "safety_vest"]

# 2단계 파이프라인 라벨.
# 1단계: 안전모/조끼 착용 여부(PPE)를 먼저 감시한다.
# 2단계: 1단계를 통과(PPE 정상 착용)했을 때만 행동 3종을 감시한다.
STAGE1_LABELS = ["helmet_off", "safety_vest"]
STAGE2_LABELS = ["cone_touch", "fence_crossing", "ladder_alone"]

# 행동별 제지 TTS 문구. 여러 개 감지되면 이어붙인다.
TTS_PHRASE = {
    "helmet_off": "안전모를 착용하세요.",
    "cone_touch": "라바콘에서 손을 떼세요.",
    "fence_crossing": "위험 지역에 들어가지 마세요.",
    "ladder_alone": "사다리를 혼자 사용하지 마세요. 보조자를 배치하세요.",
    "safety_vest": "안전 조끼를 착용하세요.",
}

# 라벨별 단일 판정 기준. 한 요청당 라벨마다 개별 추론을 돌릴 때 사용한다.
LABEL_CRITERIA = {
    "helmet_off": "Is the worker on the screen wearing a red hard hat? tell true or false",
    "cone_touch": "Is the worker touch an orange traffic cone? just tell true or false",
    "fence_crossing": "Is there someone to the left of the yellow fence? just tell true or false",
    "ladder_alone": "Is there a worker on the screen climbing a green ladder alone? just tell true or false",
    "safety_vest": "Is the worker on the screen wearing a safety vest? just tell true or false",
}

# 라벨별로 '위반(present)'을 의미하는 모델 응답.
# 질문이 긍정형(착용?/사용?)이면 false가 위반, 위험상황을 직접 묻는 질문이면 true가 위반.
VIOLATION_WHEN = {
    "helmet_off": False,      # "빨간 안전모 착용?" → 미착용(false)이 위반
    "cone_touch": True,       # "콘 접촉?" → 접촉(true)이 위반
    "fence_crossing": True,   # "펜스 넘음?" → 넘음(true)이 위반
    "ladder_alone": True,     # "사다리 단독 사용?" → 단독(true)이 위반
    "safety_vest": False,     # "안전 조끼 착용?" → 미착용(false)이 위반
}

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


class SafetyDebugRequest(BaseModel):
    # 이미지 파일 한 장 또는 프레임 폴더 경로 둘 다 허용한다.
    path: str
    prompt: str
    system_prompt: str | None = None


class DetectResponse(BaseModel):
    request_id: str
    detected: bool
    labels: list[str]
    tts_message: str
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
    # 배치 항목별 모델 원문을 그대로 찍는다(개별 테스트와 배치 결과 비교용).
    for i, out in enumerate(outs):
        log.info("배치 응답[%d/%d] | %r", i, len(outs), out.strip())
    return outs


def _run_vlm(
    image_paths: list[str],
    prompt: str,
    system_prompt: str,
) -> str:
    """단일 질의 추론. _run_vlm_batch의 1개짜리 래퍼."""
    return _run_vlm_batch(image_paths, [(system_prompt, prompt)])[0]


# 모델 응답에서 긍정/부정 토큰을 잡아낸다(대소문자 무시).
# 모델이 true/false 대신 yes/no로 답할 때가 있어 둘 다 인식한다.
_BOOL_RE = re.compile(r"\b(true|false|yes|no)\b", re.IGNORECASE)
# 긍정(True)으로 취급하는 토큰.
_TRUE_TOKENS = {"true", "yes"}


def _single_label_user_prompt(label: str) -> str:
    """라벨 하나만 판정하도록 좁힌 유저 프롬프트.

    VLM 챗 템플릿이 마지막 user 메시지를 핵심 지시로 보므로, 실제 판정 질문은
    system이 아니라 user 메시지로 보낸다(시스템 프롬프트는 DETECT_PROMPT 사용).
    """
    return (
        "Images are consecutive CCTV frames; analyze them as one sequence. "
        "If no person is visible, return [] immediately. "
        f"{LABEL_CRITERIA[label]}."
    )


def _parse_bool(raw: str) -> bool | None:
    """모델 응답 텍스트에서 첫 번째 긍정/부정 판정을 추출.

    true/yes는 True, false/no는 False로 본다.
    사람이 없으면 모델이 '[]'를 반환하므로 None(위반 아님)으로 처리한다.
    그 외에 판정 토큰을 찾지 못해도 None을 반환한다(판정 불가).
    """
    text = raw.strip()
    if "[]" in text:
        return None  # 사람 없음 → 판정 대상 없음
    m = _BOOL_RE.search(text)
    if not m:
        log.warning("true/false 파싱 실패, 원문=%s", raw[:200])
        return None
    return m.group(1).lower() in _TRUE_TOKENS


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


def _run_stage(
    image_paths: list[str],
    stage_labels: list[str],
) -> tuple[list[str], dict[str, str]]:
    """한 단계의 라벨들을 한 번의 배치로 추론하고 위반 라벨 목록을 만든다.

    반환: (위반 라벨 목록, 라벨별 true/false 원문 dict). 추론이 무거우므로
    호출자는 asyncio.to_thread로 감싼다.
    """
    # system=공통 지시(DETECT_PROMPT), user=라벨별 판정 질문.
    # 질문을 user 메시지에 둬야 모델이 제대로 판정한다(개별 debug와 동일 구조).
    items = [(DETECT_PROMPT, _single_label_user_prompt(label)) for label in stage_labels]
    raws = _run_vlm_batch(image_paths, items)

    labels: list[str] = []
    raw_by_label: dict[str, str] = {}
    for label, raw in zip(stage_labels, raws):
        raw_by_label[label] = raw.strip()
        answer = _parse_bool(raw)  # 모델이 답한 true/false (None=판정 불가)
        # 라벨마다 위반을 의미하는 답이 다르다(VIOLATION_WHEN).
        violated = answer is not None and answer == VIOLATION_WHEN[label]
        log.info("  라벨=%s | answer=%s | violated=%s | 원문=%r",
                 label, answer, violated, raw.strip())
        if violated:
            labels.append(label)
    return labels, raw_by_label


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
    """2단계 파이프라인으로 판정한다.

    1단계: 안전모/조끼(PPE) 미착용을 감시한다. 위반이 있으면 그것만 보고하고 멈춘다.
    2단계: 1단계를 통과하면(PPE 정상) 행동 3종을 감시한다.
    """
    frames = _validate_and_select(req.dir_path)
    request_id = str(uuid.uuid4())[:8]

    log.info("analyze 시작 | req=%s | 폴더=%s | 프레임=%d",
             request_id, req.dir_path, len(frames))
    t0 = time.perf_counter()

    # 리사이즈는 한 번만 하고 두 단계가 같은 이미지를 재사용한다.
    async with _lock:
        image_paths = await asyncio.to_thread(_resize_frames, frames, request_id)

        # 1단계: PPE(안전모/조끼) 미착용 감시
        log.info("  [1단계] PPE 감시 | req=%s | 라벨=%s", request_id, STAGE1_LABELS)
        labels, raw_by_label = await asyncio.to_thread(
            _run_stage, image_paths, STAGE1_LABELS,
        )
        stage = 1

        # 1단계 통과(PPE 위반 없음)일 때만 2단계 행동 감시로 넘어간다.
        if not labels:
            stage = 2
            log.info("  [2단계] 행동 감시 | req=%s | 라벨=%s", request_id, STAGE2_LABELS)
            labels, s2_raw = await asyncio.to_thread(
                _run_stage, image_paths, STAGE2_LABELS,
            )
            raw_by_label.update(s2_raw)

    elapsed = time.perf_counter() - t0
    tts = _build_tts(labels)

    log.info("analyze 완료 | req=%s | %.2fs | 단계=%d | labels=%s",
             request_id, elapsed, stage, labels)

    return DetectResponse(
        request_id=request_id,
        detected=bool(labels),
        labels=labels,
        tts_message=tts,
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


@app.post("/safety/debug")
async def safety_debug(req: SafetyDebugRequest):
    """이미지 한 장 또는 프레임 폴더 + 임의 프롬프트로 VLM을 직접 호출한다.

    2단계 안전판정 파이프라인을 거치지 않고 지정한 프롬프트로만 추론하므로,
    안전 프롬프트를 실제 입력(다중 프레임 포함)에 바로 테스트할 수 있다.
    경로가 폴더면 analyze와 동일하게 프레임을 선별해 연속 프레임으로 넣는다.
    system_prompt 미지정 시 중립적 설명 프롬프트를 사용한다.
    """
    p = Path(req.path)
    if p.is_dir():
        frames = _validate_and_select(req.path)
    elif p.is_file():
        if p.suffix.lower() not in FRAME_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 이미지 형식입니다: {p.suffix} (지원: {sorted(FRAME_EXTS)})",
            )
        frames = [p]
    else:
        raise HTTPException(status_code=400, detail=f"경로가 존재하지 않습니다: {req.path}")

    request_id = str(uuid.uuid4())[:8]
    system_prompt = req.system_prompt or DESCRIBE_SYSTEM_PROMPT

    log.info("safety/debug 시작 | req=%s | 경로=%s | 프레임=%d",
             request_id, req.path, len(frames))
    t0 = time.perf_counter()

    async with _lock:
        image_paths = await asyncio.to_thread(_resize_frames, frames, request_id)
        raw = await asyncio.to_thread(_run_vlm, image_paths, req.prompt, system_prompt)
    elapsed = time.perf_counter() - t0

    images = await asyncio.to_thread(_inspect_images, image_paths)

    log.info("safety/debug 완료 | req=%s | %.2fs", request_id, elapsed)

    return {
        "request_id": request_id,
        "response": raw.strip(),
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