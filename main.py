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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config as cfg_module
from config import ConfigUpdate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

NUM_FRAMES = 15
FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── analyze 엔드포인트용 정적 프롬프트 / 샘플링 파라미터 ────────────────────────
ANALYZE_SYSTEM_PROMPT = (
    "You are a vision-language model for CCTV safety monitoring. Answer in Korean."
)
ANALYZE_VLM_PROMPT = (
    "다음 15장의 CCTV 프레임은 시간 순서대로 촬영된 공장 작업 현장 연속 장면이다. "
    "각 프레임을 개별로 보지 말고 전체 흐름을 하나의 장면으로 분석하라. "
    "장면에 보이는 환경, 사람, 장비, 물체를 묘사하고 "
    "사람이 어떤 행동을 하고 있는지 구체적으로 서술하라."
)
ANALYZE_VLM_MAX_TOKENS = 256
ANALYZE_LLM_MAX_TOKENS = 200
ANALYZE_TEMPERATURE = 0.2
ANALYZE_TOP_P = 0.9
ANALYZE_TOP_K = 50

ANALYZE_LLM_PROMPT_TEMPLATE = (
    "현장 설명:\n{vlm_output}\n\n"
    "위 설명에서 아래 위험 행동을 감지하라.\n"
    "- hat_action: 안전모 미착용 또는 벗는 행동\n"
    "- touch_action: 스피커를 만지는 행동\n"
    "- dangerInOut_action: 금지 구역 출입\n"
    "- ladder_action: 사다리를 올라가거나 단독 사다리 작업\n\n"
    "JSON만 출력하라.\n"
    '{{"action":"감지키를_쉼표구분","tts_message":"경고문"}}\n'
    "action: 감지된 키만 나열. 없으면 빈 문자열.\n"
    "tts_message: 위험 사유와 함께 행동을 중단하십시오로 끝나는 문장. 없으면 빈 문자열."
)

DEBUG_DUMP_DIR = Path("/tmp")

# ── 런타임 (startup 때 초기화) ─────────────────────────────────────────────────
_runtime: Any = None
_edgellm: Any = None
_gpu_sem: asyncio.Semaphore = asyncio.Semaphore(1)


def _load_edgellm_module(pybind_dir: str) -> Any:
    if pybind_dir not in sys.path:
        sys.path.insert(0, pybind_dir)
    return importlib.import_module("_edgellm_runtime")


# ── 내부 상태 ─────────────────────────────────────────────────────────────────
@dataclass
class SlotState:
    slot_id: int
    request_id: Optional[str] = None
    folder_path: Optional[str] = None
    started_at: Optional[float] = None

    def is_idle(self) -> bool:
        return self.request_id is None

    def occupy(self, request_id: str, folder_path: str) -> None:
        self.request_id = request_id
        self.folder_path = folder_path
        self.started_at = time.time()

    def release(self) -> None:
        self.request_id = None
        self.folder_path = None
        self.started_at = None


@dataclass
class QueueItem:
    request_id: str
    folder_path: str
    prompt: str
    max_tokens: int
    queued_at: float = field(default_factory=time.time)
    future: asyncio.Future = field(
        default_factory=lambda: asyncio.get_event_loop().create_future()
    )


_slots: list[SlotState] = []
_queue: asyncio.Queue[QueueItem] = asyncio.Queue()
_queue_snapshot: list[QueueItem] = []

# frame_01 / frame_02 ... 또는 frame1 / frame2 ... 패턴
_FRAME_RE = re.compile(r'^frame_(\d+)$|^frame(\d+)$')


# ── 앱 수명주기 ───────────────────────────────────────────────────────────────
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

    for i in range(cfg.NUM_WORKERS):
        slot = SlotState(slot_id=i)
        _slots.append(slot)
        asyncio.create_task(worker_loop(slot), name=f"worker-slot{i}")
    log.info("워커 %d개 기동 완료", cfg.NUM_WORKERS)

    yield

    _runtime = None


app = FastAPI(title="DueGo VLM Server", lifespan=lifespan)


# ── 스키마 ────────────────────────────────────────────────────────────────────
class InferRequest(BaseModel):
    folder_path: str
    prompt: Optional[str] = None
    max_tokens: Optional[int] = None


class FrameResult(BaseModel):
    frame: str
    output_text: str


class InferResponse(BaseModel):
    request_id: str
    folder_path: str
    results: list[FrameResult]
    elapsed_sec: float


class SlotStatus(BaseModel):
    slot_id: int
    status: str
    request_id: Optional[str]
    folder_path: Optional[str]
    running_sec: Optional[float]


class QueueStatus(BaseModel):
    request_id: str
    folder_path: str
    queued_sec: float


class StatusResponse(BaseModel):
    slots: list[SlotStatus]
    queue: list[QueueStatus]
    queue_length: int


class ConfigUpdateResponse(BaseModel):
    config: dict
    workers_added: int
    restart_required: bool
    message: str


class AnalyzeRequest(BaseModel):
    dir_path: str


class AnalyzeResponse(BaseModel):
    request_id: str
    action: str
    tts_message: str
    vlm_description: str
    elapsed_sec: float


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def collect_frames(folder: Path) -> list[Path]:
    """
    frame_01.jpg / frame1.jpg 형식을 모두 지원.
    frame 번호 순으로 정렬 후 반환.
    """
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
    """30장 등에서 target(15)장을 균등 간격으로 선택."""
    n = len(frames)
    if n <= target:
        return frames
    indices = [round(i * (n - 1) / (target - 1)) for i in range(target)]
    return [frames[i] for i in indices]


def _build_generation_kwargs(
    batch_messages: list,
    max_tokens: int,
    temperature: float,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> dict:
    kwargs = dict(
        batch_messages=batch_messages,
        temperature=temperature,
        max_generate_length=max_tokens,
        apply_chat_template=True,
        add_generation_prompt=True,
    )
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k
    return kwargs


def _dump_debug_json(
    tag: str,
    frame_paths: list[Path],
    prompt: str,
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
) -> None:
    """직접 실행 JSON과 비교할 수 있도록 /tmp 에 요청 구조를 덤프."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    content = [{"type": "image", "image": str(p)} for p in frame_paths]
    content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})

    dump = {
        "batch_size": 1,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_generate_length": max_tokens,
        "requests": [{"messages": messages}],
    }
    out = DEBUG_DUMP_DIR / f"api_last_{tag}.json"
    out.write_text(json.dumps(dump, indent=2, ensure_ascii=False))
    log.info("디버그 JSON 저장: %s", out)


def _sync_infer(
    frame_paths: list[Path],
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: Optional[str] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> str:
    """15프레임을 하나의 user message에 순서대로 넣고 추론. 블로킹 — to_thread로 호출."""
    _dump_debug_json(
        "vlm", frame_paths, prompt, system_prompt,
        max_tokens, temperature, top_p, top_k,
    )

    messages: list = []
    if system_prompt:
        messages.append(
            _edgellm.Message("system", [_edgellm.MessageContent("text", system_prompt)])
        )

    contents = [_edgellm.MessageContent("image", str(p)) for p in frame_paths]
    contents.append(_edgellm.MessageContent("text", prompt))
    messages.append(_edgellm.Message("user", contents))

    kwargs = _build_generation_kwargs(
        [messages], max_tokens, temperature, top_p, top_k,
    )

    log.info(
        "VLM 추론 시작 | 이미지=%d장 | system=%s | temp=%.1f | top_p=%s | top_k=%s | max_tokens=%d",
        len(frame_paths), bool(system_prompt), temperature,
        top_p, top_k, max_tokens,
    )
    response = _runtime.handle_request(_edgellm.create_generation_request(**kwargs))
    raw = response.output_texts[0] if response.output_texts else ""
    log.info("VLM 추론 완료 | 출력길이=%d | 원문=%s", len(raw), raw[:200])
    return raw


async def run_inference(
    frame_paths: list[Path],
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: Optional[str] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> str:
    async with _gpu_sem:
        return await asyncio.to_thread(
            _sync_infer, frame_paths, prompt, max_tokens, temperature,
            system_prompt, top_p, top_k,
        )


def _sync_llm_infer(
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: Optional[str] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> str:
    """텍스트만 입력하여 추론. 블로킹 — to_thread로 호출."""
    messages: list = []
    if system_prompt:
        messages.append(
            _edgellm.Message("system", [_edgellm.MessageContent("text", system_prompt)])
        )

    contents = [_edgellm.MessageContent("text", prompt)]
    messages.append(_edgellm.Message("user", contents))

    kwargs = _build_generation_kwargs(
        [messages], max_tokens, temperature, top_p, top_k,
    )

    log.info("LLM 추론 시작 | system=%s | temp=%.1f | max_tokens=%d",
             bool(system_prompt), temperature, max_tokens)
    response = _runtime.handle_request(_edgellm.create_generation_request(**kwargs))
    raw = response.output_texts[0] if response.output_texts else ""
    log.info("LLM 추론 완료 | 출력길이=%d | 원문=%s", len(raw), raw[:300])
    return raw


async def run_llm_inference(
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: Optional[str] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> str:
    async with _gpu_sem:
        return await asyncio.to_thread(
            _sync_llm_infer, prompt, max_tokens, temperature,
            system_prompt, top_p, top_k,
        )


def _parse_llm_json(text: str) -> dict:
    """LLM 출력에서 JSON을 추출. 실패 시 빈 결과 반환."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {"action": "", "tts_message": ""}


async def process_item(slot: SlotState, item: QueueItem) -> None:
    slot.occupy(item.request_id, item.folder_path)
    log.info(
        "슬롯%d 시작 | req=%s | 폴더=%s",
        slot.slot_id, item.request_id, item.folder_path,
    )

    try:
        frame_paths = collect_frames(Path(item.folder_path))
        cfg = cfg_module.get()

        t0 = time.perf_counter()
        output_text = await run_inference(
            frame_paths[:NUM_FRAMES],
            item.prompt,
            item.max_tokens,
            cfg.TEMPERATURE,
        )
        elapsed = time.perf_counter() - t0

        response = InferResponse(
            request_id=item.request_id,
            folder_path=item.folder_path,
            results=[FrameResult(frame="sequence", output_text=output_text)],
            elapsed_sec=round(elapsed, 3),
        )
        log.info("슬롯%d 완료 | req=%s | %.2fs", slot.slot_id, item.request_id, elapsed)
        item.future.set_result(response)

    except Exception as e:
        log.error("슬롯%d 오류 | req=%s | %s", slot.slot_id, item.request_id, e)
        item.future.set_exception(e)
    finally:
        slot.release()


async def worker_loop(slot: SlotState) -> None:
    log.info("워커 슬롯%d 시작", slot.slot_id)
    while True:
        item = await _queue.get()
        _queue_snapshot.remove(item)
        try:
            await process_item(slot, item)
        finally:
            _queue.task_done()


def _add_workers(from_id: int, to_id: int) -> int:
    added = 0
    for i in range(from_id, to_id):
        slot = SlotState(slot_id=i)
        _slots.append(slot)
        asyncio.create_task(worker_loop(slot), name=f"worker-slot{i}")
        log.info("워커 슬롯%d 추가", i)
        added += 1
    return added


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    folder = Path(req.folder_path)
    if not folder.is_dir():
        raise HTTPException(
            status_code=400, detail=f"폴더가 존재하지 않습니다: {req.folder_path}"
        )

    frame_paths = collect_frames(folder)
    found = len(frame_paths)
    if found < NUM_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"프레임이 {NUM_FRAMES}장 미만입니다 "
                f"(발견: {found}장, 필요: {NUM_FRAMES}장): {req.folder_path}"
            ),
        )

    cfg = cfg_module.get()
    prompt = req.prompt if req.prompt is not None else cfg.DEFAULT_PROMPT
    max_tokens = req.max_tokens if req.max_tokens is not None else cfg.MAX_TOKENS

    request_id = str(uuid.uuid4())[:8]
    item = QueueItem(
        request_id=request_id,
        folder_path=req.folder_path,
        prompt=prompt,
        max_tokens=max_tokens,
        future=asyncio.get_event_loop().create_future(),
    )

    _queue_snapshot.append(item)
    await _queue.put(item)
    log.info(
        "요청 큐 등록 | req=%s | 폴더=%s | 프레임=%d | 대기=%d",
        request_id, req.folder_path, min(found, NUM_FRAMES), len(_queue_snapshot),
    )

    try:
        result = await item.future
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    dir_path 하나만 받아 VLM→LLM 2단계 분석 수행.
    30프레임 중 15장 선택 → VLM 장면 설명 → LLM 위험 행동 감지 → JSON 반환.
    """
    folder = Path(req.dir_path)
    if not folder.is_dir():
        raise HTTPException(
            status_code=400, detail=f"폴더가 존재하지 않습니다: {req.dir_path}"
        )

    all_frames = collect_frames(folder)
    if len(all_frames) < NUM_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"프레임이 {NUM_FRAMES}장 미만입니다 "
                f"(발견: {len(all_frames)}장): {req.dir_path}"
            ),
        )

    frames = select_frames(all_frames, NUM_FRAMES)
    request_id = str(uuid.uuid4())[:8]
    cfg = cfg_module.get()

    for fp in frames:
        if not fp.exists():
            log.error("프레임 파일 없음: %s", fp)
            raise HTTPException(status_code=400, detail=f"프레임 파일 없음: {fp}")

    log.info(
        "analyze 시작 | req=%s | 폴더=%s | 전체=%d → 선택=%d",
        request_id, req.dir_path, len(all_frames), len(frames),
    )
    t0 = time.perf_counter()

    vlm_output = await run_inference(
        frames, ANALYZE_VLM_PROMPT, ANALYZE_VLM_MAX_TOKENS,
        ANALYZE_TEMPERATURE,
        system_prompt=ANALYZE_SYSTEM_PROMPT,
        top_p=ANALYZE_TOP_P,
        top_k=ANALYZE_TOP_K,
    )
    log.info("analyze VLM 완료 | req=%s | 설명길이=%d", request_id, len(vlm_output))

    llm_prompt = ANALYZE_LLM_PROMPT_TEMPLATE.format(vlm_output=vlm_output)
    llm_output = await run_llm_inference(
        llm_prompt, ANALYZE_LLM_MAX_TOKENS,
        ANALYZE_TEMPERATURE,
        system_prompt=ANALYZE_SYSTEM_PROMPT,
        top_p=ANALYZE_TOP_P,
        top_k=ANALYZE_TOP_K,
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


@app.get("/status", response_model=StatusResponse)
async def status():
    now = time.time()
    return StatusResponse(
        slots=[
            SlotStatus(
                slot_id=s.slot_id,
                status="idle" if s.is_idle() else "running",
                request_id=s.request_id,
                folder_path=s.folder_path,
                running_sec=round(now - s.started_at, 1) if s.started_at else None,
            )
            for s in _slots
        ],
        queue=[
            QueueStatus(
                request_id=q.request_id,
                folder_path=q.folder_path,
                queued_sec=round(now - q.queued_at, 1),
            )
            for q in _queue_snapshot
        ],
        queue_length=len(_queue_snapshot),
    )


@app.get("/config/status")
async def config_status():
    return cfg_module.get().model_dump()


@app.patch("/config", response_model=ConfigUpdateResponse)
async def config_update(update: ConfigUpdate):
    changes = update.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(status_code=400, detail="변경할 항목이 없습니다.")

    old_workers = cfg_module.get().NUM_WORKERS
    new_cfg = cfg_module.apply(changes)
    new_workers = new_cfg.NUM_WORKERS

    workers_added = 0
    restart_required = False
    messages = []

    if "NUM_WORKERS" in changes:
        if new_workers > old_workers:
            workers_added = _add_workers(old_workers, new_workers)
            messages.append(
                f"워커 {workers_added}개 추가됨 (슬롯 {old_workers}~{new_workers - 1})"
            )
        elif new_workers < old_workers:
            restart_required = True
            messages.append(
                f"NUM_WORKERS 감소({old_workers}→{new_workers})는 재시작 후 적용됩니다."
            )

    if "ENGINE_DIR" in changes or "PLUGIN_PATH" in changes:
        restart_required = True
        messages.append("ENGINE_DIR / PLUGIN_PATH 변경은 재시작 후 적용됩니다.")

    if not messages:
        messages.append("설정이 즉시 적용됐습니다.")

    log.info("설정 변경: %s", changes)
    return ConfigUpdateResponse(
        config=new_cfg.model_dump(),
        workers_added=workers_added,
        restart_required=restart_required,
        message=" | ".join(messages),
    )


@app.post("/debug/analyze_raw")
async def debug_analyze_raw(req: AnalyzeRequest):
    """VLM raw output만 반환. 직접 실행 결과와 비교용."""
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

    vlm_output = await run_inference(
        frames, ANALYZE_VLM_PROMPT, ANALYZE_VLM_MAX_TOKENS,
        ANALYZE_TEMPERATURE,
        system_prompt=ANALYZE_SYSTEM_PROMPT,
        top_p=ANALYZE_TOP_P,
        top_k=ANALYZE_TOP_K,
    )
    elapsed = time.perf_counter() - t0

    return {
        "vlm_raw_output": vlm_output,
        "output_length": len(vlm_output),
        "frames_used": [str(f) for f in frames],
        "debug_json_path": str(DEBUG_DUMP_DIR / "api_last_vlm.json"),
        "elapsed_sec": round(elapsed, 3),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
