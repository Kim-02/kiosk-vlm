import asyncio
import importlib
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


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def collect_frames(folder: Path) -> list[Path]:
    """
    frame_01.jpg / frame1.jpg 형식을 모두 지원.
    frame 번호 순으로 정렬 후 반환. (최대 NUM_FRAMES장)
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


def _sync_infer(
    frame_paths: list[Path],
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """15프레임을 하나의 user message에 순서대로 넣고 추론. 블로킹 — to_thread로 호출."""
    contents = [_edgellm.MessageContent("image", str(p)) for p in frame_paths]
    contents.append(_edgellm.MessageContent("text", prompt))

    request = _edgellm.create_generation_request(
        batch_messages=[[_edgellm.Message("user", contents)]],
        temperature=temperature,
        max_generate_length=max_tokens,
        apply_chat_template=True,
        add_generation_prompt=True,
    )
    response = _runtime.handle_request(request)
    return response.output_texts[0] if response.output_texts else ""


async def run_inference(
    frame_paths: list[Path],
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    async with _gpu_sem:
        return await asyncio.to_thread(
            _sync_infer, frame_paths, prompt, max_tokens, temperature
        )


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


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
