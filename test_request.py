"""
영상을 창에 재생하면서 1초(약 30프레임) 단위로 /analyze 를 호출하고,
분석 결과를 영상 위에 실시간으로 덧그린다. (실시간 CCTV 시뮬레이션)

흐름:
  1) 영상 전체를 30fps 로 staging 폴더에 추출 (ffmpeg 우선)
  2) 추출된 프레임을 창에 30fps 로 재생
  3) 1초치(30장)가 모이면 백그라운드 스레드에서 FRAMES_DIR 로 복사 → /analyze
  4) 응답이 오면 영상 위 오버레이를 갱신, 영상은 계속 재생 (분석 중에는 다음 분석을 보류)
  q 또는 ESC 로 종료. 영상은 끝나면 처음부터 다시 반복.

실행:
  python3 test_request.py
  python3 test_request.py /path/to/video.mp4 /path/to/frames
"""
import json
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

BASE_URL = "http://localhost:8000"

VIDEO_PATH = "/media/ds/DATA/videos/VLM_1.mp4"
FRAMES_DIR = "/home/ds/Desktop/kiosk-vlm/frames"
TARGET_FPS = 30.0
BATCH_SECONDS = 1.0   # 한 번에 분석할 영상 길이(초)
MIN_FRAMES = 15       # 서버 config.NUM_FRAMES 와 동일해야 함 (미만이면 400)
WINDOW = "VLM live"

LABEL_KO = {
    "helmet_off": "안전모 미착용",
    "cone_touch": "라바콘 접촉",
    "fence_crossing": "위험 펜스 넘음",
    "ladder_alone": "사다리 단독 이용",
}

_KR_FONT_PATHS = [
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]

# 분석 결과 공유 상태 (워커 스레드 → 메인 스레드)
_latest: dict = {"result": None}
_lock = threading.Lock()


# ── 프레임 추출 ──────────────────────────────────────────────────────────────
def _clear_frames(d: Path) -> Path:
    if d.exists():
        for old in d.glob("frame_*.jpg"):
            old.unlink()
    else:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_with_ffmpeg(video_path: str, out_dir: Path, fps: float) -> None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={fps:g}",
        "-q:v", "2",
        "-start_number", "1",
        str(out_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)


def _extract_with_cv2(video_path: str, out_dir: Path, fps: float) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(native_fps / fps, 1.0)
    saved = 0
    next_capture = 0.0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= next_capture:
            saved += 1
            cv2.imwrite(str(out_dir / f"frame_{saved:06d}.jpg"), frame)
            next_capture += step
        idx += 1
    cap.release()


def extract_all_frames(video_path: str, staging_dir: Path, fps: float) -> list[Path]:
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"영상을 찾을 수 없습니다: {video_path}")
    _clear_frames(staging_dir)
    if shutil.which("ffmpeg"):
        print("ffmpeg 로 전체 프레임 추출 중...")
        _extract_with_ffmpeg(video_path, staging_dir, fps)
    else:
        print("ffmpeg 없음 → OpenCV 로 전체 프레임 추출 중...")
        _extract_with_cv2(video_path, staging_dir, fps)
    frames = sorted(staging_dir.glob("frame_*.jpg"))
    print(f"전체 {len(frames)}장 추출 (~{len(frames) / fps:.1f}초 @ {fps:g}fps)")
    return frames


# ── 분석 (백그라운드) ────────────────────────────────────────────────────────
def call_analyze(dir_path: str) -> dict | None:
    payload = json.dumps({"dir_path": dir_path}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/analyze",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code, e.read().decode())
    except urllib.error.URLError as e:
        print("연결 실패:", e.reason)
    return None


def analyze_worker(batch_paths: list[Path], frames_dir: Path) -> None:
    """1초치 프레임을 FRAMES_DIR 로 복사한 뒤 /analyze 를 호출한다."""
    _clear_frames(frames_dir)
    for i, src in enumerate(batch_paths, 1):
        shutil.copyfile(src, frames_dir / f"frame_{i:04d}.jpg")

    data = call_analyze(str(frames_dir))
    if data:
        with _lock:
            _latest["result"] = data
        print(f"[분석] detected={data['detected']} labels={data['labels']} ({data['elapsed_sec']}s)")


# ── 오버레이 ─────────────────────────────────────────────────────────────────
def _load_kr_font(size: int):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for p in _KR_FONT_PATHS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return None


_FONT = _load_kr_font(30)
_FONT_SM = _load_kr_font(26)


def _draw_kr(img, lines: list[tuple[str, int]]):
    """lines: (text, y) 목록을 한글 폰트로 그린다."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, y in lines:
        font = _FONT if y < 44 else _FONT_SM
        draw.text((16, y), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def render_overlay(frame, result: dict | None):
    img = frame.copy()
    h, w = img.shape[:2]
    use_kr = _FONT is not None

    if not result:
        color = (130, 130, 130)
        status = "분석 대기..." if use_kr else "waiting..."
        labels = []
    else:
        detected = result["detected"]
        color = (0, 0, 220) if detected else (0, 160, 0)
        elapsed = result["elapsed_sec"]
        if use_kr:
            status = ("⚠ 위험 감지" if detected else "정상") + f"   {elapsed:.1f}s"
            labels = [LABEL_KO.get(lb, lb) for lb in result["labels"]]
        else:
            status = ("DETECTED" if detected else "CLEAR") + f"   {elapsed:.1f}s"
            labels = result["labels"]

    bar_h = 80
    cv2.rectangle(img, (0, 0), (w, bar_h), color, -1)

    if use_kr:
        lines = [(status, 10)]
        if labels:
            lines.append((", ".join(labels), 46))
        img = _draw_kr(img, lines)
    else:
        cv2.putText(img, status, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if labels:
            cv2.putText(img, ", ".join(labels), (16, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def health_check() -> None:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as r:
            print("health:", json.loads(r.read()))
    except Exception as e:  # noqa: BLE001
        print("health 확인 실패(서버 미기동?):", e)


# ── 메인 루프 ────────────────────────────────────────────────────────────────
def run(video_path: str, frames_dir: str) -> None:
    frames_dir_p = Path(frames_dir)
    staging = frames_dir_p.parent / "_frames_staging"

    try:
        paths = extract_all_frames(video_path, staging, TARGET_FPS)
        if not paths:
            print("추출된 프레임이 없습니다.")
            return
        health_check()

        per_batch = max(round(TARGET_FPS * BATCH_SECONDS), MIN_FRAMES)
        delay = max(int(1000 / TARGET_FPS), 1)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        print("재생 시작 (q/ESC 종료)")

        worker: threading.Thread | None = None
        buffer: list[Path] = []
        i = 0
        window_seen = False  # 창이 실제로 떠서 보인 적이 있는지 (조기 종료 방지)
        while True:
            img = cv2.imread(str(paths[i]))
            if img is not None:
                buffer.append(paths[i])
                with _lock:
                    result = _latest["result"]
                cv2.imshow(WINDOW, render_overlay(img, result))

            busy = worker is not None and worker.is_alive()
            if not busy and len(buffer) >= per_batch:
                batch = buffer[-per_batch:]
                buffer = []
                worker = threading.Thread(
                    target=analyze_worker, args=(batch, frames_dir_p), daemon=True
                )
                worker.start()
            elif busy and len(buffer) > per_batch:
                buffer = buffer[-per_batch:]  # 분석 중엔 최신 1초만 유지

            key = cv2.waitKey(delay) & 0xFF
            if key in (27, ord("q")):
                break
            # 창 닫힘(X) 감지: 한 번 보인 뒤에만 적용. 속성 미지원(-1) 빌드에서는 무시.
            prop = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE)
            if prop >= 1:
                window_seen = True
            elif window_seen and prop == 0:
                break

            i += 1
            if i >= len(paths):  # 영상 끝 → 처음부터 반복
                i = 0
                buffer = []
    finally:
        cv2.destroyAllWindows()
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    frames = sys.argv[2] if len(sys.argv) > 2 else FRAMES_DIR
    run(video, frames)
