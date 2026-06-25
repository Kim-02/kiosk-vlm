"""
영상에서 30fps로 프레임을 추출한 뒤 /analyze 로 분석한다.

그냥 실행하면 아래 기본값으로 전 과정을 수행한다:
  python3 test_request.py

경로를 바꾸고 싶으면 인자로 넘긴다:
  python3 test_request.py /path/to/video.mp4 /path/to/frames
"""
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2

BASE_URL = "http://localhost:8000"

VIDEO_PATH = "/media/ds/DATA/videos/VLM_1.mp4"
FRAMES_DIR = "/home/ds/Desktop/kiosk-vlm/frames"
TARGET_FPS = 30.0


def extract_frames(video_path: str, frames_dir: str, target_fps: float = TARGET_FPS) -> int:
    """영상을 읽어 target_fps 간격으로 frame_0001.jpg 형식으로 저장한다.

    기존 frame_*.jpg 는 모두 지우고 새로 채운다. 저장한 프레임 수를 반환한다.
    """
    src = Path(video_path)
    if not src.is_file():
        raise FileNotFoundError(f"영상을 찾을 수 없습니다: {video_path}")

    out_dir = Path(frames_dir)
    if out_dir.exists():
        # 이전 프레임 정리 (frame_*.jpg 만 제거)
        for old in out_dir.glob("frame_*.jpg"):
            old.unlink()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    # 원본이 target_fps 보다 느리면 모든 프레임을 저장
    step = max(native_fps / target_fps, 1.0)
    print(f"영상 FPS={native_fps:.2f} → 추출 FPS={target_fps:.0f} (step={step:.2f})")

    saved = 0
    next_capture = 0.0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= next_capture:
            saved += 1
            out = out_dir / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(out), frame)
            next_capture += step
        idx += 1

    cap.release()
    print(f"프레임 추출 완료: {saved}장 → {out_dir}")
    return saved


def health_check() -> None:
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as r:
        print("health:", json.loads(r.read()))


def analyze(dir_path: str) -> None:
    payload = json.dumps({"dir_path": dir_path}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/analyze",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code, e.read().decode())
        return

    print(f"\nrequest_id: {data['request_id']}")
    print(f"elapsed:    {data['elapsed_sec']}s")
    print(f"detected:   {data['detected']}")
    print(f"labels:     {data['labels']}")
    print(f"TTS:        {data['tts_message']}")
    print("\n[감지 상세]")
    if data["details"]:
        for d in data["details"]:
            print(f"  - {d['label']} (conf={d['confidence']:.2f}): {d['evidence']}")
    else:
        print("  (없음)")
    print(f"\n[원문]\n{data['raw']}")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    frames_dir = sys.argv[2] if len(sys.argv) > 2 else FRAMES_DIR

    extract_frames(video_path, frames_dir)
    health_check()
    analyze(frames_dir)
