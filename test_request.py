"""
영상을 1초(30프레임) 단위로 잘라 같은 디렉터리에 갱신하면서
매 초마다 /analyze 로 VLM 분석을 반복한다. (실시간 CCTV 시뮬레이션)

흐름:
  1) 영상 전체를 30fps 로 staging 폴더에 추출
  2) 1초치(30장)를 FRAMES_DIR 에 덮어쓰고 → /analyze 호출
  3) 응답을 받으면 다음 1초치로 FRAMES_DIR 을 갱신하고 → 다시 /analyze
  4) 영상이 끝날 때까지 반복

실행:
  python3 test_request.py
  python3 test_request.py /path/to/video.mp4 /path/to/frames
"""
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "http://localhost:8000"

VIDEO_PATH = "/media/ds/DATA/videos/VLM_1.mp4"
FRAMES_DIR = "/home/ds/Desktop/kiosk-vlm/frames"
TARGET_FPS = 30.0
BATCH_SECONDS = 1.0   # 한 번에 분석할 영상 길이(초) → 1초당 30장
MIN_FRAMES = 15       # 서버 config.NUM_FRAMES 와 동일해야 함 (미만이면 400)


def _clear_frames(d: Path) -> Path:
    if d.exists():
        for old in d.glob("frame_*.jpg"):
            old.unlink()
    else:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_with_ffmpeg(video_path: str, out_dir: Path, target_fps: float) -> None:
    """ffmpeg 로 영상 전체를 target_fps 로 frame_000001.jpg 형식으로 저장."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={target_fps:g}",
        "-q:v", "2",
        "-start_number", "1",
        str(out_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)


def _extract_with_cv2(video_path: str, out_dir: Path, target_fps: float) -> None:
    """ffmpeg 가 없을 때의 폴백. OpenCV 로 영상 전체를 순회한다."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = max(native_fps / target_fps, 1.0)  # 원본이 더 느리면 모든 프레임 저장
    print(f"  영상 FPS={native_fps:.2f} → 추출 FPS={target_fps:.0f} (step={step:.2f})")

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


def extract_all_frames(video_path: str, staging_dir: Path, target_fps: float) -> list[Path]:
    """영상 전체를 target_fps 로 staging_dir 에 추출하고 정렬된 경로 목록을 반환한다."""
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"영상을 찾을 수 없습니다: {video_path}")

    _clear_frames(staging_dir)
    if shutil.which("ffmpeg"):
        print("ffmpeg 로 전체 프레임 추출 중...")
        _extract_with_ffmpeg(video_path, staging_dir, target_fps)
    else:
        print("ffmpeg 없음 → OpenCV 로 전체 프레임 추출 중...")
        _extract_with_cv2(video_path, staging_dir, target_fps)

    frames = sorted(staging_dir.glob("frame_*.jpg"))
    print(f"전체 {len(frames)}장 추출 (~{len(frames) / target_fps:.1f}초 @ {target_fps:g}fps)")
    return frames


def write_batch(batch: list[Path], frames_dir: Path) -> None:
    """1초치 프레임을 FRAMES_DIR 에 frame_0001.jpg.. 로 덮어쓴다."""
    _clear_frames(frames_dir)
    for i, src in enumerate(batch, 1):
        shutil.copyfile(src, frames_dir / f"frame_{i:04d}.jpg")


def health_check() -> None:
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as r:
        print("health:", json.loads(r.read()))


def analyze(dir_path: str) -> None:
    """FRAMES_DIR 을 /analyze 로 보내고 결과를 한 줄로 요약 출력한다."""
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
        print("  HTTP Error:", e.code, e.read().decode())
        return

    flag = "⚠ 감지" if data["detected"] else "정상"
    print(f"  {flag} | labels={data['labels']} | {data['elapsed_sec']}s")
    if data["tts_message"]:
        print(f"  TTS: {data['tts_message']}")
    for d in data["details"]:
        print(f"    - {d['label']} (conf={d['confidence']:.2f}): {d['evidence']}")


def run(video_path: str, frames_dir: str) -> None:
    frames_dir_p = Path(frames_dir)
    staging = frames_dir_p.parent / "_frames_staging"
    per_batch = round(TARGET_FPS * BATCH_SECONDS)  # 30

    try:
        all_frames = extract_all_frames(video_path, staging, TARGET_FPS)
        health_check()

        total = (len(all_frames) + per_batch - 1) // per_batch
        for sec, start in enumerate(range(0, len(all_frames), per_batch), 1):
            batch = all_frames[start:start + per_batch]
            if len(batch) < MIN_FRAMES:
                print(f"[{sec}/{total}] 프레임 {len(batch)}장 < {MIN_FRAMES}장, 건너뜀")
                continue
            write_batch(batch, frames_dir_p)
            print(f"[{sec}/{total}] {len(batch)}프레임 갱신 → /analyze")
            analyze(frames_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    frames = sys.argv[2] if len(sys.argv) > 2 else FRAMES_DIR
    run(video, frames)
