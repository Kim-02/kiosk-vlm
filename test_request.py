"""
사용:
  python3 test_request.py /path/to/frames
  python3 test_request.py /path/to/frames "사다리, 안전모"
"""
import sys
import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"


def health_check():
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as r:
        print("health:", json.loads(r.read()))


def analyze(dir_path: str, focus: str = None):
    body = {"dir_path": dir_path}
    if focus:
        body["focus"] = focus

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/analyze",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read())
        print(f"elapsed: {data['elapsed_sec']}s")
        print(f"\n[VLM 설명]\n{data['vlm_description']}")
        print(f"\n[감지 행동] {data['action']}")
        print(f"[TTS 메시지] {data['tts_message']}")
    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code, e.read().decode())


if __name__ == "__main__":
    health_check()
    if len(sys.argv) < 2:
        print("사용법: python3 test_request.py /path/to/frames [focus]")
        sys.exit(1)

    focus = sys.argv[2] if len(sys.argv) > 2 else None
    analyze(sys.argv[1], focus)
