"""
빠른 동작 확인용 테스트 스크립트
사용: python3 test_request.py /path/to/frames/folder
"""
import sys
import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"


def health_check():
    url = f"{BASE_URL}/health"
    with urllib.request.urlopen(url, timeout=5) as r:
        print("health:", json.loads(r.read()))


def infer(folder_path: str, prompt: str = "이 이미지들을 분석해서 위험 상황이나 주요 상황을 설명해줘."):
    payload = json.dumps({"folder_path": folder_path, "prompt": prompt}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/infer",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        print(f"elapsed: {data['elapsed_sec']}s")
        for item in data["results"]:
            print(f"\n[{item['frame']}]")
            print(item["output_text"])
    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code, e.read().decode())


if __name__ == "__main__":
    health_check()
    if len(sys.argv) > 1:
        infer(sys.argv[1])
    else:
        print("폴더 경로를 인자로 넘겨주세요: python3 test_request.py /path/to/frames")
