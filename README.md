# DueGo VLM Server

Qwen2.5-VL 모델로 건설 현장 CCTV 프레임을 분석해 **위험 행동 5종**을 판정하고, 위반 시 제지용 TTS 문구를 생성하는 VLM 추론 서버입니다.

**Jetson Thor** 에서 실행하는 것을 전제로 합니다. TensorRT-Edge-LLM 엔진과 CUDA 환경이 Jetson Thor 기준으로 빌드되어 있습니다.

> **`POST /analyze`** — 프레임 폴더 경로만 전달하면, 라벨 5종을 한 번의 배치 추론으로 판정하고 위반 결과 + TTS 문구를 JSON으로 반환합니다.
> 디버그용으로 임의 프롬프트를 직접 던지는 `/vlm`, `/safety/debug`, `/v1/debug` 엔드포인트도 제공합니다.

---

## 감지 대상 (라벨 5종)

각 라벨은 라벨별 질문(`LABEL_CRITERIA`)을 모델에 던져 `true/false/none` 답을 받고, 라벨마다 정의된 위반 기준(`VIOLATION_WHEN`)으로 위반 여부를 판정합니다. 사람이 없으면 모델이 `none`을 반환하며 위반 아님으로 처리됩니다.

| 라벨 | 모델에 묻는 질문 | 위반 조건 | TTS 문구 |
|------|------------------|:---------:|----------|
| `helmet_off` | 모든 작업자가 빨간 안전모를 썼는가? | `false`(미착용) | 안전모를 착용하세요. |
| `cone_touch` | 작업자가 주황색 라바콘을 만지는가? | `true`(접촉) | 라바콘에서 손을 떼세요. |
| `fence_crossing` | 노란 펜스 왼쪽에 사람이 있는가? | `true` | 위험 지역에 들어가지 마세요. |
| `ladder_alone` | 혼자 사다리를 사용하는 사람이 있는가? | `true` | 사다리를 혼자 사용하지 마세요. 보조자를 배치하세요. |
| `safety_vest` | 모든 작업자가 안전 조끼를 입었는가? | `false`(미착용) | 안전 조끼를 착용하세요. |

위반이 하나도 없으면 TTS는 `"안전 이상이 없습니다."`, 위반이 있으면 `"안전 이상이 발생했습니다. "` 뒤에 해당 문구들을 라벨 정의 순서대로 이어붙입니다.

---

## 환경 정보

| 항목 | 값 |
|------|----|
| 플랫폼 | Jetson Thor (aarch64) |
| Python | 3.12.3 |
| 모델 | Qwen2.5-VL-7B |
| 엔진(기본) | `qwen25-vl-7b-2k-b5` |
| 기본 포트 | 8000 |

---

## 사전 준비 (최초 1회)

### 1. Python 바인딩 빌드

```bash
source /media/ds/DATA/duego-server-venv/.duego-vlm-server/bin/activate
pip install pybind11

cd ~/edge_llm/TensorRT-Edge-LLM/experimental/pybind
mkdir -p build && cd build

cmake .. \
    -DTRT_PACKAGE_DIR=/usr \
    -DTRT_INCLUDE_DIR=/usr/include/aarch64-linux-gnu \
    -DEDGELLM_BUILD_DIR=/home/ds/edge_llm/TensorRT-Edge-LLM/build \
    -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
    -DCUDA_DIR=/usr/local/cuda-13.2

make -j$(nproc)
# 결과물: _edgellm_runtime.cpython-312-aarch64-linux-gnu.so
```

빌드된 `.so`가 있는 디렉터리를 `config.json`의 `EDGELLM_PYBIND_DIR`에 지정합니다.

### 2. 서버 패키지 설치

```bash
source /media/ds/DATA/duego-server-venv/.duego-vlm-server/bin/activate

cd /path/to/kiosk-vlm
pip install -r requirements.txt
```

---

## 서버 실행

```bash
cd /path/to/kiosk-vlm
./run.sh
```

`run.sh`는 venv를 활성화하고 `EDGELLM_PLUGIN_PATH`를 export한 뒤 `python3 main.py`를 실행하며, TensorRT 엔진의 잡다한 stderr 로그(FMHA 등)를 필터링합니다.

서버가 정상 시작되면 아래 순서로 로그가 출력됩니다.

```
설정 적용 | ENGINE_DIR=... | NUM_FRAMES=1 | FRAME_SIZE=1280 | MAX_TOKENS=16 | ...
_edgellm_runtime 모듈 로드 중...
LLMRuntime 엔진 로드 중...        ← 최초 1회, 수십 초 소요
LLMRuntime 로드 완료
Uvicorn running on http://0.0.0.0:8000
```

> 서버 하나는 LLMRuntime 하나만 사용하며, 추론 요청은 `asyncio.Lock`으로 직렬화됩니다(한 번에 한 추론). 동시 처리가 필요하면 서버 프로세스를 포트별로 따로 실행하세요.

---

## Swagger UI

서버 실행 후 브라우저에서 아래 주소로 접속하면 모든 API를 웹에서 직접 테스트할 수 있습니다.

| UI | 주소 |
|----|------|
| Swagger UI | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |

---

## 입력 이미지 규칙

서버는 지정된 폴더의 파일을 직접 읽습니다. 요청 전에 아래 형식으로 이미지를 준비하세요.

- **기본 파일명**: `frame_01.jpg`, `frame_02.jpg`, ... (또는 `frame1.jpg`, `frame2.jpg`, ...)
- **확장자**: `.jpg` / `.jpeg` / `.png` / `.bmp`
- **필수 프레임 수**: 최소 `NUM_FRAMES`장 이상 (미만이면 400 에러). 기본값은 **1장**입니다.
- 폴더에 프레임이 `NUM_FRAMES`보다 많으면 **균등 간격으로 자동 선택**합니다. (`NUM_FRAMES=1`이면 가운데 프레임 1장)
- 선택된 프레임들은 하나의 시퀀스로 모델에 입력됩니다(프레임별 독립 분석이 아님).
- 긴 변이 `FRAME_SIZE`(기본 1280px)를 넘는 이미지는 추론 전 자동 리사이즈되며, 임시 파일은 요청이 끝나면 삭제됩니다.

```
/path/to/frames/
├── frame_0001.jpg
├── frame_0002.jpg
└── ...
```

---

## API 목록

### POST `/analyze`

프레임 폴더 경로만 전달하면 라벨 5종을 **한 번의 배치 추론**으로 판정합니다.

**Request Body**

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| `dir_path` | string | ✅ | 프레임 이미지가 있는 폴더의 절대 경로 |
| `labels` | string[] | ❌ | 이 요청에서 판정할 라벨 목록. 미지정/빈 목록이면 5종 전체를 판정. 지원하지 않는 라벨이 섞이면 `400` 에러 |

```json
{ "dir_path": "/home/ds/Desktop/kiosk-vlm/frames" }
```

특정 상황만 판정하려면 `labels`로 대상을 좁힙니다(반환 순서는 라벨 정의 순서로 정규화됩니다).

```json
{
  "dir_path": "/home/ds/Desktop/kiosk-vlm/frames",
  "labels": ["helmet_off", "ladder_alone"]
}
```

**Response** (`DetectResponse`)

```json
{
  "request_id": "a1b2c3d4",
  "detected": true,
  "labels": ["helmet_off", "ladder_alone"],
  "tts_message": "안전 이상이 발생했습니다. 안전모를 착용하세요. 사다리를 혼자 사용하지 마세요. 보조자를 배치하세요.",
  "raw": "{\"helmet_off\": \"false\", \"cone_touch\": \"none\", ...}",
  "elapsed_sec": 3.456
}
```

| 필드 | 설명 |
|------|------|
| `request_id` | 요청 식별자(8자) |
| `detected` | 위반 라벨이 하나라도 있으면 `true` |
| `labels` | 위반으로 판정된 라벨 목록 |
| `tts_message` | 제지용 TTS 문구 (위반 없으면 "안전 이상이 없습니다.") |
| `raw` | 라벨별 모델 원문 응답을 담은 JSON 문자열 |
| `elapsed_sec` | 리사이즈+추론 전체 소요 시간(초) |

```bash
# 5종 전체 판정
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"dir_path": "/home/ds/Desktop/kiosk-vlm/frames"}'

# 특정 라벨만 판정
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"dir_path": "/home/ds/Desktop/kiosk-vlm/frames", "labels": ["helmet_off", "ladder_alone"]}'
```

---

### POST `/vlm`

이미지 **한 장** 경로 + 프롬프트로 VLM을 직접 호출해 응답 원문을 반환합니다. 안전판정 로직을 거치지 않습니다.

**Request Body**

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| `image_path` | string | ✅ | 이미지 파일 절대 경로 |
| `prompt` | string | ✅ | 모델에 전달할 유저 프롬프트 |
| `system_prompt` | string | ❌ | 생략 시 중립적 장면 설명 프롬프트 사용 |

```bash
curl -X POST "http://localhost:8000/vlm" \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/frame_0001.jpg", "prompt": "이 사진에 보이는 걸 설명해줘."}'
```

**Response**: `request_id`, `response`(원문), `image`, `image_fed`(리사이즈 후 경로), `elapsed_sec`

---

### POST `/safety/debug`

이미지 한 장 **또는 프레임 폴더** + 임의 프롬프트로 VLM을 직접 호출합니다. 폴더면 `/analyze`와 동일하게 프레임을 선별해 연속 시퀀스로 입력하므로, 실제 입력 조건 그대로 프롬프트를 시험할 수 있습니다.

**Request Body**

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| `path` | string | ✅ | 이미지 파일 또는 프레임 폴더 경로 |
| `prompt` | string | ✅ | 유저 프롬프트 |
| `system_prompt` | string | ❌ | 생략 시 중립적 설명 프롬프트 사용 |

**Response**: `request_id`, `response`, `frames_used`, `images_fed`(모델에 실제 입력된 프레임의 경로/크기/로드여부), `elapsed_sec`

---

### POST `/v1/debug`

`dir_path`만 받아 안전판정 없이 프레임에 보이는 장면을 **그대로 설명만** 합니다. (입력 검증/리사이즈는 `/analyze`와 동일)

```json
{ "dir_path": "/home/ds/Desktop/kiosk-vlm/frames" }
```

**Response**: `request_id`, `description`, `frames_used`, `images_fed`, `elapsed_sec`

---

### GET `/health`

서버 생존 여부 확인. `{ "status": "ok" }`

---

### GET `/config`

현재 적용된 설정값 전체를 조회합니다. (응답 형식은 아래 설정 표 참조)

```bash
curl "http://localhost:8000/config"
```

### PUT `/config`

변경할 필드만 보내는 **부분 업데이트**입니다. 변경 내용은 `config.json`에 저장되어 재시작 후에도 유지되며, 갱신된 설정 전체를 응답으로 돌려줍니다.

- **즉시 적용**(다음 요청부터): `NUM_FRAMES`, `FRAME_SIZE`, `JPEG_QUALITY`, `RESIZE_TMP_DIR`, `MAX_TOKENS`, `TEMPERATURE`, `TOP_P`, `TOP_K`
- **재시작 후 적용**: `ENGINE_DIR`, `PLUGIN_PATH`, `EDGELLM_PYBIND_DIR`, `HOST`, `PORT`
- 알 수 없는 키나 타입이 맞지 않는 값은 `400` 에러로 거부됩니다.

```bash
curl -X PUT "http://localhost:8000/config" \
  -H "Content-Type: application/json" \
  -d '{"MAX_TOKENS": 32, "NUM_FRAMES": 3}'
```

---

## 설정 파일 (`config.json`)

서버 최초 실행 시 `config.py`의 기본값으로 자동 생성됩니다. 직접 수정하거나 `PUT /config`로 변경할 수 있으며, 직접 수정한 경우 재시작해야 반영됩니다.

| 키 | 기본값 | 재시작 필요 | 설명 |
|----|--------|:-----------:|------|
| `ENGINE_DIR` | `/media/ds/DATA/engines/qwen25-vl-7b-2k-b5` | ✅ | TensorRT 엔진 경로 (멀티모달 엔진도 동일 경로 사용) |
| `PLUGIN_PATH` | `.../libNvInfer_edgellm_plugin.so` | ✅ | EdgeLLM 플러그인 `.so` 경로 |
| `EDGELLM_PYBIND_DIR` | `.../experimental/pybind/build` | ✅ | `_edgellm_runtime` 바인딩 `.so` 위치 |
| `NUM_FRAMES` | `1` | ❌ | 균등 선택할 프레임 수 (이 값 미만이면 400) |
| `FRAME_SIZE` | `1280` | ❌ | 리사이즈 목표 변(긴 쪽 기준) 픽셀 |
| `JPEG_QUALITY` | `95` | ❌ | 리사이즈 JPEG 저장 품질 (1~100) |
| `RESIZE_TMP_DIR` | `/tmp/vlm_resized` | ❌ | 리사이즈 임시 디렉터리 (요청별로 생성/삭제) |
| `MAX_TOKENS` | `16` | ❌ | 최대 출력 토큰 수 (true/false 판정엔 작게 유지) |
| `TEMPERATURE` | `0.0` | ❌ | 샘플링 온도 (0.0 = 그리디/결정론적) |
| `TOP_P` | `1.0` | ❌ | nucleus 샘플링 top-p |
| `TOP_K` | `1` | ❌ | top-k 샘플링 |
| `HOST` | `0.0.0.0` | ✅ | 바인딩 호스트 |
| `PORT` | `8000` | ✅ | 바인딩 포트 |

> `TEMPERATURE=0.0`(그리디)일 때는 `TOP_P=1.0`, `TOP_K=1`로 두어야 수치 불안정/경고가 없습니다.

---

## 실시간 테스트 (`test_request.py`)

영상을 창에 재생하면서 1초(약 30프레임) 단위로 `/analyze`를 호출하고, 판정 결과를 영상 위에 실시간 오버레이로 그리는 CCTV 시뮬레이션 스크립트입니다.

```bash
python3 test_request.py
python3 test_request.py /path/to/video.mp4 /path/to/frames
```

- `q` 또는 `ESC`로 종료, 영상이 끝나면 처음부터 반복합니다.
- 분석 중에는 다음 분석을 보류하고 영상은 계속 재생합니다(최신 1초치만 버퍼 유지).
- 스크립트 상단의 `VIDEO_PATH`, `FRAMES_DIR`, `BATCH_SECONDS`, `MIN_FRAMES`를 환경에 맞게 조정하세요. `MIN_FRAMES`는 서버의 `NUM_FRAMES` 이상이어야 합니다.
- 한글 오버레이는 NanumGothic / NotoSansCJK 폰트가 있으면 자동 사용합니다.
