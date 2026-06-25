# DueGo VLM Server

Qwen2.5-VL-7B 모델을 이용해 연속 프레임 시퀀스를 분석하는 VLM+LLM 2단계 추론 서버.  
**Jetson Thor** 에서 실행해야 합니다. TensorRT-Edge-LLM 엔진과 CUDA 환경이 Jetson Thor 기준으로 빌드되어 있습니다.

> **`/analyze`** — dir 경로만 전달하면 VLM(장면 설명) → LLM(위험 행동 감지) 파이프라인을 자동 수행하고 JSON 결과를 반환합니다.  
> **`/infer`** — 기존 범용 VLM 추론 엔드포인트. raw text를 그대로 반환합니다.

---

## 환경 정보

| 항목 | 값 |
|------|----|
| 플랫폼 | Jetson Thor (aarch64) |
| Python | 3.12.3 |
| CUDA | 13.2 |
| TensorRT | 10.16.2 |
| 엔진 | qwen25-vl-7b-4k-b1-10x448 (4K, batch=1) |
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

### 2. 서버 패키지 설치

```bash
source /media/ds/DATA/vlm_venv/.qwen25-vl-7b-venv/bin/activate

cd /path/to/duego_vlm_server
pip install -r requirements.txt
```

---

## 서버 실행

```bash
cd /path/to/duego_vlm_server
./run.sh
```

포트를 지정해서 실행할 수도 있습니다. 서버 2개를 동시에 띄울 때 사용합니다.

```bash
PORT=8000 ./run.sh   # 첫 번째 서버
PORT=8001 ./run.sh   # 두 번째 서버 (별도 터미널)
```

> 서버 하나는 LLMRuntime 하나만 사용합니다. 동시 처리가 필요하면 서버 프로세스를 포트별로 따로 실행하세요.

서버가 정상 시작되면 아래 순서로 로그가 출력됩니다.

```
_edgellm_runtime 모듈 로드 중...
LLMRuntime 엔진 로드 중...        ← 최초 1회, 수십 초 소요
LLMRuntime 로드 완료
워커 4개 기동 완료
Uvicorn running on http://0.0.0.0:8000
```

엔진 로드 완료 이후부터는 추론만 실행됩니다.

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

- **기본 파일명** (권장): `frame_01.jpg`, `frame_02.jpg`, ..., `frame_30.jpg`
- **호환 파일명**: `frame1.jpg`, `frame2.jpg`, ..., `frame30.jpg`
- **확장자**: `.jpg` / `.jpeg` / `.png` / `.bmp`
- **필수 프레임 수**: 최소 **15장** 이상 (미만이면 400 에러)
- `/infer` — 15장 초과 시 번호 순으로 앞 15장만 사용합니다.
- `/analyze` — 30장 중 **균등 간격으로 15장**을 자동 선택합니다.
- 15장은 하나의 행동 시퀀스로 처리되며, 프레임별 독립 분석이 아닙니다.

```
/path/to/frames/
├── frame_01.jpg
├── frame_02.jpg
├── ...
└── frame_30.jpg    ← /analyze 시 15장 균등 선택
```

> **주의**: 4K b1 엔진은 448×448 이미지 15장이 한계에 가깝습니다.  
> prompt가 길면 maxInputLen 4096을 초과할 수 있으므로 prompt는 짧게 유지하세요.

---

## API 목록

### POST `http://localhost:8000/analyze`

dir 경로만 전달하면 **VLM → LLM 2단계 파이프라인**을 자동 수행합니다.

1. 폴더에서 30프레임 중 **15장을 균등 간격으로 선택**
2. **VLM** — 선택된 15장으로 장면/행동을 자세히 설명
3. **LLM** — VLM 설명에서 4가지 위험 행동을 감지하고 JSON 반환

**감지 대상 행동**

| 키 | 설명 |
|----|------|
| `hat_action` | 안전모를 착용하지 않았거나 벗는 행동 |
| `touch_action` | 스피커를 만지는 행동 |
| `dangerInOut_action` | 금지 구역에 출입하는 행동 |
| `ladder_action` | 사다리를 올라가거나 단독 사다리 작업 |

**Request Body**

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| `dir_path` | string | ✅ | 프레임 이미지가 있는 폴더의 절대 경로 |

```json
{
  "dir_path": "/home/ds/Desktop/vlm_test/frames_448_30"
}
```

**Response**

```json
{
  "request_id": "a1b2c3d4",
  "action": "hat_action,ladder_action",
  "tts_message": "안전모를 착용하지 않은 상태에서 단독 사다리 작업을 하고 있어 위험합니다. 안전모를 착용하고 사다리 작업을 중단하십시오",
  "vlm_description": "VLM이 생성한 장면 설명 텍스트",
  "elapsed_sec": 3.456
}
```

| 필드 | 설명 |
|------|------|
| `action` | 감지된 위험 행동 키를 쉼표로 구분. 없으면 빈 문자열 |
| `tts_message` | 감지된 행동에 대한 TTS 경고 메시지. 없으면 빈 문자열 |
| `vlm_description` | VLM 단계에서 생성된 장면 설명 원문 |
| `elapsed_sec` | VLM+LLM 전체 소요 시간(초) |

**curl 예시**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"dir_path": "/home/ds/Desktop/vlm_test/frames_448_30"}'
```

---

### POST `http://localhost:8000/infer`

15장의 연속 프레임을 하나의 시퀀스로 입력해 모델이 생성한 텍스트를 반환합니다. (범용 VLM 추론)

**Request Body**

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| `folder_path` | string | ✅ | 프레임 이미지가 있는 폴더의 절대 경로 |
| `prompt` | string | ❌ | 모델에 전달할 프롬프트. 생략 시 `DEFAULT_PROMPT` 사용 |
| `max_tokens` | int | ❌ | 최대 출력 토큰 수. 생략 시 `MAX_TOKENS`(128) 사용 |

```json
{
  "folder_path": "/home/ds/Desktop/vlm_test/frames_448_15_5fps",
  "prompt": "15장의 연속 프레임에서 사람이 한 행동을 한 문장으로 설명해줘.",
  "max_tokens": 64
}
```

**Response**

결과는 항상 시퀀스 전체에 대한 결과 1개입니다.

```json
{
  "request_id": "a1b2c3d4",
  "folder_path": "/home/ds/Desktop/vlm_test/frames_448_15_5fps",
  "results": [
    {
      "frame": "sequence",
      "output_text": "모델이 생성한 원본 텍스트"
    }
  ],
  "elapsed_sec": 1.234
}
```

**curl 예시 — 한국어 설명 요청**

```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "/home/ds/Desktop/vlm_test/frames_448_15_5fps",
    "prompt": "15장의 연속 프레임을 보고 사람이 어떤 행동을 했는지 한국어 한 문장으로 설명해줘.",
    "max_tokens": 64
  }'
```

**curl 예시 — 라벨 출력 요청**

```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "/home/ds/Desktop/vlm_test/frames_448_15_5fps",
    "prompt": "Classify the main action in the sequential frames. Output one short phrase only.",
    "max_tokens": 16
  }'
```

---

### GET `http://localhost:8000/status`

워커 슬롯의 실행 상태와 대기 큐를 조회합니다.

**Response**

```json
{
  "slots": [
    {
      "slot_id": 0,
      "status": "running",
      "request_id": "a1b2c3d4",
      "folder_path": "/home/ds/Desktop/vlm_test/frames_448_15_5fps",
      "running_sec": 1.2
    },
    { "slot_id": 1, "status": "idle", "request_id": null, "folder_path": null, "running_sec": null },
    { "slot_id": 2, "status": "idle", "request_id": null, "folder_path": null, "running_sec": null },
    { "slot_id": 3, "status": "idle", "request_id": null, "folder_path": null, "running_sec": null }
  ],
  "queue": [
    {
      "request_id": "e5f6a7b8",
      "folder_path": "/home/ds/Desktop/vlm_test/frames_448_15_5fps",
      "queued_sec": 0.5
    }
  ],
  "queue_length": 1
}
```

| 필드 | 설명 |
|------|------|
| `slots` | 워커 슬롯 상태. `status`는 `idle` 또는 `running` |
| `running_sec` | 현재 요청이 실행된 시간(초) |
| `queue` | 빈 슬롯을 기다리는 대기 요청 목록 |
| `queue_length` | 대기 중인 요청 수 |

---

### GET `http://localhost:8000/config`

현재 서버에 적용된 설정값 전체를 조회합니다.

**Response**

```json
{
  "ENGINE_DIR": "/media/ds/DATA/engines/qwen25-vl-7b-4k-b1-10x448",
  "PLUGIN_PATH": "/home/ds/edge_llm/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so",
  "EDGELLM_PYBIND_DIR": "/home/ds/edge_llm/TensorRT-Edge-LLM/experimental/pybind/build",
  "REFERENCE_IMAGE_PATH": "/mnt/user-data/uploads/ChatGPT_Image_2026년_6월_25일_오후_02_19_27.png",
  "NUM_FRAMES": 15,
  "FRAME_SIZE": 448,
  "JPEG_QUALITY": 95,
  "RESIZE_TMP_DIR": "/tmp/vlm_resized",
  "MAX_TOKENS": 384,
  "TEMPERATURE": 0.0,
  "TOP_P": 0.9,
  "TOP_K": 50,
  "CONFIDENCE_THRESHOLD": {
    "helmet_off": 0.6,
    "cone_touch": 0.6,
    "fence_crossing": 0.6,
    "ladder_alone": 0.6
  },
  "HOST": "0.0.0.0",
  "PORT": 8000
}
```

**curl 예시**

```bash
curl "http://localhost:8000/config"
```

---

### PUT `http://localhost:8000/config`

변경할 필드만 포함해서 전송하는 **부분 업데이트**입니다. 변경 내용은 `config.json`에 저장되어 재시작 후에도 유지되며, 갱신된 설정 전체를 응답으로 돌려줍니다.

- **즉시 적용** (다음 요청부터 반영): `NUM_FRAMES`, `FRAME_SIZE`, `JPEG_QUALITY`, `RESIZE_TMP_DIR`, `MAX_TOKENS`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `CONFIDENCE_THRESHOLD`
- **재시작 후 적용** (시작 시 1회 사용): `ENGINE_DIR`, `PLUGIN_PATH`, `EDGELLM_PYBIND_DIR`, `REFERENCE_IMAGE_PATH`, `HOST`, `PORT`
- `CONFIDENCE_THRESHOLD`는 **라벨 단위로 병합**됩니다. 일부 라벨만 보내면 나머지 라벨 임계값은 그대로 유지됩니다.
- 알 수 없는 키나 타입이 맞지 않는 값은 `400` 에러로 거부됩니다.

**Request Body** (예: 출력 토큰을 늘리고 안전모 임계값만 올림)

```json
{
  "MAX_TOKENS": 512,
  "CONFIDENCE_THRESHOLD": {
    "helmet_off": 0.8
  }
}
```

**Response** — 갱신된 설정 전체 (`GET /config`와 동일한 형식)

```json
{
  "MAX_TOKENS": 512,
  "CONFIDENCE_THRESHOLD": {
    "helmet_off": 0.8,
    "cone_touch": 0.6,
    "fence_crossing": 0.6,
    "ladder_alone": 0.6
  },
  "...": "(나머지 필드 동일)"
}
```

**curl 예시**

```bash
curl -X PUT "http://localhost:8000/config" \
  -H "Content-Type: application/json" \
  -d '{
    "MAX_TOKENS": 512,
    "CONFIDENCE_THRESHOLD": { "helmet_off": 0.8 }
  }'
```

---

### GET `http://localhost:8000/health`

서버 생존 여부를 확인합니다.

**Response**

```json
{ "status": "ok" }
```

---

## 설정 파일 (`config.json`)

서버 최초 실행 시 자동 생성됩니다. 직접 수정하거나 `PUT /config` API로 변경할 수 있습니다. 직접 수정한 경우에는 재시작해야 반영됩니다.

| 키 | 기본값 | 재시작 필요 | 설명 |
|----|--------|:-----------:|------|
| `ENGINE_DIR` | `/media/ds/DATA/engines/qwen25-vl-7b-4k-b1-10x448` | ✅ | TensorRT 엔진 경로 |
| `PLUGIN_PATH` | `.../libNvInfer_edgellm_plugin.so` | ✅ | EdgeLLM 플러그인 경로 |
| `EDGELLM_PYBIND_DIR` | `.../experimental/pybind/build` | ✅ | Python 바인딩 `.so` 위치 |
| `REFERENCE_IMAGE_PATH` | `.../ChatGPT_Image_...png` | ✅ | 참조(few-shot) 콜라주 이미지 경로. 시작 시 1회 리사이즈해 캐시 |
| `NUM_FRAMES` | `15` | ❌ | 균등 선택할 프레임 수 (이 값 미만이면 400) |
| `FRAME_SIZE` | `448` | ❌ | 리사이즈 목표 변(긴 쪽 기준) 픽셀 |
| `JPEG_QUALITY` | `95` | ❌ | 리사이즈 JPEG 저장 품질 (1~100) |
| `RESIZE_TMP_DIR` | `/tmp/vlm_resized` | ❌ | 리사이즈 결과 캐시 디렉터리 |
| `MAX_TOKENS` | `384` | ❌ | 최대 출력 토큰 수 |
| `TEMPERATURE` | `0.0` | ❌ | 샘플링 온도 (0.0 = 결정론적) |
| `TOP_P` | `0.9` | ❌ | nucleus 샘플링 top-p |
| `TOP_K` | `50` | ❌ | top-k 샘플링 |
| `CONFIDENCE_THRESHOLD` | `{helmet_off, cone_touch, fence_crossing, ladder_alone: 0.6}` | ❌ | 라벨별 알림 임계값. 오탐 잦으면 ↑, 미탐 잦으면 ↓ |
| `HOST` | `0.0.0.0` | ✅ | 바인딩 호스트 |
| `PORT` | `8000` | ✅ | 바인딩 포트 |
