#!/bin/bash
set -euo pipefail

# VLM venv 활성화 (transformers 등 필요 시)
source /media/ds/DATA/duego-server-venv/.duego-vlm-server/bin/activate

export EDGELLM_PLUGIN_PATH="/home/ds/edge_llm/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"

cd "$(dirname "$0")"
python3 main.py
