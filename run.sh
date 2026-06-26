#!/bin/bash
set -euo pipefail

# VLM venv 활성화
source /media/ds/DATA/duego-server-venv/.duego-vlm-server/bin/activate

export EDGELLM_PLUGIN_PATH="/home/ds/edge_llm/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"

cd "$(dirname "$0")"

python3 main.py 2> >(awk '
  !/FMHA DEBUG/ &&
  !/FMHA SELECTED/ &&
  !/FMHA FUNC ATTR: result=0/ &&
  !/Switching optimization profile/
' >&2)