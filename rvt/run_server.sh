#!/bin/bash

# === Usage Help ===
usage() {
  echo "Usage: bash run_remote_agent.sh --epoch <epoch> --model_folder <path> --lang_type <type> --device <device> --port <port>"
  exit 1
}

# === Argument Parsing ===
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --epoch) epoch="$2"; shift ;;
    --model_folder) model_folder="$2"; shift ;;
    --lang_type) lang_type="$2"; shift ;;
    --device) device="$2"; shift ;;
    --port) port="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

# === Check Required Params ===
if [[ -z "$epoch" || -z "$model_folder" || -z "$lang_type" || -z "$device" || -z "$port" ]]; then
  echo "Error: Missing required arguments."
  usage
fi

# === Run Remote Agent ===
uv run models/remote_agent.py \
  --model-folder "$model_folder" \
  --model-name "model_${epoch}.pth" \
  --device "$device" \
  --lang-type "$lang_type" \
  --port "$port"