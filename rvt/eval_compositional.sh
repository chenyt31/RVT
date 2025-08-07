#!/bin/bash

# === Usage Help ===
usage() {
  echo "Usage: bash eval_compositional.sh --epoch <epoch> --model_folder <path> --visualize_bbox <True|False> --lang_type <type> --agent_type <type> --device <device> --port <port>"
  exit 1
}

# === Argument Parsing ===
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --epoch) epoch="$2"; shift ;;
        --model_folder) model_folder="$2"; shift ;;
        --visualize_bbox) visualize_bbox="$2"; shift ;;
        --lang_type) lang_type="$2"; shift ;;
        --agent_type) agent_type="$2"; shift ;;
        --device) device="$2"; shift ;;
        --port) port="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# === Check Required Params ===
if [[ -z "$epoch" || -z "$model_folder" || -z "$visualize_bbox" || -z "$lang_type" || -z "$agent_type" || -z "$device" || -z "$port" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# === Env Settings ===
DEMO_PATH_ROOT="/data1/cyt/HiMan_data/test_compositional"
export TF_CPP_MIN_LOG_LEVEL=3

# === Compositional Tasks ===
root_tasks=(
    "box_exchange"
    "put_in_and_close"
    "put_in_without_close"
    "put_two_in_different"
    "put_two_in_same"
    "retrieve_and_sweep"
    "sweep_and_drop"
    "take_out_and_close"
    "take_out_without_close"
    "take_two_out_of_different"
    "take_two_out_of_same"
    "transfer_box"
)

# === Evaluation Loop ===
for root_task in "${root_tasks[@]}"; do
    for i in {0..17}; do
        task_name="${root_task}_${i}"
        DATA_PATH="${DEMO_PATH_ROOT}/${task_name}/"

        if [ ! -d "$DATA_PATH" ]; then
            echo "[Skip] $DATA_PATH does not exist."
            continue
        fi

        cmd_args=(
            uv run eval.py
            --model-folder "$model_folder"
            --eval-datafolder "$DEMO_PATH_ROOT"
            --tasks "$task_name"
            --eval-episodes 1
            --log-name "epoch_${epoch}"
            --device "$device"
            --headless
            --model-name "model_${epoch}.pth"
            --colosseum
            --save-video
            --lang_type "$lang_type"
            --agent_type "$agent_type"
            --port "$port"
            --tasks_type "compositional"
        )

        if [[ "$visualize_bbox" == "True" ]]; then
            cmd_args+=(--visualize_bbox)
        fi

        echo "[Run] Evaluating $task_name ..."
        "${cmd_args[@]}"
    done
done