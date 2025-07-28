export epoch=$1
export model_folder=$2  # PATH_TO_MODEL_FOLDER
export visualize_bbox=$3
export lang_type=$4
export agent_type=$5
export device=$6
export port=$7
export DEMO_PATH_ROOT=/data1/cyt/HiMan_data/train_val
export TF_CPP_MIN_LOG_LEVEL=3

for root_task in \
    "open_drawer"; do
    # uv run debugpy --listen 5678 --wait-for-client eval.py \
    uv run eval.py \
        --model-folder $model_folder \
        --eval-datafolder $DEMO_PATH_ROOT \
        --tasks $root_task \
        --eval-episodes 20 \
        --log-name epoch_${epoch} \
        --device $device \
        --headless \
        --model-name model_${epoch}.pth \
        --save-video \
        --lang_type $lang_type \
        --agent_type $agent_type \
        --port $port
done