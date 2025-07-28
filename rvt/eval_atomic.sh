export epoch=$1
export model_folder=$2  # PATH_TO_MODEL_FOLDER
export visualize_bbox=$3
export lang_type=$4
export agent_type=$5
export device=$6
export port=$7

export DEMO_PATH_ROOT=/data1/cyt/HiMan_data/train
export TF_CPP_MIN_LOG_LEVEL=3

for root_task in \
    "open_drawer"; do
    for i in {0..17}; do
        export task_name=${root_task}_${i}
        export DATA_PATH=$DEMO_PATH_ROOT/$task_name/
        if [ ! -e "$DATA_PATH" ]; then
            echo "$DATA_PATH does not exist, skipping this iteration."
            continue
        fi   
        if [ $visualize_bbox == "True" ]; then
            uv run eval.py \
                --model-folder $model_folder \
                --eval-datafolder $DEMO_PATH_ROOT \
                --tasks $task_name \
                --eval-episodes 1 \
                --log-name epoch_${epoch} \
                --device $device \
                --headless \
                --model-name model_${epoch}.pth \
                --colosseum \
                --save-video \
                --visualize_bbox \
                --lang_type $lang_type \
                --agent_type $agent_type \
                --port $port \
                --tasks_type atomic
        else
            uv run eval.py \
                --model-folder $model_folder \
                --eval-datafolder $DEMO_PATH_ROOT \
                --tasks $task_name \
                --eval-episodes 20 \
                --log-name epoch_${epoch} \
                --device $device \
                --headless \
                --model-name model_${epoch}.pth \
                --colosseum \
                --save-video \
                --lang_type $lang_type \
                --agent_type $agent_type \
                --port $port \
                --tasks_type atomic
        fi
    done

done