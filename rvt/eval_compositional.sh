export epoch=$1
export model_folder=$2  # PATH_TO_MODEL_FOLDER
export visualize_bbox=$3
export lang_type=$4
export agent_type=$5
export device=$6
export port=$7

export DEMO_PATH_ROOT=/data1/cyt/HiMan_data/test_compositional
export TF_CPP_MIN_LOG_LEVEL=3

for root_task in \
    "box_exchange" \
    "put_in_and_close" \
    "put_in_without_close" \
    "put_two_in_different" \
    "put_two_in_same" \
    "retrieve_and_sweep" \
    "sweep_and_drop" \
    "take_out_and_close" \
    "take_out_without_close" \
    "take_two_out_of_different" \
    "take_two_out_of_same" \
    "transfer_box"; do
    for i in {0..17}; do
        export task_name=${root_task}_${i}
        export DATA_PATH=$DEMO_PATH_ROOT/$task_name/
        if [ ! -e "$DATA_PATH" ]; then
            echo "$DATA_PATH does not exist, skipping this iteration."
            continue
        fi   
        if [ "$visualize_bbox" == "True" ]; then
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
                --tasks_type compositional
        else
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
                --lang_type $lang_type \
                --agent_type $agent_type \
                --port $port \
                --tasks_type compositional
        fi
    done

done