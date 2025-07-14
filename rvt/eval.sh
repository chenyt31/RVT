export epoch=$1
export model_folder=$2  # PATH_TO_MODEL_FOLDER
export visualize_bbox=$3
export lang_type=$4
export DEMO_PATH_ROOT=/data1/cyt/HiMan_data/test
export TF_CPP_MIN_LOG_LEVEL=3

for root_task in \
    "basketball_in_hoop" \
    "close_box" \
    "empty_dishwasher" \
    "get_ice_from_fridge" \
    "hockey" \
    "meat_on_grill" \
    "move_hanger" \
    "wipe_desk" \
    "open_drawer" \
    "slide_block_to_target" \
    "reach_and_drag" \
    "put_money_in_safe" \
    "place_wine_at_rack_location" \
    "insert_onto_square_peg" \
    "turn_oven_on" \
    "straighten_rope" \
    "setup_chess" \
    "scoop_with_spatula" \
    "close_laptop_lid" \
    "stack_cups"; do

        
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
                --device 1 \
                --headless \
                --model-name model_${epoch}.pth \
                --colosseum \
                --save-video \
                --visualize_bbox \
                --lang_type $lang_type
        else
            uv run eval.py \
                    --model-folder $model_folder \
                    --eval-datafolder $DEMO_PATH_ROOT \
                    --tasks $task_name \
                    --eval-episodes 1 \
                    --log-name epoch_${epoch} \
                    --device 0 \
                    --headless \
                    --model-name model_${epoch}.pth \
                    --colosseum \
                    --save-video \
                    --lang_type $lang_type
        fi
    done

done