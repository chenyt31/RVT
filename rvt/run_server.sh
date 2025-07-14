export epoch=$1
export model_folder=$2  # PATH_TO_MODEL_FOLDER
export lang_type=$3

uv run models/remote_agent.py \
 --model-folder $model_folder \
 --model-name model_${epoch}.pth \
 --device 0 \
 --lang-type $lang_type