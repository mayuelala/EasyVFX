# training parameters
learning_rate=1e-5 # 3e-3
ipa_layers=42
ip_scale=1.0
proportion_empty_prompts=0
ipa_cross_dim=1024
ckpt_path=null
wo_kv=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PROJECT_NAME="REFadapter"
export MODEL_PATH="Your/CogVideoX-5b-I2V/Path"
export CACHE_PATH="~/.cache"
export DATASET_PATH="aaa"
export OUTPUT_PATH="./exp_outputs/${PROJECT_NAME}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=7200
config_file="scripts/accelerate_config.yaml"
python_file="scripts/train.py"
model_dir="models"

# Parse command line arguments to obtain the user-specified num_processes, default is 1
visible_devices=7
num_processes=1
while getopts ":n:v:" opt; do
  case $opt in
    n) num_processes=$OPTARG ;;
    v) visible_devices=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG" >&2 ;;
    :) echo "Option -$OPTARG requires an argument." >&2 ;;
  esac
done
if [[ "$num_processes" -eq 1 && "$visible_devices" == *","* ]]; then
  num_processes=$(awk -F',' '{print NF}' <<< "$visible_devices")
fi
echo "num_processes: $num_processes (from visible_devices=$visible_devices)"

main_process_port=$((20117 + visible_devices))
echo "main_process_port: $main_process_port"

# Save codes and config file
mkdir -p $OUTPUT_PATH
echo "Save codes and config file to $OUTPUT_PATH"
chmod -R 777 $OUTPUT_PATH
cp $python_file $OUTPUT_PATH
cp $config_file $OUTPUT_PATH
cp $0 $OUTPUT_PATH
cp -r $model_dir $OUTPUT_PATH

which python
CUDA_VISIBLE_DEVICES=$visible_devices accelerate launch --config_file $config_file --machine_rank 0 --num_processes $num_processes --main_process_port $main_process_port \
  $python_file \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --enable_tiling \
  --enable_slicing \
  --instance_data_root $DATASET_PATH \
  --caption_column prompts.txt \
  --video_column videos.txt \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --train_batch_size 1 \
  --num_train_epochs 40000 \
  --checkpointing_steps 500 \
  --gradient_accumulation_steps 1 \
  --learning_rate $learning_rate \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 0 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to tensorboard \
  --meta_file_path /Your/data/path/meta_all_caption.csv \
  --val_meta_file_path /Your/data/path/meta_all_caption_val.csv \
  --validating_steps 500 \
  --resume_from_checkpoint latest \
  --proportion_empty_prompts $proportion_empty_prompts \
  --ckpt_path $ckpt_path \
  --max_train_steps 1000000 \
  --lora_alpha 32 \
  --rank 16 \
  --lora_weight 1.0 \
  --checkpoints_total_limit 100 \
  --use_different_first_frame \
