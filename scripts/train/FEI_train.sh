# training parameters
learning_rate=3e-3 # 3e-3
ipa_layers=42
ip_scale=1.0
proportion_empty_prompts=0
ipa_cross_dim=1024
lora_weight=0.7
lora_step=40000
ckpt_path="/RefAdapter/pytorch_model.pt"
wo_kv=0

# Parse command line arguments to obtain the user-specified num_processes, default is 1
visible_devices=4
num_processes=1
motion_type=null
while getopts ":n:v:a:" opt; do
  case $opt in
    n) num_processes=$OPTARG ;;
    v) visible_devices=$OPTARG ;; # visible devices
    a) motion_type=$OPTARG ;;  # motion type
    \?) echo "Invalid option: -$OPTARG" >&2 ;;
    :) echo "Option -$OPTARG requires an argument." >&2 ;;
  esac
done

if [[ "$num_processes" -eq 1 && "$visible_devices" == *","* ]]; then
  num_processes=$(awk -F',' '{print NF}' <<< "$visible_devices")
fi
echo "num_processes: $num_processes (from visible_devices=$visible_devices)"

export PROJECT_NAME="vidu_zongzi_wrap_${motion_type}_${lora_weight}lora_${lora_step}lorasteps"
export MODEL_PATH="Please set your CogVideoX-5b-I2V path"
export OUTPUT_PATH="./exp_outputs/${PROJECT_NAME}"
export CACHE_PATH="~/.cache"
export DATASET_PATH="aaa"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=7200
config_file="scripts/accelerate_config.yaml"
python_file="scripts/train-laplas.py"
model_dir="models"

main_process_port=$((20117 + visible_devices))
echo "main_process_port: $main_process_port"

# Save codes and config file
mkdir -p "$OUTPUT_PATH"
echo "Save codes and config file to $OUTPUT_PATH"
chmod -R 777 "$OUTPUT_PATH"
cp $python_file $OUTPUT_PATH
cp $config_file $OUTPUT_PATH
cp $0 $OUTPUT_PATH
cp -r $model_dir $OUTPUT_PATH

which python
stage=0
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
  --num_train_epochs 1 \
  --checkpointing_steps 10 \
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
  --meta_file_path benchmark/captions/${motion_type}/crop.csv \
  --val_meta_file_path benchmark/captions/${motion_type}/crop.csv \
  --validating_steps 5000 \
  --resume_from_checkpoint latest \
  --proportion_empty_prompts $proportion_empty_prompts \
  --ckpt_path $ckpt_path \
  --max_train_steps 1000000 \
  --val_csv_path benchmark/captions/${motion_type}/val_image.csv \
  --stage $stage \
  --rank 64 \
  --lora_alpha 32 \
  --checkpoints_total_limit 100 \
  --dataloader_num_workers 16    \
  --use_hf_perturb \
  --hf_strength 0.1 \
  --hf_jitter 0.1 \
