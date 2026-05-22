export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200

# Please set your CogVideoX-5b-I2V path
export MODEL_PATH="Your/CogVideoX-5b-I2V/Path"
# Set your input image path. This is an example from https://huggingface.co/datasets/shiyi0408/FlexiAct
export INPUT_IMAGE="target_images/path"

which python

CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
    --pretrained_model_name_or_path $MODEL_PATH \
    --refadapter_ckpt_path ckpts/RefAdapter_model_path \
    --emb_ckpt_path ckpts/FAE/motion_ckpts/path \
    --output_path outputs \
    --gt_img_path $INPUT_IMAGE \
    --prompt "" \
    --reweight_scale 1.0