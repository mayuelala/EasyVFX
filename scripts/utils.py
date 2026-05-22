from transformers import T5EncoderModel, T5Tokenizer
from typing import List, Optional, Union, Tuple
import argparse
import torch
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
import torchvision.transforms as TT
import numpy as np
import random
import os
from decord import VideoReader, cpu
from PIL import Image
from collections import OrderedDict
from diffusers import CogVideoXDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import export_to_video
from diffusers.training_utils import free_memory


####### Parse Args >>>>>>>
def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")

    parser.add_argument("--biased_loss_ratio", type=float, default=1.0, 
                            help="Biased loss ratio for FAE training.")
    parser.add_argument("--use_biased_loss", type=int, default=1, 
                            help="Whether to use biased loss for training.")
    parser.add_argument("--use_different_first_frame", action="store_true", default=False, 
                            help="Whether to use different first frame for FAE training.")
    parser.add_argument("--stage", type=int, default=None, 
                            help="Stage for FlexiAct training.")
    parser.add_argument("--val_csv_path", type=str, default=None,
                            help="Path to validation csv file.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True, 
                            help="Path to pretrained model or model identifier from huggingface.co/models.")
    
    parser.add_argument("--lora_weight", type=float, default=1.0, 
                            help="Lora weight for RefAdapter.")
    parser.add_argument("--lora_alpha", type=float, default=1.0, 
                            help="Lora alpha for RefAdapter.")
    parser.add_argument("--rank", type=int, default=4, 
                            help="Lora rank for RefAdapter.")
    parser.add_argument("--proportion_empty_prompts", type=float, default=0, 
                            help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).")
    parser.add_argument("--revision", type=str, default=None, required=False, 
                            help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type=str, default=None, 
                            help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")
    parser.add_argument("--cache_dir", type=str, default=None, 
                            help="The directory where the downloaded models and datasets will be stored.")
    parser.add_argument("--ckpt_path", type=str, default=None, 
                            help="Path to the checkpoint file.")

    parser.add_argument("--meta_file_path", type=str, default=None, 
                            help="The path to training meta data.")
    parser.add_argument("--val_meta_file_path", type=str, default=None, 
                            help="The path to validation meta data.")
    parser.add_argument("--instance_data_root", type=str, default=None, 
                            help="The training video folder.")
    parser.add_argument("--video_column", type=str, default="video", 
                            help="The column of the dataset containing videos.")
    parser.add_argument("--caption_column", type=str, default="text", 
                            help="The column of the dataset containing the instance prompt for each video. Or, the name of the file in `--instance_data_root` folder containing the line-separated instance prompts.")
    parser.add_argument("--id_token", type=str, default=None, 
                            help="Identifier token appended to the start of each prompt if provided.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, 
                            help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")

    parser.add_argument("--guidance_scale", type=float, default=6, 
                            help="The guidance scale to use while sampling validation videos.")
    parser.add_argument("--use_dynamic_cfg", action="store_true", default=False, 
                            help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.")
    parser.add_argument("--num_validation_videos", type=int, default=1, 
                            help="Number of validation videos to sample.")
    parser.add_argument("--emptytxt", action="store_true", default=False, 
                            help="Whether to use empty text for validation.")

    parser.add_argument("--seed", type=int, default=None, 
                            help="A seed for reproducible training.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
                            help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="cogvideox-i2v-lora", 
                            help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--height", type=int, default=480, 
                            help="All input videos are resized to this height.")
    parser.add_argument("--width", type=int, default=720, 
                            help="All input videos are resized to this width.")
    parser.add_argument("--video_reshape_mode", type=str, default="center", 
                            help="All input videos are reshaped to this mode. Choose between ['center', 'random', 'none']")
    parser.add_argument("--fps", type=int, default=8, 
                            help="All input videos will be used at this FPS.")
    parser.add_argument("--max_num_frames", type=int, default=49, 
                            help="All input videos will be truncated to these many frames.")
    parser.add_argument("--skip_frames_start", type=int, default=0, 
                            help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.")
    parser.add_argument("--skip_frames_end", type=int, default=0, 
                            help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.")
    parser.add_argument("--random_flip", action="store_true", 
                            help="whether to randomly flip videos horizontally")
    parser.add_argument("--train_batch_size", type=int, default=4, 
                            help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                            help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, 
                            help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.")
    parser.add_argument("--checkpointing_steps", type=int, default=500, 
                            help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument("--validating_steps", type=int, default=50, 
                            help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument("--checkpoints_total_limit", type=int, default=10, 
                            help=("Max number of checkpoints to store."))
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                            help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true", 
                            help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                            help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False, 
                            help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", 
                            help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, 
                            help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, 
                            help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0, 
                            help="Power factor of the polynomial scheduler.")
    parser.add_argument("--enable_slicing", action="store_true", default=False, 
                            help="Whether or not to use VAE slicing for saving memory.")
    parser.add_argument("--enable_tiling", action="store_true", default=False, 
                            help="Whether or not to use VAE tiling for saving memory.")
    parser.add_argument("--noised_image_dropout", type=float, default=0.05, 
                            help="Image condition dropout probability.")

    parser.add_argument("--optimizer", type=lambda s: s.lower(), default="adam", choices=["adam", "adamw", "prodigy"], 
                            help=("The optimizer type to use."))
    parser.add_argument("--use_8bit_adam", action="store_true", 
                            help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW")
    parser.add_argument("--adam_beta1", type=float, default=0.9, 
                            help="The beta1 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, 
                            help="The beta2 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--prodigy_beta3", type=float, default=None, 
                            help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.")
    parser.add_argument("--prodigy_decouple", action="store_true", 
                            help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, 
                            help="Weight decay to use for unet params")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, 
                            help="Epsilon value for the Adam optimizer and Prodigy optimizers.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, 
                            help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", 
                            help="Turn on Adam's bias correction.")
    parser.add_argument("--prodigy_safeguard_warmup", action="store_true", 
                            help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.")

    parser.add_argument("--tracker_name", type=str, default=None, 
                            help="Project tracker name")
    parser.add_argument("--push_to_hub", action="store_true", 
                            help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, 
                            help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, 
                            help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--logging_dir", type=str, default="logs", 
                            help="Directory where logs are stored.")
    parser.add_argument("--allow_tf32", action="store_true", 
                            help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default=None, 
                            help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--max_text_seq_length", type=int, default=226, 
                            help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).")
    parser.add_argument("--nccl_timeout", type=int, default=6000, 
                            help="NCCL backend timeout in seconds.")

    parser.add_argument("--use_hf_perturb", action="store_true",
                        help="Enable high-frequency perturbation on latent noise.")
    parser.add_argument("--hf_strength", type=float, default=0.1,
                        help="Strength of the high-frequency residual (e.g., 0.05~0.15).")
    parser.add_argument("--hf_jitter", type=float, default=0.1,
                        help="Random jitter on the HF strength per sample/frame (0.0~0.2).")

    return parser.parse_args()


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if text_input_ids is None:
        b = len(prompt)
    else:
        b = text_input_ids.shape[0]

    if tokenizer is not None and text_input_ids is None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(b * num_videos_per_prompt, seq_len, -1)
    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False, token_ids=None
):
    r"""
    Compute the prompt embeddings for the given prompt [str, token_ids].

    Parameters:
        tokenizer (`T5Tokenizer`):
            The T5 tokenizer to use.
        text_encoder (`T5EncoderModel`):
            The T5 text encoder to use.
        prompt (`str`):
            The prompt to compute the embeddings for.
        max_sequence_length (`int`):
            The maximum sequence length of the input text embeddings.
        device (`torch.device`):
            The device to use.
        dtype (`torch.dtype`):
            The dtype to use.
        requires_grad (`bool`, defaults to `False`):
            Whether to require gradients for the prompt embeddings.
        token_ids (`torch.Tensor`):
            The token ids to compute the embeddings for.
    Returns:
        prompt_embeds (`torch.Tensor`):
            The prompt embeddings.
    """
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
            text_input_ids=token_ids,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
                text_input_ids=token_ids,
            )
    return prompt_embeds


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin

def resize_wo_crop(arr, image_size):
    arr = resize(
        arr,
        size=[image_size[0], image_size[1]],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return arr


def resize_for_rectangle_crop(arr, height, width, reshape_mode="center"):
    r"""
    Resize the input array to the given height and width.
    """
    image_size = height, width
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    
    return arr


def get_gt_img(gt_img_path, height, width):
    r"""
    Get the target image. Returns a tensor of shape [B, 1, C, H, W].
    """
    if isinstance(gt_img_path, str):
        gt_img = Image.open(gt_img_path)
    gt_img = gt_img.convert("RGB")
    gt_img = np.array(gt_img, dtype=np.uint8)
    gt_img = torch.tensor(gt_img).permute(2, 0, 1).unsqueeze(0)
    gt_img = resize_for_rectangle_crop(gt_img, height, width, reshape_mode="center").unsqueeze(0).unsqueeze(0).float() / 127.5 - 1.0

    return gt_img


def get_gt_video(gt_video_path, height, width, max_num_frames):
    r"""
    Get the video from the given path. Returns a tensor of shape [B, T, C, H, W].
    """
    cpu_idx = random.randint(0, os.cpu_count() - 1)
    vr = VideoReader(gt_video_path, ctx=cpu(cpu_idx))
    frames = vr.get_batch(range(len(vr))).asnumpy()
    video = np.array(frames)
    frame = video.shape[0]
    if frame > max_num_frames:
        begin_idx = random.randint(0, frame - max_num_frames)
        end_idx = begin_idx + max_num_frames
        video = video[begin_idx:end_idx]
        frame = end_idx - begin_idx
    video = torch.from_numpy(video).permute(0, 3, 1, 2)
    video = resize_for_rectangle_crop(video, height, width, reshape_mode="center")
    video = video.permute(0, 2, 3, 1).numpy()
    video = (video.astype(np.float32) / 127.5) - 1.0
    video = torch.tensor(video).permute(0, 3, 1, 2).unsqueeze(0)

    return video


def save_trainable_state_dict_wo_ds(unwrapped_model, save_path, sd=None):
    r"""
    Save the trainable parameters.
    """
    os.makedirs(save_path, exist_ok=True)
    transformer_ipa_layers_to_save = {}
    for n, p in unwrapped_model.named_parameters():
        if p.requires_grad:
            transformer_ipa_layers_to_save[n] = p
    transformer_ipa_layers_to_save = sd if sd is not None else transformer_ipa_layers_to_save
    trainable_params = [p for p in transformer_ipa_layers_to_save.values() if p.requires_grad]
    print('Saving', sum([p.numel() for p in trainable_params]) / 1000000, 'M parameters')
    sd = unwrapped_model.state_dict()
    trainable_state_dict = {k: v for k, v in sd.items() if k in transformer_ipa_layers_to_save}
    torch.save(OrderedDict(trainable_state_dict), f"{save_path}/pytorch_model.pt")


def load_trainable_state_dict_wo_ds(unwrapped_model, load_path):
    r"""
    Load the trainable parameters.
    """
    if not os.path.exists(load_path):
        print(f"Checkpoint '{load_path}' does not exist. Starting a new training run.")
        return
    unwrapped_model.load_state_dict(torch.load(load_path), strict=False)
    print(f"Loaded {load_path}")


####### Evaluation >>>>>>>
def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
    global_step = 0,
    is_final_validation: bool = False,
    gt_img=None,
    caption=None,
    routing_state=None,
):
    scheduler_args = {}
    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(accelerator.device)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    if accelerator.is_main_process:
        for _ in range(args.num_validation_videos):
            # pt_images = pipe(**pipeline_args, generator=generator, output_type="pt").frames[0]
            def _extract_video_latents(latents: torch.Tensor) -> Optional[torch.Tensor]:
                if latents is None or latents.ndim != 5:
                    return None

                # expect T is relatively small (<= args.max_num_frames, usually <= 49)
                T_max = getattr(args, "max_num_frames", 49)

                # Case A: (B, T, C, H, W)
                if latents.shape[1] <= T_max:
                    return latents.contiguous()

                # Case B: (B, C, T, H, W)
                if latents.shape[2] <= T_max:
                    return latents.permute(0, 2, 1, 3, 4).contiguous()

                # otherwise unknown layout
                return None


            def _routing_callback(pipe, step_index: int, timestep: int, callback_kwargs: dict):
                """
                Diffusers callback_on_step_end style.
                Needs callback_on_step_end_tensor_inputs=["latents"] so callback_kwargs contains 'latents'.
                """
                if routing_state is None:
                    return callback_kwargs

                latents = callback_kwargs.get("latents", None)
                if latents is None:
                    return callback_kwargs

                with torch.no_grad():
                    video_latents = _extract_video_latents(latents)
                    if video_latents is None:
                        return callback_kwargs

                    # CFG duplicates batch to 2B; keep cond/uncond routing consistent
                    if video_latents.shape[0] % 2 == 0 and getattr(args, "guidance_scale", 1.0) > 1.0:
                        half = video_latents.shape[0] // 2
                        v_half = video_latents[:half]
                        routing_state.update_from_video(v_half)

                        # repeat alpha/e_t/e_app/e_mot to match 2B inside pipeline
                        routing_state.alpha = routing_state.alpha.repeat(2, 1)
                        if getattr(routing_state, "e_t", None) is not None:
                            routing_state.e_t = routing_state.e_t.repeat(2, 1)
                        if getattr(routing_state, "e_app", None) is not None:
                            routing_state.e_app = routing_state.e_app.repeat(2, 1)
                        if getattr(routing_state, "e_mot", None) is not None and routing_state.e_mot is not None:
                            routing_state.e_mot = routing_state.e_mot.repeat(2, 1)
                    else:
                        routing_state.update_from_video(video_latents)
                
                if accelerator.is_main_process and step_index % 20 == 0:
                    a = routing_state.alpha
                    if a is not None:
                        msg = f"[VAL][step {step_index}] alpha_mean={a.mean(dim=0).tolist()}"
                        if getattr(routing_state, "e_app", None) is not None:
                            msg += f" | fei_app_mean={routing_state.e_app.mean(dim=0).tolist()}"
                        if getattr(routing_state, "e_mot", None) is not None and routing_state.e_mot is not None:
                            msg += f" | fei_mot_mean={routing_state.e_mot.mean(dim=0).tolist()}"
                        print(msg)

                return callback_kwargs

            if routing_state is not None and pipeline_args.get("latents", None) is not None:
                with torch.no_grad():
                    v0 = _extract_video_latents(pipeline_args["latents"])
                    if v0 is not None:
                        routing_state.update_from_video(v0)

                        if getattr(args, "guidance_scale", 1.0) > 1.0:
                            routing_state.alpha = routing_state.alpha.repeat(2, 1)
                            if getattr(routing_state, "e_t", None) is not None:
                                routing_state.e_t = routing_state.e_t.repeat(2, 1)
                            if getattr(routing_state, "e_app", None) is not None:
                                routing_state.e_app = routing_state.e_app.repeat(2, 1)
                            if getattr(routing_state, "e_mot", None) is not None and routing_state.e_mot is not None:
                                routing_state.e_mot = routing_state.e_mot.repeat(2, 1)    
    

            pt_images = pipe(
                **pipeline_args,
                generator=generator,
                output_type="pt",
                callback_on_step_end=_routing_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            ).frames[0]




            pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])

            image_np = VaeImageProcessor.pt_to_numpy(pt_images) 
            if gt_img is not None:
                gt_img = gt_img[0].permute(0, 2, 3, 1)
                gt_imgs_np = (np.array(gt_img) + 1.0) / 2.0
                if gt_imgs_np.shape[0] != image_np.shape[0]:
                    gt_imgs_np = np.tile(gt_imgs_np, (image_np.shape[0], 1, 1, 1))
                image_np = np.concatenate([gt_imgs_np, image_np], axis=2)
            image_pil = VaeImageProcessor.numpy_to_pil(image_np)

            videos.append(image_pil)

    phase_name = "test" if is_final_validation else "validation"
    video_filenames = []
    for i, video in enumerate(videos):
        cap_len = min(len(caption), 100)
        caption = caption[:cap_len]
        save_dir = os.path.join(args.output_dir, "videos", f"step{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        idx = 0
        filename = os.path.join(save_dir, f"inf_{caption}_{idx}.mp4")
        while os.path.exists(filename):
            idx += 1
            filename = os.path.join(save_dir, f"inf_{caption}_{idx}.mp4")
        export_to_video(video, filename, fps=8)
        video_filenames.append(filename)

    del pipe
    free_memory()

    return videos


####### Get Optimizer >>>>>>>
def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        print(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and args.optimizer.lower() not in ["adam", "adamw"]:
        print(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            print(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer