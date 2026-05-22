import torch
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
import math
import shutil
from pathlib import Path
import pandas as pd
import random
from datetime import timedelta

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel

import diffusers
from diffusers import CogVideoXDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import check_min_version

from models.my_CogVideoI2V import CogVideoXTransformer3DModelIP
from models.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
from data.videovo_dataset import VideoInpaintingDataset, MyWebDataset
from models.my_pipeline_i2v import CogVideoXImageToVideoPipeline
from models.FlexiAct_processor import RefNetLoRAProcessor
import itertools

from scripts.utils import (
    get_args,
    compute_prompt_embeddings, 
    prepare_rotary_positional_embeddings,
    get_gt_img,
    load_trainable_state_dict_wo_ds,
    save_trainable_state_dict_wo_ds,
    log_validation,
    get_optimizer,
)

import torch
import torch.nn.functional as F

def add_highfreq_perturb(noisy_in: torch.Tensor, c_latent: int,
                         strength: float = 0.1, jitter: float = 0.1):
    if strength <= 0:
        return noisy_in

    x = noisy_in[:, :, :c_latent, :, :]
    shape = x.shape                      
    B, T, C, H, W = shape
    y = x.contiguous().view(B * T, C, H, W).float()
    k = torch.tensor([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]], device=y.device, dtype=y.dtype).view(1, 1, 3, 3)
    k = k.repeat(C, 1, 1, 1)             
    hi = F.conv2d(y, k, padding=1, groups=C)
    hi = hi.view(shape)                  
    std = hi.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    hi = (hi / std).to(noisy_in.dtype)
    if jitter > 0:
        j = 1.0 + torch.empty(B, T, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(-jitter, jitter)
        hi = hi * j
    noisy_in[:, :, :c_latent, :, :] = x + strength * hi
    return noisy_in

def phase_unitvec_loss(pred: torch.Tensor, target: torch.Tensor, fft_dims=(1, 3, 4)):
    pred_fp32 = pred.float()
    target_fp32 = target.float()
    P = torch.fft.fftn(pred_fp32, dim=fft_dims, norm="ortho")
    T = torch.fft.fftn(target_fp32, dim=fft_dims, norm="ortho")
    ph_p = torch.angle(P)
    ph_t = torch.angle(T)
    up = torch.stack([torch.cos(ph_p), torch.sin(ph_p)], dim=-1)
    ut = torch.stack([torch.cos(ph_t), torch.sin(ph_t)], dim=-1)
    b = pred.shape[0]
    return (up - ut).abs().reshape(b, -1).mean(dim=1)



os.environ['CURL_CA_BUNDLE'] = ''

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)


def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CogVideoXTransformer3DModelIP.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # We only train the additional adapter LoRA layers
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
            weight_dtype = torch.bfloat16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    flexiact_procs = {}
    train_lora_weight = 0 if args.stage == 0 else args.lora_weight
    for name, _ in transformer.attn_processors.items():
        flexiact_procs[name] = RefNetLoRAProcessor(
            dim=3072, 
            rank=args.rank, 
            network_alpha=args.lora_alpha, 
            lora_weight=train_lora_weight,
            stage=args.stage,
        )
    flexiact_procs = flexiact_procs.copy()
    transformer.set_attn_processor(flexiact_procs)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            if args.ckpt_path is not None:
                accelerator.print(
                    f"Resuming from checkpoint '{args.ckpt_path}"
                )
                load_trainable_state_dict_wo_ds(transformer, args.ckpt_path)

            else:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            load_trainable_state_dict_wo_ds(transformer, os.path.join(args.output_dir, path, "pytorch_model.pt"))

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    trainable_key = 'motion_inversion_tokens'if args.stage == 0 else 'lora'
    for n, param in transformer.named_parameters():
        param.requires_grad = True if trainable_key in n else False

    # Check the number of trainable parameters
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    if accelerator.is_main_process:
        print("====== Trainable Parameters ======")
        trainable_names = [n for n, p in transformer.named_parameters() if p.requires_grad]
        for name in trainable_names:
            print(name)
        print("================================")

    print(sum([p.numel() for p in trainable_params]) / 1000000, 'M parameters')
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    fae_params = []
    lora_params = []
    for n, p in transformer.named_parameters():
        if not p.requires_grad:
            continue
        if "motion_inversion_tokens" in n:
            fae_params.append(p.numel())
        elif "lora" in n:
            lora_params.append(p.numel())

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)
    transformer_parameters_with_lr = {"params": trainable_params, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]
    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    train_dataset = VideoInpaintingDataset(
        meta_file_path=args.meta_file_path,
        instance_data_root=args.instance_data_root,
        dataset_name=None,
        dataset_config_name=None,
        caption_column=args.caption_column,
        video_column=args.video_column,
        height=args.height,
        width=args.width,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        cache_dir=args.cache_dir,
        id_token=args.id_token,
    )

    validation_dataset = VideoInpaintingDataset(
        meta_file_path=args.val_meta_file_path,
        instance_data_root=args.instance_data_root,
        dataset_name=None,
        dataset_config_name=None,
        caption_column=args.caption_column,
        video_column=args.video_column,
        height=args.height,
        width=args.width,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        cache_dir=args.cache_dir,
        id_token=args.id_token,
        is_train=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=MyWebDataset(
            resolution = (args.height, args.width), 
            tokenizer=tokenizer,
            max_num_frames=args.max_num_frames,
            max_sequence_length=args.max_text_seq_length,
            proportion_empty_prompts=args.proportion_empty_prompts,
            is_train=True,
            emptytxt=args.emptytxt,
            use_different_first_frame=args.use_different_first_frame,
        ),
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=MyWebDataset(
            resolution = (args.height, args.width), 
            tokenizer=tokenizer,
            max_num_frames=args.max_num_frames,
            max_sequence_length=args.max_text_seq_length,
            proportion_empty_prompts=args.proportion_empty_prompts,
            is_train=False,
            emptytxt=args.emptytxt,
        ),
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    dataloader_len = len(train_dataloader)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(dataloader_len / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    num_update_steps_per_epoch = math.ceil(dataloader_len / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-i2v-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {dataloader_len}")
    logger.info(f"  Num batches each epoch = {dataloader_len}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if args.stage == 0:
            initial_global_step = 0
        else:
            if path is None:
                if args.ckpt_path is not None and os.path.exists(args.ckpt_path):
                    try:
                        initial_global_step = int(args.ckpt_path.split("/")[-2].split("-")[-1])
                    except:
                        initial_global_step = 0
                else:
                    args.resume_from_checkpoint = None
                    initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                global_step = int(path.split("-")[1])
                initial_global_step = global_step

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    global_step = initial_global_step

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    # if args.stage is not None or (args.stage is None and args.max_train_steps is not None):
    if args.stage is not None:
        train_dataloader = itertools.cycle(train_dataloader)

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]

            if "timing" in batch and accelerator.is_local_main_process:
                for k, v in batch["timing"].items():
                    try:

                        accelerator.log({f"time/{k}": float(v)}, step=global_step)
                    except Exception:
                        pass
            if "timing" in batch and accelerator.is_local_main_process and (global_step % 20 == 0):

                keys_to_show = [
                    "io/open_ms", "io/decode_ms", "io/fps_downsample_ms",
                    "data/resize_video_ms", "data/resize_image_ms",
                    "data/np2torch_ms", "data/normalize_ms",
                    "data/tokenize_ms", "data/stack_ms",
                ]
                pieces = [f"{k}={batch['timing'].get(k, 0):.1f}ms" for k in keys_to_show]
                print("[timing:data]", " | ".join(pieces))

            with accelerator.accumulate(models_to_accumulate):
                with torch.no_grad():
                    # Get input video and video embedding
                    pixel_values = batch["pixel_values"]
                    num_pix_frames = pixel_values.size(1)
                    # rand_idx = random.randint(0, num_pix_frames - 1) if args.stage is None else 0
                    rand_idx = 0
                    model_input = vae.encode(pixel_values.permute(0, 2, 1, 3, 4).to(dtype=weight_dtype, device=accelerator.device)).latent_dist.sample()
                    model_input = model_input.permute(0, 2, 1, 3, 4) * vae.config.scaling_factor  # [B, F, C=16, H, W]

                    # Encode the condition image [First frame for FAE / Random frame for RefAdapter]
                    image_values = batch["image_values"].permute(0, 2, 1, 3, 4)[:, :, rand_idx:rand_idx+1]
                    img_cat_latents = vae.encode(image_values.to(dtype=weight_dtype, device=accelerator.device)).latent_dist.sample()
                    img_cat_latents = img_cat_latents.permute(0, 2, 1, 3, 4) * vae.config.scaling_factor  # [B, 1, C, H, W]
                    if args.stage is None:
                        model_input[:, 0, :, :, :] = img_cat_latents[:, 0, :, :, :]

                    # Get text prompt embedding
                    prompts = batch["prompts"] if batch['input_ids'] is None else 'xx'
                    prompt_embeds = compute_prompt_embeddings(
                        tokenizer,
                        text_encoder,
                        prompts,
                        args.max_text_seq_length,
                        accelerator.device,
                        weight_dtype,
                        requires_grad=False,
                        token_ids=batch["input_ids"],
                    )

                    # Set image latent and prompt embedding
                    image_latents = model_input.to(dtype=weight_dtype)
                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                    b, t, c, h, w = image_latents.shape
                    LATENT_SHAPE= image_latents.shape

                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (b,), device=image_latents.device)
                    timesteps = timesteps.long()
                    noise = torch.randn_like(image_latents)
                    noisy_model_input = scheduler.add_noise(image_latents, noise, timesteps)
                    noisy_latents = noisy_model_input.clone().detach()
                    
                    # Concat and Pad the condition image embedding
                    padding_shape = (b, t - 1, c, h, w)
                    latent_padding = torch.zeros(padding_shape).to(device=accelerator.device, dtype=weight_dtype)
                    img_cat_latents = torch.cat([img_cat_latents, latent_padding], dim=1) # [B, F, C, H, W]
                    if random.random() < args.noised_image_dropout:
                        img_cat_latents = torch.zeros_like(img_cat_latents)
                    img_cat_latents = img_cat_latents.to(device=accelerator.device, dtype=weight_dtype)
                    noisy_model_input = torch.cat([noisy_model_input, img_cat_latents], dim=2)

                    use_hf = bool(getattr(args, "use_hf_perturb", False))
                    if use_hf:
                        hf_strength = float(getattr(args, "hf_strength", 0.1))  # 0.05~0.15
                        hf_jitter = float(getattr(args, "hf_jitter", 0.1))  # 0.0~0.2
                        if accelerator.is_main_process:
                            print(f"[HF] totalC={noisy_model_input.shape[2]}, condC={img_cat_latents.shape[2]}, "
                                  f"c_latent={c}  (expect totalC == condC + c_latent)")

                        noisy_model_input = add_highfreq_perturb(
                            noisy_model_input, c_latent=c, strength=hf_strength, jitter=hf_jitter
                        )
                        if accelerator.is_main_process:
                            print(f"[HF] totalC={noisy_model_input.shape[2]}, condC={img_cat_latents.shape[2]}, "
                                  f"c_latent={c}  (expect totalC == condC + c_latent)")

                # Prepare rotary embeds
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=args.height,
                        width=args.width,
                        num_frames=t,
                        vae_scale_factor_spatial=vae_scale_factor_spatial,
                        patch_size=model_config.patch_size,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )

                # Predict the noise residual
                model_output = transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
                model_pred = scheduler.get_velocity(model_output, noisy_latents, timesteps)

                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1 / (1 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = image_latents.clone().detach()
                loss = torch.mean((weights * (model_pred - target) ** 2).reshape(b, -1), dim=1)
                lambda_phase = float(getattr(args, "lambda_phase", 0.02))  # 建议先 0.01~0.05
                if lambda_phase > 0:
                    loss_phase = phase_unitvec_loss(model_pred, target, fft_dims=(1, 3, 4)) 
                    loss = loss + lambda_phase * loss_phase
                if args.stage == 0 and args.use_biased_loss:
                    rand_idx = random.randint(0, t - 2)
                    gt_delta = math.sqrt(2) * image_latents[:, rand_idx+1] - image_latents[:, rand_idx]
                    pred_delta = math.sqrt(2) * model_pred[:, rand_idx+1] - model_pred[:, rand_idx]
                    loss_biased = torch.mean(((gt_delta - pred_delta) ** 2).reshape(b, -1), dim=1)
                    loss = loss + loss_biased * args.biased_loss_ratio

                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Save checkpoint
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        unwrapped_model = unwrap_model(transformer)
                        save_trainable_state_dict_wo_ds(unwrapped_model, save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            # Evaluation
            eval_step = 3500 if args.stage == 0 else 1500
            need_validate = (global_step == 1) or (
                    (global_step % args.validating_steps == 0) and (global_step > eval_step)
            )

            if need_validate:
                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    if args.stage is not None:
                        print(f"Validating at global_step: {global_step}, Stage: {args.stage}")

                    if args.stage == 0:
                        for name, processor in transformer.attn_processors.items():
                            if isinstance(processor, RefNetLoRAProcessor):
                                processor.lora_weight = args.lora_weight

                    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        transformer=unwrap_model(transformer),
                        scheduler=scheduler,
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )
                    pipe.vae = vae

                    for val_step, batch in enumerate(validation_dataloader):

                        if "timing" in batch and accelerator.is_local_main_process:
                            for k, v in batch["timing"].items():
                                try:
                                    accelerator.log({f"time/{k}": float(v)}, step=global_step)
                                except Exception:
                                    pass
                        if "timing" in batch and accelerator.is_local_main_process and (global_step % 20 == 0):
                            keys_to_show = [
                                "io/open_ms", "io/decode_ms", "io/fps_downsample_ms",
                                "data/resize_video_ms", "data/resize_image_ms",
                                "data/np2torch_ms", "data/normalize_ms",
                                "data/tokenize_ms", "data/stack_ms",
                            ]
                            pieces = [f"{k}={batch['timing'].get(k, 0):.1f}ms" for k in keys_to_show]
                            print("[timing:data]", " | ".join(pieces))
                        if val_step < 1:
                            with torch.no_grad():
                                gt_img = batch["pixel_values"]  # B, F, C, H, W
                                num_pix_frames = gt_img.size(1)
                                rand_idx = random.randint(0, num_pix_frames - 1) if args.stage is None else 0
                                rand_idx = 0
                                gt_img[:, 0] = gt_img[:, rand_idx]
                                image_values = batch["image_values"].permute(0, 2, 1, 3, 4)[:, :, rand_idx:rand_idx + 1]
                                img_cat_latents = vae.encode(
                                    image_values.to(dtype=weight_dtype, device=accelerator.device)
                                ).latent_dist.sample()
                                img_cat_latents = img_cat_latents.permute(0, 2, 1, 3,
                                                                          4) * vae.config.scaling_factor  # [B, 1, C, H, W]

                                caption = batch["captions"][0]
                                prompts = batch["prompts"] if batch['input_ids'] is None else 'xx'
                                text_encoder.to(device=accelerator.device)
                                prompt_embeds = compute_prompt_embeddings(
                                    tokenizer,
                                    text_encoder,
                                    prompts,
                                    args.max_text_seq_length,
                                    accelerator.device,
                                    weight_dtype,
                                    requires_grad=False,
                                    token_ids=batch["input_ids"],
                                )  # [2B, L, C]
                                text_encoder.to(device='cpu')
                                prompt_embeds = prompt_embeds.to(dtype=weight_dtype)

                                if args.stage == 0:
                                    for name, processor in transformer.attn_processors.items():
                                        if isinstance(processor, RefNetLoRAProcessor):
                                            processor.lora_weight = args.lora_weight

                                _, t, c, h, w = LATENT_SHAPE
                                b = gt_img.shape[0]
                                noise = torch.randn(b, t, c, h, w, device=accelerator.device)

                                # Concat and Pad the condition image embedding
                                padding_shape = (b, t - 1, c, h, w)
                                latent_padding = torch.zeros(padding_shape).to(device=accelerator.device,
                                                                               dtype=weight_dtype)
                                img_cat_latents = torch.cat([img_cat_latents, latent_padding], dim=1)  # [B, F, C, H, W]

                                pipeline_args = {
                                    "image": img_cat_latents,
                                    "guidance_scale": args.guidance_scale,
                                    "use_dynamic_cfg": args.use_dynamic_cfg,
                                    "height": args.height,
                                    "width": args.width,
                                    "latents": noise.to(dtype=weight_dtype, device=accelerator.device),
                                    "prompt_embeds": prompt_embeds,
                                    "num_frames": args.max_num_frames,
                                }
                                validation_outputs = log_validation(
                                    pipe=pipe,
                                    args=args,
                                    accelerator=accelerator,
                                    pipeline_args=pipeline_args,
                                    global_step=global_step,
                                    gt_img=gt_img,
                                    caption=caption,
                                )
                        elif args.stage is not None and args.val_csv_path is not None:
                            val_csv_path = args.val_csv_path
                            val_df = pd.read_csv(val_csv_path)

                            for idx, row in val_df.iterrows():
                                img_path = row['path']
                                prompt = row['caption']
                                with torch.no_grad():
                                    gt_img = get_gt_img(img_path, args.height, args.width)  # B, T, C, H, W
                                    caption = prompt
                                    prompts = prompt
                                    prompts = ["", prompts]
                                    text_encoder.to(device=accelerator.device)
                                    prompt_embeds = compute_prompt_embeddings(
                                        tokenizer,
                                        text_encoder,
                                        prompts,
                                        args.max_text_seq_length,
                                        accelerator.device,
                                        weight_dtype,
                                        requires_grad=False,
                                    )  # [2B, L, C]
                                    text_encoder.to(device='cpu')
                                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)

                                    # Load RefAdapter
                                    for name, processor in transformer.attn_processors.items():
                                        if isinstance(processor, RefNetLoRAProcessor):
                                            processor.lora_weight = args.lora_weight

                                    # Sample a random timestep for each image
                                    b, t, c, h, w = 1, args.max_num_frames // 4 + 1, 16, args.height // 8, args.width // 8
                                    noise = torch.randn(b, t, c, h, w, device=accelerator.device)

                                    # Sample the first frame as concat image conditioning
                                    img_cat_latents = vae.encode(
                                        gt_img.permute(0, 2, 1, 3, 4)[:, :, :1].to(dtype=weight_dtype,
                                                                                   device=accelerator.device)
                                    ).latent_dist.sample()
                                    img_cat_latents = img_cat_latents.permute(0, 2, 1, 3,
                                                                              4) * vae.config.scaling_factor  # [B, 1, C, H, W]
                                    padding_shape = (b, t - 1, c, h, w)
                                    latent_padding = torch.zeros(padding_shape).to(device=accelerator.device,
                                                                                   dtype=weight_dtype)
                                    img_cat_latents = torch.cat([img_cat_latents, latent_padding],
                                                                dim=1)  # [B, F, C, H, W]

                                    pipeline_args = {
                                        "image": img_cat_latents,
                                        "guidance_scale": args.guidance_scale,
                                        "use_dynamic_cfg": args.use_dynamic_cfg,
                                        "height": args.height,
                                        "width": args.width,
                                        "latents": noise.to(dtype=weight_dtype, device=accelerator.device),
                                        "prompt_embeds": prompt_embeds,
                                        "num_frames": args.max_num_frames,
                                    }
                                    with torch.no_grad():
                                        validation_outputs = log_validation(
                                            pipe=pipe,
                                            args=args,
                                            accelerator=accelerator,
                                            pipeline_args=pipeline_args,
                                            global_step=global_step,
                                            gt_img=gt_img,
                                            caption=caption,
                                        )
                            break

                    text_encoder.to(device=accelerator.device)
                    if args.stage == 0:
                        for name, processor in transformer.attn_processors.items():
                            if isinstance(processor, RefNetLoRAProcessor):
                                processor.lora_weight = 0.0

                    del pipe
                accelerator.wait_for_everyone()


if __name__ == "__main__":
    args = get_args()
    main(args)
