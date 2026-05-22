import os
import torch
from typing import Optional
import argparse
import sys
sys.path.append(os.getcwd())

from diffusers import CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from diffusers.image_processor import VaeImageProcessor

from models.my_CogVideoI2V import CogVideoXTransformer3DModelIP
from models.my_pipeline_i2v import CogVideoXImageToVideoPipeline
from models.FlexiAct_processor import RefNetLoRAProcessor
from models.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX

from scripts.utils import get_gt_img, compute_prompt_embeddings


def get_args():
    parser = argparse.ArgumentParser(description="Inference arguments of FlexiAct.")
    parser.add_argument(
        "--pretrained_model_name_or_path", 
        type=str, 
        default="/Your/Path/To/CogVideoX-5b-I2V",
        help="The path of the CogVideoX-5b-I2V weights."
    )
    parser.add_argument(
        "--refadapter_ckpt_path", 
        type=str, 
        default="ckpts/refnetlora_step40000_model.pt",
        help="The path of the RefAdapter weights."
    )
    parser.add_argument(
        "--emb_ckpt_path", 
        type=str, 
        default="ckpts/FAE/motion_ckpts/rotation.pt",
        help="The path of the Frequency-aware Embedding weights."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="outputs",
        help="The path where the generated video will be saved."
    )
    parser.add_argument(
        "--gt_img_path", 
        type=str, 
        default="benchmark/target_images/human/game_girl_1.webp",
        help="The path of the target image."
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default='''A woman is performing a fitness exercise. She clasps her hands together, \
    placing them in front of her chest, and keeps her forearms horizontal. She stands \
    with her legs apart. She rotates her upper body, and her arms rotate along with \
    her upper body. Meanwhile, her hands remain clasped together, and her lower body \
    stays stationary.''',
        help="The prompt for video generation."
    )
    parser.add_argument(
        "--high_timesteps", 
        type=float, 
        default=0.8,
        help="The transition timestep for FAE, which can be turn off by setting it to None."
    )
    parser.add_argument(
        "--reweight_scale", 
        type=float, 
        default=None,
        help="The scale for the additional attention weight in FAE, which can be turn off by setting it to None."
    )
    return parser.parse_args()


def save_output(
    pipe,
    pipeline_args,
    caption=None,
    device=None,
    seed=42,
    output_dir=None,
):
    scheduler_args = {}
    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"
        scheduler_args["variance_type"] = variance_type
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(device)

    # run inference
    generator = torch.Generator(device=device).manual_seed(seed) if seed else None
    pt_images = pipe(**pipeline_args, generator=generator, output_type="pt").frames[0]
    pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])
    image_np = VaeImageProcessor.pt_to_numpy(pt_images) # float32 [0, 1]
    video = VaeImageProcessor.numpy_to_pil(image_np)

    caption = caption[:min(len(caption), 100)]
    caption = '_'.join(caption.split(' '))
    save_dir = os.path.join(output_dir, "videos")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"output_{caption}.mp4")
    output_video_path = export_to_video(video, filename, fps=8)
    del pipe

    return output_video_path


def generate_video(
    gt_img: torch.Tensor,
    prompt: str,
    emb_ckpt_path: str,
    pretrained_model_name_or_path: str,
    refadapter_ckpt_path: str = 'ckpts/refnetlora_step40000_model.pt',
    lora_weight: float = 0.7,
    output_path: str = "outputs",
    guidance_scale: float = 6.0,
    seed: int = 42,
    high_timesteps: Optional[float] = None,
    reweight_scale: float = 1.0,
):
    """
    Generates a video based on the given prompt and target image, and saves it to the specified path.

    Parameters:
    - gt_img (torch.Tensor): The target image.
    - prompt (str): The description of the video to be generated.
    - emb_ckpt_path (str): The path of the Frequency-aware Embedding weights.
    - pretrained_model_name_or_path (str): The path of the pre-trained model to be used.
    - refadapter_ckpt_path (str): The path of the RefAdapter weights.
    - lora_weight (float): The weight of the RefAdapter weights.
    - output_path (str): The path where the generated video will be saved.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - seed (int): The seed for reproducibility.
    - high_timesteps (float): The transition timestep for FAE, which can be turn off by setting it to None.
    - reweight_scale (float): The scale for the additional attention weight in FAE, which can be turn off by setting it to None.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    weight_dtype = torch.bfloat16
    device = torch.device("cuda")

    scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    transformer = CogVideoXTransformer3DModelIP.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        revision=None,
        variant=None,
    )
    transformer.to(device, dtype=weight_dtype)

    flexiact_procs = {}
    for name, _ in transformer.attn_processors.items():
        flexiact_procs[name] = RefNetLoRAProcessor(
            dim=3072, 
            rank=64, 
            network_alpha=32, 
            lora_weight=lora_weight,
            stage=0,
        )
    flexiact_procs = flexiact_procs.copy()
    transformer.set_attn_processor(flexiact_procs)

    transformer.load_state_dict(torch.load(refadapter_ckpt_path), strict=False)
    print(f"Loaded RefAdapter from {refadapter_ckpt_path}")
    transformer.load_state_dict(torch.load(emb_ckpt_path), strict=False)
    print(f"Loading Frequency-Aware Embedding from {emb_ckpt_path}")
    
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        pretrained_model_name_or_path,
        transformer=transformer,
        scheduler=scheduler,
        revision=None,
        variant=None,
        torch_dtype=weight_dtype,
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None
    )
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # 2. Enable CPU offload for the model.
    pipe.to(device=device, dtype=weight_dtype)
    vae.to(device=device, dtype=weight_dtype)
    vae.enable_slicing()
    vae.enable_tiling()
    pipe.vae = vae
    
    # Inference
    # 3 Process input
    prompts = ["", prompt]
    text_encoder.to(device=device)
    prompt_embeds = compute_prompt_embeddings(
        tokenizer,
        text_encoder,
        prompts,
        226,
        device,
        weight_dtype,
        requires_grad=False,
    ) # [2B, L, C]
    text_encoder.to(device='cpu')
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)

    # Sample a random timestep for each image
    b, t, c, h, w = 1, 13, 16, 480 // 8, 720 // 8
    noise = torch.randn(b, t, c, h, w, device=device)
    # Sample the first frame as concat image conditioning
    img_cat_latents = pipe.vae.encode(gt_img.permute(0, 2, 1, 3, 4)[:, :, :1].to(dtype=weight_dtype, device=device)).latent_dist.sample()
    img_cat_latents = img_cat_latents.permute(0, 2, 1, 3, 4) * pipe.vae.config.scaling_factor  # [B, 1, C, H, W]
    padding_shape = (b, t - 1, c, h, w)
    latent_padding = torch.zeros(padding_shape).to(device=device, dtype=weight_dtype)
    img_cat_latents = torch.cat([img_cat_latents, latent_padding], dim=1) # [B, F, C, H, W]

    # 4. Generate the video frames based on the prompt.
    pipeline_args = {
        "image": img_cat_latents,
        "guidance_scale": guidance_scale,
        "use_dynamic_cfg": False,
        "height": 480,
        "width": 720,
        "latents": noise.to(dtype=weight_dtype, device=device),
        "prompt_embeds": prompt_embeds,
        "num_frames": 49,
        "high_timesteps": high_timesteps,
        "reweight_scale": reweight_scale,
    }
    output_video_path = save_output(
        pipe=pipe,
        pipeline_args=pipeline_args,
        caption=prompt,
        device=device,
        seed=seed,
        output_dir=output_path,
    )
    print(f"Generated video saved to {output_video_path}")
    return output_video_path


if __name__ == "__main__":
    args = get_args()
    gt_img = get_gt_img(args.gt_img_path, 480, 720)
    generate_video(
        prompt=args.prompt,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        refadapter_ckpt_path=args.refadapter_ckpt_path,
        emb_ckpt_path=args.emb_ckpt_path,
        lora_weight=0.7,
        gt_img=gt_img,
        output_path=args.output_path,
        guidance_scale=6.0,
        seed=42,
        high_timesteps=args.high_timesteps,
        reweight_scale=args.reweight_scale,
    )
