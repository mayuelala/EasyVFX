import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyEnergyIndicator(nn.Module):
    def __init__(self, num_bands: int = 3):
        super().__init__()
        assert num_bands >= 2, "num_bands must be >= 2"
        self.num_bands = num_bands
        self.scales = [2 ** k for k in range(num_bands - 1)]

    @staticmethod
    def _gaussian_kernel2d(kernel_size: int, sigma: float, channels: int, device, dtype):
        x = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
        x_grid = x.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy = torch.stack([x_grid, y_grid], dim=-1)

        var = sigma * sigma
        kernel = (1.0 / (2.0 * math.pi * var)) * torch.exp(-torch.sum(xy * xy, dim=-1) / (2.0 * var))
        kernel = kernel / kernel.sum()

        return kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    def _blur(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0:
            return x
        k = int(2 * 4 * sigma + 1)
        k = k | 1 
        pad = k // 2

        B, C, H, W = x.shape
        weight = self._gaussian_kernel2d(k, sigma, C, x.device, x.dtype)
        return F.conv2d(x, weight, padding=pad, groups=C)

    def forward(self, z_t: torch.Tensor):
        """
        z_t: (B, C, H, W)
        """
        assert z_t.ndim == 4, f"Expected (B,C,H,W), got {z_t.shape}"
        B, C, H, W = z_t.shape

        kappa = min(H, W) / 128.0
        sigmas = [kappa * s for s in self.scales]  

        blurs = [self._blur(z_t, s) for s in sigmas]

        band_components = [blurs[-1]]
        for i in range(len(blurs) - 2, -1, -1):
            band_components.append(blurs[i] - blurs[i + 1])

        band_components.append(z_t - blurs[0])
        if len(band_components) > self.num_bands:
            band_components = band_components[: self.num_bands]
        elif len(band_components) < self.num_bands:
            pad_n = self.num_bands - len(band_components)
            band_components += [torch.zeros_like(z_t) for _ in range(pad_n)]

        energies = [torch.sum(b * b, dim=(1, 2, 3)) for b in band_components]  
        energy_vec = torch.stack(energies, dim=1)  

        e_t = energy_vec / (energy_vec.sum(dim=1, keepdim=True) + 1e-8)
        return e_t, band_components

class SoftFrequencyRouter(nn.Module):
    def __init__(self, in_dim: int, num_experts: int, tau: float = 0.7, hidden: int = 64):
        super().__init__()
        self.in_dim = in_dim
        self.num_experts = num_experts
        self.tau = tau

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(e)
        alpha = F.softmax(logits / max(self.tau, 1e-6), dim=-1)
        return alpha

class FeRARoutingState(nn.Module):
    def __init__(
        self,
        num_bands: int = 3,
        num_experts: int = 3,
        router_tau: float = 0.7,
        router_hidden: int = 64,
        detach_alpha: bool = False,
        use_dual_component: bool = True,  
    ):
        super().__init__()
        self.num_bands = num_bands
        self.use_dual_component = use_dual_component

        self.fei = FrequencyEnergyIndicator(num_bands=num_bands)

        in_dim = (2 * num_bands) if use_dual_component else num_bands
        self.router = SoftFrequencyRouter(in_dim=in_dim, num_experts=num_experts, tau=router_tau, hidden=router_hidden)

        self.detach_alpha = detach_alpha

        self.alpha = None
        self.e_t = None
        self.e_app = None
        self.e_mot = None

    def clear(self):
        self.alpha = None
        self.e_t = None
        self.e_app = None
        self.e_mot = None


    def update_from_video(self, noisy_video: torch.Tensor) -> torch.Tensor:
        assert noisy_video.dim() == 5, f"Expected [B,T,C,H,W], got {tuple(noisy_video.shape)}"
        with torch.no_grad():
            appearance_proxy = noisy_video.mean(dim=1)
            e_app, _ = self.fei(appearance_proxy)

            if self.use_dual_component:
                diff = noisy_video[:, 1:] - noisy_video[:, :-1]
                motion_t = (diff * diff).mean(dim=(2,3,4))  # [B, T-1]
               
                motion_proxy = (diff * diff).mean(dim=1)
                
                motion_proxy = torch.log1p(motion_proxy)
                e_mot, _ = self.fei(motion_proxy)

                e_joint = torch.cat([e_app, e_mot], dim=-1)
            else:
                e_mot = None
                e_joint = e_app


        e_joint = e_joint.detach()
        router_dtype = next(self.router.parameters()).dtype
        e_joint = e_joint.to(dtype=router_dtype)
        alpha = self.router(e_joint)

        if self.detach_alpha:
            alpha = alpha.detach()

        self.alpha = alpha
        self.e_t = e_joint
        self.e_app = e_app
        self.e_mot = e_mot
        return alpha

    def update_from_latent(self, latent2d: torch.Tensor) -> torch.Tensor:

        e_app, _ = self.fei(latent2d)

        if self.use_dual_component:
            e_mot = torch.zeros_like(e_app)
            e_joint = torch.cat([e_app, e_mot], dim=-1)
        else:
            e_mot = None
            e_joint = e_app

        alpha = self.router(e_joint)
        if self.detach_alpha:
            alpha = alpha.detach()

        self.alpha = alpha
        self.e_t = e_joint
        self.e_app = e_app
        self.e_mot = e_mot
        return alpha


class MultiExpertLoRALinearLayer(nn.Module):
    """
    Implements: sum_k alpha_k * LoRA_k(x)
    where each LoRA_k is a low-rank (down->up) without bias.
    """
    def __init__(self, in_features: int, out_features: int, total_rank: int, num_experts: int = 3, network_alpha=None,
                 device=None, dtype=None):
        super().__init__()
        assert num_experts >= 1
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.network_alpha = network_alpha
        self.total_rank = total_rank

        
        assert total_rank >= num_experts,\
            f"total_rank ({total_rank}) must >= num_experts ({num_experts}) to keep params constant"

        base = total_rank // num_experts
        rem  = total_rank % num_experts
        ranks = [base + (1 if k < rem else 0) for k in range(num_experts)]
        assert sum(ranks) == total_rank
        self.rank_per_expert = ranks


        self.down = nn.ModuleList([
            nn.Linear(in_features, ranks[k], bias=False, device=device, dtype=dtype)
            for k in range(num_experts)
        ])
        self.up = nn.ModuleList([
            nn.Linear(ranks[k], out_features, bias=False, device=device, dtype=dtype)
            for k in range(num_experts)
        ])


        for k in range(num_experts):
            nn.init.normal_(self.down[k].weight, std=1 / max(self.rank_per_expert[k], 1))
            nn.init.zeros_(self.up[k].weight)

    def forward(self, hidden_states: torch.Tensor, alpha: Optional[torch.Tensor]) -> torch.Tensor:
        """
        hidden_states: [B, N, D]
        alpha: [B, K] or None
        returns delta: [B, N, out_features]
        """
        if alpha is None:
            return torch.zeros(hidden_states.shape[0],hidden_states.shape[1],self.out_features,device=hidden_states.device,dtype=hidden_states.dtype,)

        B, N, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        dtype = self.down[0].weight.dtype

        deltas = []
        hs = hidden_states.to(dtype)
        for k in range(self.num_experts):
            d = self.up[k](self.down[k](hs))  
            if self.network_alpha is not None:
                d = d * (self.network_alpha / max(self.rank_per_expert[k], 1))

            deltas.append(d)
        deltas = torch.stack(deltas, dim=1)

        w = alpha[:, :, None, None].to(deltas.dtype)
        out = (w * deltas).sum(dim=1) 
        return out.to(orig_dtype)

class FeRARefNetLoRAProcessor(nn.Module):
    """
    Drop-in replacement for RefNetLoRAProcessor.
    You MUST pass a shared FeRARoutingState for stage=None (RefAdapter).
    """

    def __init__(
        self,
        dim: int,
        rank: int = 4,
        network_alpha=None,
        lora_weight: float = 1.0,
        stage=None,

        num_motion_tokens: int = 1024,
        is_train: bool = True,

        reweight_scale=None,
        vid2embed: int = 1,
        embed2vid: int = 0,
        attn_map_save_path: Optional[str] = None,
        cur_layer: int = 0,
        cur_step: int = 0,
        save_step=(10,),
        save_layer=(40,),

        routing_state: Optional[FeRARoutingState] = None,
        num_experts: int = 3,     
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.lora_weight = float(lora_weight)
        self.stage = stage
        self.is_train = is_train
        self.attention_mask = None

        # Attn reweight
        self.reweight_scale = reweight_scale
        self.vid2embed = vid2embed
        self.embed2vid = embed2vid

        # Visualization
        self.attn_map_save_path = attn_map_save_path
        self.cur_layer = cur_layer
        self.cur_step = cur_step
        self.save_step = list(save_step)
        self.save_layer = list(save_layer)

        self.routing_state = routing_state
        self.num_experts = num_experts
        self.lora_q = MultiExpertLoRALinearLayer(dim, dim, total_rank=rank, num_experts=num_experts, network_alpha=network_alpha)
        self.lora_k = MultiExpertLoRALinearLayer(dim, dim, total_rank=rank, num_experts=num_experts, network_alpha=network_alpha)
        self.lora_v = MultiExpertLoRALinearLayer(dim, dim, total_rank=rank, num_experts=num_experts, network_alpha=network_alpha)
        self.lora_proj = MultiExpertLoRALinearLayer(dim, dim, total_rank=rank, num_experts=num_experts, network_alpha=network_alpha)

        # Stage 2 (FAE)
        if stage is not None:
            self.num_motion_tokens = num_motion_tokens
            self.motion_inversion_tokens = nn.Parameter(torch.zeros(1, num_motion_tokens, dim))
            nn.init.zeros_(self.motion_inversion_tokens)
            print(
                f"[FAE-INIT] motion_inversion_tokens shape={self.motion_inversion_tokens.shape}, "
                f"num_tokens={self.num_motion_tokens}, emb_dim={dim}, total_params={self.motion_inversion_tokens.numel()}"
            )

    def save_attn_map(self, q_vis, k_vis, v_vis, save_step=(10,), save_layer=(40,)):
        attn_map = torch.zeros_like(v_vis)
        if not os.path.exists(os.path.join(self.attn_map_save_path, "status.txt")):
            with open(os.path.join(self.attn_map_save_path, "status.txt"), "w") as f:
                f.write("0")
                cur_status = 0
        else:
            with open(os.path.join(self.attn_map_save_path, "status.txt"), "r") as f:
                cur_status = int(f.read()) + 1
            with open(os.path.join(self.attn_map_save_path, "status.txt"), "w") as f:
                f.write(f"{cur_status}")

        cur_step = cur_status // 42
        cur_layer = cur_status % 42

        if cur_step in save_step and cur_layer in save_layer:
            print(f"save attn map at step {cur_step} layer {cur_layer}")
            for i in range(q_vis.shape[1]):
                q_mini = q_vis[:, i : i + 1]
                k_mini = k_vis[:, i : i + 1]
                attn_map_mini = F.scaled_dot_product_attention(
                    q_mini, k_mini, v_vis, attn_mask=None, dropout_p=0.0, is_causal=False
                )
                attn_map += attn_map_mini
            save_path = os.path.join(self.attn_map_save_path, f"{cur_step}_{cur_layer}.pt")
            torch.save(attn_map, save_path)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        text_seq_length = encoder_hidden_states.size(1)

        if self.stage is not None:
            cat_tokens = self.motion_inversion_tokens.repeat(encoder_hidden_states.size(0), 1, 1)
            encoder_hidden_states = torch.cat([encoder_hidden_states, cat_tokens], dim=1)
            encoder_h_seq_length = encoder_hidden_states.size(1)
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        else:
            encoder_h_seq_length = encoder_hidden_states.size(1)
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        alpha = None
        if self.stage is None:
            if self.routing_state is not None:
                alpha = self.routing_state.alpha 

        query = attn.to_q(hidden_states)
        query = query + self.lora_q(hidden_states, alpha) * self.lora_weight

        key = attn.to_k(hidden_states)
        key = key + self.lora_k(hidden_states, alpha) * self.lora_weight

        value = attn.to_v(hidden_states)
        value = value + self.lora_v(hidden_states, alpha) * self.lora_weight

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, encoder_h_seq_length:] = apply_rotary_emb(query[:, :, encoder_h_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, encoder_h_seq_length:] = apply_rotary_emb(key[:, :, encoder_h_seq_length:], image_rotary_emb)

        if self.reweight_scale is not None:
            q_len = query.shape[-2]
            attention_mask = torch.zeros((q_len, q_len), device=query.device, dtype=query.dtype)
            if self.vid2embed:
                attention_mask[encoder_h_seq_length:, text_seq_length:encoder_h_seq_length] += self.reweight_scale
            if self.embed2vid:
                attention_mask[text_seq_length:encoder_h_seq_length, encoder_h_seq_length:] += self.reweight_scale

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if self.attn_map_save_path is not None:
            q_vis = query[1:2].clone().to("cuda")
            k_vis = key[1:2].clone().to("cuda")
            v_vis = torch.eye(q_vis.shape[-2]).to("cuda", dtype=q_vis.dtype).unsqueeze(0).unsqueeze(0)
            q_vis.requires_grad = False
            k_vis.requires_grad = False
            v_vis.requires_grad = False
            self.save_attn_map(q_vis, k_vis, v_vis, save_step=self.save_step, save_layer=self.save_layer)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + self.lora_proj(hidden_states, alpha) * self.lora_weight
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [encoder_h_seq_length, hidden_states.size(1) - encoder_h_seq_length], dim=1
        )
        encoder_hidden_states = encoder_hidden_states[:, :text_seq_length, :]
        return hidden_states, encoder_hidden_states
