"""
WAN 2.2 Image-to-Video Pipeline - Direct Implementation
Based on ComfyUI nodes and official WAN architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import T5Tokenizer, T5EncoderModel
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict
import math
from einops import rearrange, repeat


# ==================== MODEL ARCHITECTURES ====================

class TimestepEmbedding(nn.Module):
    """Timestep embedding for diffusion"""
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, timesteps):
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class Attention3D(nn.Module):
    """3D Attention for video generation"""
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class FeedForward(nn.Module):
    """MLP with GELU activation"""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class DiTBlock(nn.Module):
    """Diffusion Transformer Block with adaptive layer norm"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention3D(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, mlp_hidden_dim)
        
        # Adaptive layer norm parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        
    def forward(self, x, c):
        """
        x: input features [B, N, D]
        c: conditioning [B, D]
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention with modulation
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        
        # MLP with modulation
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        
        return x


def modulate(x, shift, scale):
    """Apply adaptive layer norm modulation"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class WanDiT(nn.Module):
    """
    Wan Diffusion Transformer for video generation
    Based on the MMDiT architecture with MoE (Mixture of Experts)
    """
    def __init__(
        self,
        in_channels=16,
        hidden_size=3072,
        depth=28,
        num_heads=24,
        patch_size=(1, 2, 2),  # Temporal, Height, Width
        num_experts=8,
        active_experts=2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, hidden_size))
        
        # Time embedding
        self.time_embed = TimestepEmbedding(hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False),
            nn.Linear(hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * in_channels)
        )
        
        # Conditioning projections
        self.context_embedder = nn.Linear(4096, hidden_size)  # For T5 embeddings
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize weights
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
    def unpatchify(self, x, t, h, w):
        """Convert patches back to video"""
        p_t, p_h, p_w = self.patch_size
        x = rearrange(x, 'b (t h w) (p1 p2 p3 c) -> b c (t p1) (h p2) (w p3)',
                     t=t, h=h, w=w, p1=p_t, p2=p_h, p3=p_w)
        return x
        
    def forward(self, x, timesteps, context, **kwargs):
        """
        x: [B, C, T, H, W] latent video
        timesteps: [B] timesteps
        context: [B, N, D] text conditioning from T5
        """
        B, C, T, H, W = x.shape
        
        # Patchify
        x = self.patch_embed(x)  # [B, hidden_size, T', H', W']
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        
        # Add positional embedding
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Time conditioning
        t_emb = self.time_embed(timesteps)
        t_emb = self.time_mlp(t_emb)
        
        # Project context
        c = self.context_embedder(context.mean(dim=1))  # Pool sequence
        c = c + t_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final layer
        x = self.final_layer(x)
        
        # Unpatchify
        t_out = T // self.patch_size[0]
        h_out = H // self.patch_size[1]
        w_out = W // self.patch_size[2]
        x = self.unpatchify(x, t_out, h_out, w_out)
        
        return x


class WanVAE(nn.Module):
    """Wan VAE for encoding/decoding video"""
    def __init__(self, in_channels=3, latent_channels=16, base_channels=128):
        super().__init__()
        self.latent_channels = latent_channels
        
        # Encoder - More robust architecture
        self.encoder = nn.Sequential(
            # First conv - no stride
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            # Downsample 1
            nn.Conv3d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(32, base_channels * 2),
            nn.SiLU(),
            # Downsample 2  
            nn.Conv3d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.GroupNorm(32, base_channels * 4),
            nn.SiLU(),
            # Final conv
            nn.Conv3d(base_channels * 4, latent_channels, 3, padding=1)
        )
        
        # Decoder - Fixed architecture to match encoder
        self.decoder = nn.Sequential(
            nn.Conv3d(latent_channels, base_channels * 4, 3, padding=1),
            nn.GroupNorm(32, base_channels * 4),
            nn.SiLU(),
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, base_channels * 2),
            nn.SiLU(),
            nn.ConvTranspose3d(base_channels * 2, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, in_channels, 3, padding=1)
        )
        
        self.scaling_factor = 0.18215
        
    def encode(self, x):
        """Encode video to latent"""
        h = self.encoder(x)
        return h * self.scaling_factor
        
    def decode(self, z):
        """Decode latent to video"""
        z = z / self.scaling_factor
        return self.decoder(z)


# ==================== COMFYUI NODE IMPLEMENTATIONS ====================

class UNETLoader:
    """Node 37, 56: Load UNET/Transformer models"""
    @staticmethod
    def load(unet_path: str, weight_dtype: str = "default"):
        print(f"Loading UNET from: {unet_path}")
        
        # Load state dict
        state_dict = load_file(unet_path)
        
        # Create model
        model = WanDiT(
            in_channels=16,
            hidden_size=3072,
            depth=28,
            num_heads=24
        )
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        
        return model


class VAELoader:
    """Node 39: Load VAE model"""
    @staticmethod
    def load(vae_path: str):
        print(f"Loading VAE from: {vae_path}")
        
        state_dict = load_file(vae_path)
        
        vae = WanVAE(
            in_channels=3,
            latent_channels=16,
            base_channels=128
        )
        
        vae.load_state_dict(state_dict, strict=False)
        
        return vae


class CLIPLoader:
    """Node 38: Load CLIP/T5 text encoder"""
    @staticmethod
    def load(clip_path: str, clip_type: str = "wan", device: str = "default"):
        print(f"Loading text encoder from: {clip_path}")
        # Load tokenizer and model from full local repo directory (umt5_xxl_fp16)
        local_dir = "models/text_encoders/umt5_xxl_fp16"
        tokenizer = T5Tokenizer.from_pretrained(local_dir, legacy=False)
        text_encoder = T5EncoderModel.from_pretrained(local_dir, torch_dtype=torch.float16)
        return tokenizer, text_encoder


class CLIPTextEncode:
    """Nodes 6, 7: Encode text prompts"""
    def __init__(self, tokenizer, text_encoder, device="cuda"):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.to(device)
        self.device = device
        
    @torch.no_grad()
    def encode(self, text: str):
        """Encode text to conditioning"""
        # Tokenize
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)
        
        # Encode
        encoder_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return encoder_output.last_hidden_state


class ModelSamplingSD3:
    """Nodes 54, 55: Apply model sampling shift"""
    def __init__(self, model, shift: float = 8.0):
        self.model = model
        self.shift = shift
        
        # Apply shift to noise schedule
        self._apply_shift()
        
    def _apply_shift(self):
        """Modify the model's noise schedule"""
        # The shift affects timestep scheduling
        # Store for use in sampling
        self.model.shift = self.shift
        
    def get_model(self):
        return self.model


class WanImageToVideo:
    """Node 63: Prepare image to video conditioning"""
    def __init__(self, vae, device="cuda"):
        self.vae = vae.to(device)
        self.device = device
        
    @torch.no_grad()
    def prepare(
        self,
        positive_cond,
        negative_cond,
        start_image_path: str,
        width: int,
        height: int,
        length: int,
        batch_size: int = 1
    ):
        """
        Prepare conditioning and latents for video generation
        Returns: (positive_cond, negative_cond, latent)
        """
        # Load and preprocess image
        image = self._load_image(start_image_path, width, height)
        
        # Add temporal dimension for 3D VAE
        # Shape: [B, C, H, W] -> [B, C, T, H, W]
        image_3d = image.unsqueeze(2)  # Add temporal dim
        
        try:
            # Encode image to latent
            image_latent = self.vae.encode(image_3d)
        except RuntimeError as e:
            if "Kernel size can't be greater than actual input size" in str(e):
                print(f"VAE dimension error, trying with smaller resolution...")
                # Fallback: resize to smaller dimensions
                smaller_size = min(512, width, height)
                image_small = self._load_image(start_image_path, smaller_size, smaller_size)
                image_3d_small = image_small.unsqueeze(2)
                image_latent = self.vae.encode(image_3d_small)
                # Upsample latent to match expected size
                image_latent = F.interpolate(
                    image_latent, 
                    size=(length, height//8, width//8), 
                    mode='trilinear', 
                    align_corners=False
                )
            else:
                raise e
        
        # Expand to video length
        video_latent = image_latent.repeat(1, 1, length, 1, 1)
        
        return positive_cond, negative_cond, video_latent
        
    def _load_image(self, image_path: str, width: int, height: int):
        """Load and preprocess image"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((width, height), Image.LANCZOS)
        
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - 0.5) * 2  # Normalize to [-1, 1]
        
        # Ensure dimensions are compatible with VAE
        # Pad to make dimensions divisible by 8 (common VAE requirement)
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            img_array = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)


class EulerSampler:
    """Euler sampling method"""
    @staticmethod
    def step(model, x, sigma, sigma_next, cond, uncond, cfg_scale):
        """Single Euler step"""
        # Predict noise with conditioning
        with torch.no_grad():
            noise_pred_cond = model(x, sigma, cond)
            noise_pred_uncond = model(x, sigma, uncond)
        
        # Classifier-free guidance
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Euler step
        dt = sigma_next - sigma
        x = x + noise_pred * dt
        
        return x


class KSamplerAdvanced:
    """Nodes 57, 58: Advanced K-Sampler for diffusion"""
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device
        
    def get_sigmas(self, steps: int, scheduler: str = "simple"):
        """Get noise schedule"""
        if scheduler == "simple":
            # Simple linear schedule with shift
            shift = getattr(self.model, 'shift', 8.0)
            sigmas = torch.linspace(1.0, 0.0, steps + 1)
            # Apply shift transformation
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
            return sigmas
        else:
            raise NotImplementedError(f"Scheduler {scheduler} not implemented")
    
    @torch.no_grad()
    def sample(
        self,
        positive_cond,
        negative_cond,
        latent,
        add_noise: str = "enable",
        seed: int = 0,
        steps: int = 20,
        cfg: float = 3.5,
        sampler_name: str = "euler",
        scheduler: str = "simple",
        start_at_step: int = 0,
        end_at_step: int = 10000,
        return_with_leftover_noise: str = "disable"
    ):
        """Perform diffusion sampling"""
        # Set seed
        if seed > 0:
            torch.manual_seed(seed)
            
        # Get noise schedule
        sigmas = self.get_sigmas(steps, scheduler).to(self.device)
        
        # Add noise if needed
        if add_noise == "enable":
            noise = torch.randn_like(latent)
            x = latent + noise * sigmas[start_at_step]
        else:
            x = latent
            
        # Clamp step range
        end_step = min(end_at_step, steps)
        
        # Sampling loop
        for i in range(start_at_step, end_step):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i < steps else torch.zeros(1)
            
            # Create timestep tensor
            t = torch.full((x.shape[0],), sigma.item() * 1000, device=self.device)
            
            if sampler_name == "euler":
                x = EulerSampler.step(
                    self.model, x, t, t * 0 + sigma_next.item() * 1000,
                    positive_cond, negative_cond, cfg
                )
            else:
                raise NotImplementedError(f"Sampler {sampler_name} not implemented")
                
        return x


class VAEDecode:
    """Node 8: Decode latents to video frames"""
    def __init__(self, vae, device="cuda"):
        self.vae = vae.to(device)
        self.device = device
        
    @torch.no_grad()
    def decode(self, latent):
        """Decode latent to video"""
        # Decode
        video = self.vae.decode(latent)
        
        # Denormalize
        video = (video + 1.0) / 2.0
        video = video.clamp(0, 1)
        
        return video


class LoraLoaderModelOnly:
    """Nodes 101, 102: Load and apply LoRA"""
    def __init__(self, model, lora_path: str, strength: float = 1.0):
        self.model = model
        self.strength = strength
        
        if lora_path:
            print(f"Loading LoRA from: {lora_path}")
            self.lora_state_dict = load_file(lora_path)
            self._apply_lora()
        
    def _apply_lora(self):
        """Apply LoRA weights to model"""
        for name, param in self.model.named_parameters():
            # Check for LoRA layers
            lora_up_key = f"{name}.lora_up.weight"
            lora_down_key = f"{name}.lora_down.weight"
            
            if lora_up_key in self.lora_state_dict and lora_down_key in self.lora_state_dict:
                lora_up = self.lora_state_dict[lora_up_key]
                lora_down = self.lora_state_dict[lora_down_key]
                
                # Compute LoRA update: W' = W + alpha * (up @ down)
                lora_weight = torch.mm(lora_up, lora_down) * self.strength
                param.data += lora_weight
                
        print(f"LoRA applied with strength {self.strength}")


# ==================== MAIN PIPELINE ====================

class WAN22Pipeline:
    """Complete WAN 2.2 Image-to-Video Pipeline"""
    
    def __init__(
        self,
        model_dir: str = "models",
        device: str = "cuda",
        dtype=torch.float16,
        enable_offload: bool = False
    ):
        self.device = device
        self.dtype = dtype
        self.enable_offload = enable_offload
        self.model_dir = model_dir
        
        print("="*60)
        print("Initializing WAN 2.2 Pipeline")
        print("="*60)
        
        self._load_all_models()
        
    def _load_all_models(self):
        """Load all required models - matching ComfyUI workflow"""
        
        # Node 38: CLIPLoader
        tokenizer, text_encoder = CLIPLoader.load(
            f"{self.model_dir}/text_encoders/umt5_xxl_fp16.safetensors",
            clip_type="wan"
        )
        self.clip_encode = CLIPTextEncode(tokenizer, text_encoder, self.device)
        
        # Node 39: VAELoader
        self.vae = VAELoader.load("models/vae/wan_2.1_vae.safetensors")
        
        # Node 37: UNETLoader (high noise)
        transformer_high = UNETLoader.load(
            "models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
        )
        # Node 56: UNETLoader (low noise)
        transformer_low = UNETLoader.load(
            "models/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
        )
        
        # Nodes 54, 55: ModelSamplingSD3
        self.transformer_high = ModelSamplingSD3(transformer_high, shift=8.0).get_model()
        self.transformer_low = ModelSamplingSD3(transformer_low, shift=8.0).get_model()
        
        # Node 63: WanImageToVideo
        self.image_to_video = WanImageToVideo(self.vae, self.device)
        
        # Node 8: VAEDecode
        self.vae_decode = VAEDecode(self.vae, self.device)
        
        print("All models loaded successfully!")
        
    def generate(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 640,
        height: int = 640,
        num_frames: int = 81,
        seed: int = 1043065861446219,
        steps: int = 20,
        cfg: float = 3.5
    ):
        """
        Generate video following the exact ComfyUI workflow
        """
        print("\n" + "="*60)
        print("Starting Video Generation")
        print("="*60)
        
        # Node 6: CLIPTextEncode (Positive)
        print("\n[1/5] Encoding positive prompt...")
        positive_cond = self.clip_encode.encode(prompt)
        
        # Node 7: CLIPTextEncode (Negative)
        print("[2/5] Encoding negative prompt...")
        negative_cond = self.clip_encode.encode(negative_prompt)
        
        # Node 63: WanImageToVideo
        print(f"[3/5] Preparing image to video (size: {width}x{height}, frames: {num_frames})...")
        positive_cond, negative_cond, video_latent = self.image_to_video.prepare(
            positive_cond, negative_cond, image_path, width, height, num_frames
        )
        
        # Node 57: KSamplerAdvanced (High Noise - Steps 0-10)
        print("[4/5] Running high noise sampling (steps 0-10)...")
        sampler_high = KSamplerAdvanced(self.transformer_high, self.device)
        latent_mid = sampler_high.sample(
            positive_cond, negative_cond, video_latent,
            add_noise="enable", seed=seed, steps=steps, cfg=cfg,
            sampler_name="euler", scheduler="simple",
            start_at_step=0, end_at_step=10,
            return_with_leftover_noise="enable"
        )
        
        # Node 58: KSamplerAdvanced (Low Noise - Steps 10-20)
        print("      Running low noise sampling (steps 10-20)...")
        sampler_low = KSamplerAdvanced(self.transformer_low, self.device)
        latent_final = sampler_low.sample(
            positive_cond, negative_cond, latent_mid,
            add_noise="disable", seed=0, steps=steps, cfg=cfg,
            sampler_name="euler", scheduler="simple",
            start_at_step=10, end_at_step=10000,
            return_with_leftover_noise="disable"
        )
        
        # Node 8: VAEDecode
        print("[5/5] Decoding latents to frames...")
        video_tensor = self.vae_decode.decode(latent_final)
        
        # Convert to numpy frames
        frames = self._tensor_to_frames(video_tensor)
        
        print(f"\n✓ Generation complete! Generated {len(frames)} frames")
        return frames
        
    def _tensor_to_frames(self, tensor):
        """Convert video tensor to list of numpy frames"""
        # tensor shape: [B, C, T, H, W]
        tensor = tensor.squeeze(0).cpu()  # Remove batch, move to CPU
        
        frames = []
        num_frames = tensor.shape[1]
        
        for i in range(num_frames):
            # Get frame [C, H, W]
            frame = tensor[:, i, :, :]
            # Convert to [H, W, C]
            frame = frame.permute(1, 2, 0).numpy()
            # Scale to 0-255
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            frames.append(frame)
            
        return frames
        
    def save_video(self, frames: List[np.ndarray], output_path: str, fps: int = 16):
        """Save frames as video"""
        if not frames:
            print("No frames to save!")
            return
            
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"✓ Video saved to: {output_path}")


# ==================== MEMORY MANAGEMENT ====================

class GroupOffloadManager:
    """
    Implements group offload strategy for memory management
    Moves groups of layers between GPU and CPU
    """
    def __init__(self, model, onload_device="cuda", offload_device="cpu"):
        self.model = model
        self.onload_device = onload_device
        self.offload_device = offload_device
        self.hooks = []
        
    def enable(self):
        """Enable group offload by adding hooks"""
        # Group layers into chunks
        layer_groups = self._create_layer_groups()
        
        for group_idx, group in enumerate(layer_groups):
            # Add pre-forward hook to move to GPU
            pre_hook = self._create_pre_hook(group)
            # Add post-forward hook to move to CPU
            post_hook = self._create_post_hook(group)
            
            for layer in group:
                handle1 = layer.register_forward_pre_hook(pre_hook)
                handle2 = layer.register_forward_hook(post_hook)
                self.hooks.extend([handle1, handle2])
                
        print(f"Group offload enabled: {len(layer_groups)} groups")
        
    def _create_layer_groups(self):
        """Group model layers for efficient offloading"""
        groups = []
        current_group = []
        
        # Group transformer blocks together
        if hasattr(self.model, 'blocks'):
            blocks_per_group = 4  # Offload 4 blocks at a time
            for i, block in enumerate(self.model.blocks):
                current_group.append(block)
                if len(current_group) == blocks_per_group:
                    groups.append(current_group)
                    current_group = []
            
            if current_group:
                groups.append(current_group)
                
        return groups
        
    def _create_pre_hook(self, group):
        """Create hook to move group to GPU before forward"""
        def hook(module, input):
            for layer in group:
                layer.to(self.onload_device)
        return hook
        
    def _create_post_hook(self, group):
        """Create hook to move group to CPU after forward"""
        def hook(module, input, output):
            for layer in group:
                layer.to(self.offload_device)
            torch.cuda.empty_cache()
        return hook
        
    def disable(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    
    # Example 1: Basic usage (fp16 models)
    print("\n" + "="*60)
    print("WAN 2.2 Image-to-Video Pipeline")
    print("="*60 + "\n")
    
    # Initialize pipeline
    pipeline = WAN22Pipeline(
        model_dir="models",  # Directory containing your models
        device="cuda",
        dtype=torch.float16,
        enable_offload=False  # Set True for group offload
    )
    
    # Prompts from ComfyUI workflow
    positive_prompt = """The white dragon warrior stands still, eyes full of determination 
    and strength. The camera slowly moves closer or circles around the warrior, highlighting 
    the powerful presence and heroic spirit of the character."""
    
    negative_prompt = """色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，
    静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，
    画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，
    静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"""
    
    # Generate video
    frames = pipeline.generate(
        image_path="input-18.jpg",
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        width=640,
        height=640,
        num_frames=81,
        seed=1043065861446219,
        steps=20,
        cfg=3.5
    )
    
    # Save video (Node 61: SaveVideo)
    pipeline.save_video(frames, "output_video.mp4", fps=16)
    
    print("\n" + "="*60)
    print("Pipeline execution completed successfully!")
    print("="*60)


# ==================== ADVANCED USAGE ====================

class WAN22PipelineAdvanced(WAN22Pipeline):
    """
    Advanced pipeline with LoRA support and group offload
    Matches the fp8_scaled + 4steps LoRA workflow
    """
    
    def __init__(self, model_dir: str = "models", device: str = "cuda", **kwargs):
        super().__init__(model_dir, device, **kwargs)
        
    def load_with_lora(
        self,
        lora_high_path: Optional[str] = None,
        lora_low_path: Optional[str] = None,
        lora_strength: float = 1.0
    ):
        """
        Load LoRA weights for 4-step inference
        Nodes 101, 102 from ComfyUI workflow
        """
        if lora_high_path:
            print(f"Loading high noise LoRA...")
            LoraLoaderModelOnly(self.transformer_high, lora_high_path, lora_strength)
            
        if lora_low_path:
            print(f"Loading low noise LoRA...")
            LoraLoaderModelOnly(self.transformer_low, lora_low_path, lora_strength)
    
    def enable_group_offload(self):
        """Enable group offload for all models"""
        print("Enabling group offload for memory management...")
        
        # Offload transformers
        self.offload_high = GroupOffloadManager(self.transformer_high)
        self.offload_high.enable()
        
        self.offload_low = GroupOffloadManager(self.transformer_low)
        self.offload_low.enable()
        
        # Offload VAE
        self.offload_vae = GroupOffloadManager(self.vae)
        self.offload_vae.enable()
        
    def generate_4step(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 129,
        seed: int = 1082484781986446,
        cfg: float = 1.0
    ):
        """
        Generate with 4-step LoRA (faster inference)
        Matches the fp8_scaled + 4steps LoRA workflow
        """
        print("\n" + "="*60)
        print("Starting 4-Step LoRA Video Generation")
        print("="*60)
        
        # Encode prompts
        print("\n[1/5] Encoding prompts...")
        positive_cond = self.clip_encode.encode(prompt)
        negative_cond = self.clip_encode.encode(negative_prompt)
        
        # Prepare video latents
        print(f"[2/5] Preparing video latents ({width}x{height}, {num_frames} frames)...")
        positive_cond, negative_cond, video_latent = self.image_to_video.prepare(
            positive_cond, negative_cond, image_path, width, height, num_frames
        )
        
        # High noise - 4 steps (0-2)
        print("[3/5] Running high noise sampling (2 steps)...")
        sampler_high = KSamplerAdvanced(self.transformer_high, self.device)
        latent_mid = sampler_high.sample(
            positive_cond, negative_cond, video_latent,
            add_noise="enable", seed=seed, steps=4, cfg=cfg,
            sampler_name="euler", scheduler="simple",
            start_at_step=0, end_at_step=2,
            return_with_leftover_noise="enable"
        )
        
        # Low noise - 4 steps (2-4)
        print("      Running low noise sampling (2 steps)...")
        sampler_low = KSamplerAdvanced(self.transformer_low, self.device)
        latent_final = sampler_low.sample(
            positive_cond, negative_cond, latent_mid,
            add_noise="disable", seed=0, steps=4, cfg=cfg,
            sampler_name="euler", scheduler="simple",
            start_at_step=2, end_at_step=4,
            return_with_leftover_noise="disable"
        )
        
        # Decode
        print("[4/5] Decoding latents...")
        video_tensor = self.vae_decode.decode(latent_final)
        
        # Optional: Apply RIFE interpolation (Node 110)
        # This would double the frame rate
        
        frames = self._tensor_to_frames(video_tensor)
        
        print(f"\n✓ 4-step generation complete! {len(frames)} frames")
        return frames


# ==================== HELPER FUNCTIONS ====================

def calculate_vram_usage():
    """Calculate VRAM usage for monitoring"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"\nVRAM Usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Total:     {total:.2f} GB")
        print(f"  Usage:     {(allocated/total)*100:.1f}%")


def run_workflow_example():
    """
    Complete example matching your ComfyUI workflow
    """
    
    # Choose which workflow to run
    use_lora = True  # Set False for standard fp16 workflow
    
    if use_lora:
        # fp8_scaled + 4steps LoRA workflow (faster, 4 steps)
        print("Running fp8_scaled + 4steps LoRA workflow...")
        
        pipeline = WAN22PipelineAdvanced(model_dir="models", device="cuda")
        
        # Load LoRAs (Nodes 101, 102)
        pipeline.load_with_lora(
            lora_high_path="models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
            lora_low_path="models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            lora_strength=1.0
        )
        
        # Enable offloading if needed
        # pipeline.enable_group_offload()
        
        # Example prompt from workflow
        prompt = """Soft morning daylight filters through a kitchen window, warm golden tones, 
        soft side lighting with gentle edge highlights. Medium close-up shot, eye-level perspective, 
        symmetrical composition. A young woman in a cream-white sweater stands at a wooden kitchen counter. 
        On the counter sits a light blue moka coffee pot with floral accents and a patterned white ceramic cup."""
        
        negative = """third hand, stuck hand, unnatural grip, fused fingers, distorted hands, 
        extra limbs, broken anatomy, robotic hand movement, cup floating, missing cup handle, 
        jittery transitions, unnatural motion, jerky animation, flickering, blurry, pixelated, 
        warped textures, poor depth of field, cartoonish style, low detail, overexposed lighting, 
        watermarks, logos, text overlays."""
        
        frames = pipeline.generate_4step(
            image_path="blue_coffeemaker_00011_.png",
            prompt=prompt,
            negative_prompt=negative,
            width=1024,
            height=1024,
            num_frames=129,
            seed=1082484781986446,
            cfg=1.0
        )
        
        # Save with higher fps for 4-step (Node 94: CreateVideo)
        pipeline.save_video(frames, "output_4step.mp4", fps=32)
        
    else:
        # Standard fp16 workflow (20 steps)
        print("Running standard fp16 workflow...")
        
        pipeline = WAN22Pipeline(model_dir="models", device="cuda")
        
        prompt = """The white dragon warrior stands still, eyes full of determination 
        and strength. The camera slowly moves closer or circles around the warrior, 
        highlighting the powerful presence and heroic spirit of the character."""
        
        negative = """色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，
        静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指"""
        
        frames = pipeline.generate(
            image_path="input-18.jpg",
            prompt=prompt,
            negative_prompt=negative,
            width=640,
            height=640,
            num_frames=81,
            seed=1043065861446219,
            steps=20,
            cfg=3.5
        )
        
        pipeline.save_video(frames, "output_standard.mp4", fps=16)
    
    # Show VRAM usage
    calculate_vram_usage()


if __name__ == "__main__":
    # Run the complete workflow
    run_workflow_example()
            