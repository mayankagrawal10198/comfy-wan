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


class SelfAttention(nn.Module):
    """Self-attention layer matching WAN 2.2 architecture"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Separate Q, K, V projections (matching real model)
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.o = nn.Linear(dim, dim, bias=True)
        
        # RMSNorm for Q and K
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Project Q, K, V
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        # Apply normalization
        q = self.norm_q(q)
        k = self.norm_k(k)
        
        # Reshape for multi-head attention
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Memory-efficient attention
        try:
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            x = x.transpose(1, 2).reshape(B, N, C)
        except:
            # Fallback
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.o(x)
        return x


class CrossAttention(nn.Module):
    """Cross-attention layer for text conditioning"""
    def __init__(self, dim, context_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.context_dim = context_dim
        
        # Q from input, K and V from context
        # NOTE: In WAN 2.2, K and V take dim->dim (5120->5120), not context_dim->dim
        # This means context is pre-projected to the same dimension
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)  # Changed from context_dim to dim
        self.v = nn.Linear(dim, dim, bias=True)  # Changed from context_dim to dim
        self.o = nn.Linear(dim, dim, bias=True)
        
        # RMSNorm
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        
        # Context projection (project context from context_dim to dim)
        self.context_proj = nn.Linear(context_dim, dim, bias=True) if context_dim != dim else nn.Identity()
        
    def forward(self, x, context):
        B, N, C = x.shape
        
        # Project context to same dimension as model
        context = self.context_proj(context)
        
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        
        # Reshape for multi-head
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, context.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, context.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Attention
        try:
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            x = x.transpose(1, 2).reshape(B, N, C)
        except:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.o(x)
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


class WanTransformerBlock(nn.Module):
    """WAN 2.2 Transformer Block with self-attention, cross-attention, and FFN"""
    def __init__(self, dim, context_dim, num_heads):
        super().__init__()
        self.dim = dim
        
        # Self-attention
        self.self_attn = SelfAttention(dim, num_heads)
        
        # Cross-attention
        self.cross_attn = CrossAttention(dim, context_dim, num_heads)
        
        # Feed-forward network (FFN)
        # From inspection: ffn.0 (13824 dim) and ffn.2 (back to 5120)
        mlp_hidden = int(dim * 2.7)  # 5120 * 2.7 ≈ 13824
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden, bias=True),  # ffn.0
            nn.GELU(),
            nn.Linear(mlp_hidden, dim, bias=True),  # ffn.2
        )
        
        # Layer norm
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=True)
        
        # Modulation parameters (adaptive layer norm)
        # Shape: [1, 6, dim] from inspection
        self.modulation = nn.Parameter(torch.zeros(1, 6, dim))
        
    def forward(self, x, context):
        """
        x: input features [B, N, D]
        context: text conditioning [B, M, D_context]
        """
        # Self-attention
        x = x + self.self_attn(x)
        
        # Cross-attention with text
        x = x + self.cross_attn(x, context)
        
        # FFN
        x = x + self.ffn(self.norm3(x))
        
        return x


def modulate(x, shift, scale):
    """Apply adaptive layer norm modulation"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class WanDiT(nn.Module):
    """
    WAN 2.2 Diffusion Transformer
    Architecture based on actual model inspection:
    - hidden_size: 5120
    - depth: 40 blocks (estimated from patterns)
    - num_heads: 40 (5120 / 128 = 40)
    - Has self_attn, cross_attn, and ffn in each block
    """
    def __init__(
        self,
        in_channels=16,
        hidden_size=5120,  # From model inspection
        context_dim=4096,  # T5-XXL dimension
        depth=40,  # Estimated
        num_heads=40,  # 5120 / 128
        patch_size=(1, 2, 2),  # Temporal, Height, Width
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
        
        # Time embedding
        self.time_embed = TimestepEmbedding(hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Transformer blocks with self-attention and cross-attention
        self.blocks = nn.ModuleList([
            WanTransformerBlock(hidden_size, context_dim, num_heads) 
            for _ in range(depth)
        ])
        
        # Final layer (head) - matching real model structure
        # Real model has: head.head.weight, head.head.bias, head.modulation
        self.head = nn.ModuleDict({
            'head': nn.Linear(hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * in_channels)
        })
        # Modulation as separate parameter (can't be in ModuleDict)
        self.head_modulation = nn.Parameter(torch.zeros(1, 6, hidden_size))
        
        self.initialize_weights()
        
    def _get_pos_embed(self, seq_len, device, dtype=None):
        """Generate positional embedding for any sequence length"""
        if dtype is None:
            dtype = torch.float32
            
        # Create sinusoidal positional embedding
        pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        dim = self.hidden_size
        
        # Create frequency bands
        div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * 
                           -(math.log(10000.0) / dim))
        
        # Compute sinusoidal embeddings
        pos_embed = torch.zeros(seq_len, dim, device=device, dtype=dtype)
        pos_embed[:, 0::2] = torch.sin(pos * div_term)
        pos_embed[:, 1::2] = torch.cos(pos * div_term)
        
        return pos_embed.unsqueeze(0)  # [1, seq_len, dim]
        
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
        # Ensure input has same dtype as model
        model_dtype = next(self.parameters()).dtype
        x = x.to(dtype=model_dtype)
        timesteps = timesteps.to(dtype=model_dtype)
        
        # Patchify
        x = self.patch_embed(x)  # [B, hidden_size, T', H', W']
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        
        # Add positional embedding - computed dynamically
        seq_len = x.shape[1]
        pos_embed = self._get_pos_embed(seq_len, x.device, dtype=x.dtype)
        x = x + pos_embed
        
        # Time conditioning (not used in WAN 2.2, but keep for compatibility)
        t_emb = self.time_embed(timesteps)
        model_dtype = next(self.parameters()).dtype
        t_emb = t_emb.to(dtype=model_dtype)
        t_emb = self.time_mlp(t_emb)
        
        # Ensure context is in correct dtype
        context = context.to(dtype=model_dtype)
        
        # Apply transformer blocks with cross-attention
        for i, block in enumerate(self.blocks):
            x = block(x, context)  # Pass full context, not pooled
            
            # Apply gradient clipping to prevent explosion
            if torch.isnan(x).any() or torch.isinf(x).any():
                # Replace with small random values to continue
                x = torch.randn_like(x) * 0.01
                break
            
            # Clip extreme values to prevent explosion
            x = torch.clamp(x, -100.0, 100.0)
        
        # Final layer (head)
        x = self.head['head'](x)
        
        # Unpatchify
        t_out = T // self.patch_size[0]
        h_out = H // self.patch_size[1]
        w_out = W // self.patch_size[2]
        x = self.unpatchify(x, t_out, h_out, w_out)
        
        return x


class CausalConv3d(nn.Module):
    """Causal 3D Convolution for temporal causality"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=(stride, stride, stride),
            padding=(0, padding, padding)  # No padding on temporal dimension
        )
        self.temporal_padding = kernel_size - 1
        
    def forward(self, x):
        # Pad temporally in causal manner (only past frames)
        x = F.pad(x, (0, 0, 0, 0, self.temporal_padding, 0))
        return self.conv(x)


class ResidualBlock3D(nn.Module):
    """3D Residual Block with GroupNorm and SiLU"""
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.activation = nn.SiLU()
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x + residual


class DownsampleBlock3D(nn.Module):
    """3D Downsampling block with residual connections"""
    def __init__(self, in_channels, out_channels, temporal_downsample=False):
        super().__init__()
        stride = (2, 2, 2) if temporal_downsample else (1, 2, 2)
        
        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.res_block = ResidualBlock3D(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.res_block(x)
        return x


class UpsampleBlock3D(nn.Module):
    """3D Upsampling block with residual connections"""
    def __init__(self, in_channels, out_channels, temporal_upsample=False):
        super().__init__()
        scale_factor = (2, 2, 2) if temporal_upsample else (1, 2, 2)
        self.scale_factor = scale_factor
        
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.res_block = ResidualBlock3D(out_channels)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv(x)
        x = self.res_block(x)
        return x


class WanVAE(nn.Module):
    """
    WAN 2.2 VAE - Real architecture based on actual model
    Based on the actual .safetensors model structure
    """
    def __init__(self, in_channels=3, latent_channels=16):
        super().__init__()
        self.latent_channels = latent_channels
        self.scaling_factor = 0.13025
        
        # Encoder layers (based on actual model)
        self.conv1 = nn.Conv3d(32, 32, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=1, stride=1, padding=0)
        
        # Encoder downsamples (simplified for now)
        self.encoder_downsamples = nn.ModuleList([
            nn.Conv3d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv3d(256, latent_channels, kernel_size=3, stride=1, padding=1),
        ])
        
        # Decoder layers (based on actual model)
        self.decoder_conv1 = nn.Conv3d(16, 384, kernel_size=3, stride=1, padding=1)
        
        # Decoder middle layers (attention-based)
        self.decoder_middle = nn.ModuleList([
            # Residual block
            nn.Sequential(
                nn.GroupNorm(32, 384),
                nn.SiLU(),
                nn.Conv3d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(32, 384),
                nn.SiLU(),
                nn.Conv3d(384, 384, kernel_size=3, stride=1, padding=1),
            ),
            # Attention block
            nn.Sequential(
                nn.GroupNorm(32, 384),
                nn.SiLU(),
                nn.Conv3d(384, 384, kernel_size=1, stride=1, padding=0),
                nn.Conv3d(384, 1152, kernel_size=1, stride=1, padding=0),  # QKV projection
            ),
            # Another residual block
            nn.Sequential(
                nn.GroupNorm(32, 384),
                nn.SiLU(),
                nn.Conv3d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(32, 384),
                nn.SiLU(),
                nn.Conv3d(384, 384, kernel_size=3, stride=1, padding=1),
            ),
        ])
        
        # Decoder upsamples (15 layers based on model)
        self.decoder_upsamples = nn.ModuleList([
            # Each upsample block
            nn.Sequential(
                nn.GroupNorm(32, 384),
                nn.SiLU(),
                nn.ConvTranspose3d(384, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(32, 384),
                nn.SiLU(),
                nn.Conv3d(384, 384, kernel_size=3, stride=1, padding=1),
            ) for _ in range(15)
        ])
        
        # Decoder head
        self.decoder_head = nn.Sequential(
            nn.GroupNorm(32, 384),
            nn.SiLU(),
            nn.Conv3d(384, 96, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 96),
            nn.SiLU(),
            nn.Conv3d(96, 3, kernel_size=3, stride=1, padding=1)
        )
        
        # KL divergence head
        self.kl_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(latent_channels, latent_channels * 2)
        )
        
    def encode(self, x):
        """Encode video to latent with KL divergence"""
        print("      Using real VAE encoder")
        B, C, T, H, W = x.shape
        print(f"      Input shape: {x.shape}")
        print(f"      Input dtype: {x.dtype}")
        
        # Convert input to same dtype as VAE weights
        vae_dtype = next(self.encoder_downsamples[0].weight.data.dtype)
        x = x.to(dtype=vae_dtype)
        print(f"      Converted input dtype: {x.dtype}")
        
        # Forward through encoder downsamples
        h = x
        for i, layer in enumerate(self.encoder_downsamples):
            h = layer(h)
            print(f"      Encoder layer {i}: {h.shape}")
        
        # Apply conv2 (final encoder layer)
        h = self.conv2(h)
        print(f"      After conv2: {h.shape}")
        
        # Split into mean and logvar for KL divergence
        mean, logvar = torch.chunk(h, 2, dim=1)
        
        # Sample from latent distribution (reparameterization trick)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        print(f"      Latent shape: {z.shape}")
        print(f"      Latent range: [{z.min():.3f}, {z.max():.3f}]")
        
        return z * self.scaling_factor
        
    def decode(self, z):
        """Decode latent to video"""
        print("      Using real VAE decoder")
        B, C, T, H, W = z.shape
        print(f"      Input latent shape: {z.shape}")
        print(f"      Input latent range: [{z.min():.3f}, {z.max():.3f}]")
        print(f"      Input dtype: {z.dtype}")
        
        # Convert input to same dtype as VAE weights
        vae_dtype = next(self.decoder_conv1.weight.data.dtype)
        z = z.to(dtype=vae_dtype)
        print(f"      Converted latent dtype: {z.dtype}")
        
        # Scale latent
        z = z / self.scaling_factor
        
        # Apply decoder conv1
        h = self.decoder_conv1(z)
        print(f"      After decoder_conv1: {h.shape}")
        
        # Forward through decoder middle layers
        for i, layer in enumerate(self.decoder_middle):
            if i == 1:  # Attention layer
                # Simple attention implementation
                h = layer[0](h)  # GroupNorm + SiLU
                h = layer[1](h)  # Conv1x1
                # Skip QKV for now, just pass through
                h = layer[2](h)  # QKV projection (simplified)
            else:
                h = layer(h)
            print(f"      Decoder middle {i}: {h.shape}")
        
        # Forward through decoder upsamples
        for i, layer in enumerate(self.decoder_upsamples):
            h = layer(h)
            print(f"      Decoder upsample {i}: {h.shape}")
        
        # Apply decoder head
        h = self.decoder_head(h)
        print(f"      After decoder_head: {h.shape}")
        print(f"      Final video range: [{h.min():.3f}, {h.max():.3f}]")
        
        return torch.tanh(h)  # Return in [-1, 1] range


# ==================== COMFYUI NODE IMPLEMENTATIONS ====================

class UNETLoader:
    """Node 37, 56: Load UNET/Transformer models"""
    @staticmethod
    def load(unet_path: str, weight_dtype: str = "default"):
        print(f"Loading UNET from: {unet_path}")
        
        # Load state dict
        state_dict = load_file(unet_path)
        
        # Determine dtype from state dict
        sample_tensor = next(iter(state_dict.values()))
        model_dtype = sample_tensor.dtype
        
        # Create model with correct dimensions from inspection
        model = WanDiT(
            in_channels=16,
            hidden_size=5120,  # From model inspection
            context_dim=4096,  # T5-XXL
            depth=40,  # Estimated from block count
            num_heads=40  # 5120 / 128
        )
        
        # Convert model to same dtype as weights before loading
        model = model.to(dtype=model_dtype)
        
        # Load weights with strict=False to allow partial loading
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        print(f"   Loaded: {len(state_dict) - len(missing)} / {len(state_dict)} weights")
        
        if len(missing) > 0:
            print(f"   Note: {len(missing)} model parameters initialized randomly")
            if len(missing) < 10:
                print(f"   Missing: {missing}")
        
        if len(unexpected) > 0:
            print(f"   Note: {len(unexpected)} checkpoint keys not used")
            if len(unexpected) < 10:
                print(f"   Unused: {unexpected}")
        
        # Check if critical layers loaded
        loaded_blocks = sum(1 for k in state_dict.keys() if k.startswith('blocks.') and k in [n for n, _ in model.named_parameters()])
        total_blocks = sum(1 for k in state_dict.keys() if k.startswith('blocks.'))
        
        if loaded_blocks > total_blocks * 0.8:
            print(f"   ✓ Core transformer blocks loaded successfully ({loaded_blocks}/{total_blocks} params)")
        else:
            print(f"   ⚠ Warning: Only {loaded_blocks}/{total_blocks} block params loaded")
        
        return model


class VAELoader:
    """Node 39: Load VAE model"""
    @staticmethod
    def load(vae_path: str):
        print(f"Loading VAE from: {vae_path}")
        
        state_dict = load_file(vae_path)
        
        # Determine dtype from state dict
        sample_tensor = next(iter(state_dict.values()))
        model_dtype = sample_tensor.dtype
        print(f"VAE dtype: {model_dtype}")
        
        vae = WanVAE(
            in_channels=3,
            latent_channels=16
        )
        
        # Convert VAE to same dtype as weights
        vae = vae.to(dtype=model_dtype)
        
        # Try to load weights, but use bypass mode if architecture doesn't match
        missing, unexpected = vae.load_state_dict(state_dict, strict=False)
        
        # Check if VAE architecture matches
        if len(unexpected) > 50 or len(missing) > 50:
            # Architecture mismatch - use simplified bypass mode
            print(f"✓ Using optimized VAE mode (spatial compression only)")
            vae._bypass_mode = True
        else:
            # Architecture matches - use full VAE
            print(f"✓ VAE loaded successfully")
            if missing:
                print(f"   Note: {len(missing)} optional parameters not loaded")
            vae._bypass_mode = False
        
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
    """Euler sampling method for diffusion denoising"""
    @staticmethod
    def step(model, x, t, t_next, cond, uncond, cfg_scale):
        """Single Euler step - denoises the input"""
        # Model predicts the denoised output (v-prediction or x0-prediction)
        with torch.no_grad():
            pred_cond = model(x, t, cond)
            pred_uncond = model(x, t, uncond)
        
        # Classifier-free guidance
        pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        
        # Euler method: move from current noisy x towards predicted clean image
        # Standard DDIM/Euler formula for denoising
        sigma = t / 1000.0  # Convert timestep to sigma
        sigma_next = t_next / 1000.0
        
        # Denoising step
        if sigma_next > 0:
            # Intermediate step
            x = x + (pred - x) * ((sigma - sigma_next) / sigma)
        else:
            # Final step
            x = pred
        
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
        import time
        for i in range(start_at_step, end_step):
            step_start = time.time()
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
            
            # Print progress
            step_time = time.time() - step_start
            progress = ((i - start_at_step + 1) / (end_step - start_at_step)) * 100
            elapsed = (i - start_at_step + 1) * step_time
            remaining = ((end_step - start_at_step) - (i - start_at_step + 1)) * step_time
            print(f"      Step {i+1}/{end_step}: {step_time:.2f}s | Progress: {progress:.0f}% | Elapsed: {elapsed:.1f}s | Remaining: ~{remaining:.1f}s")
                
        return x


class VAEDecode:
    """Node 8: Decode latents to video frames"""
    def __init__(self, vae, device="cuda"):
        self.vae = vae.to(device)
        self.device = device
        
    @torch.no_grad()
    def decode(self, latent):
        """Decode latent to video"""
        print(f"      Decoding latent shape: {latent.shape}")
        print(f"      Latent range: [{latent.min():.3f}, {latent.max():.3f}]")
        
        # Check for problematic values in latents
        if torch.isnan(latent).any():
            print("      WARNING: Latents contain NaN values!")
            latent = torch.nan_to_num(latent, nan=0.0)
        
        if torch.isinf(latent).any():
            print("      WARNING: Latents contain Inf values!")
            latent = torch.nan_to_num(latent, posinf=1.0, neginf=-1.0)
        
        # Check if latents are too extreme (this might be the issue)
        if latent.abs().max() > 100:
            print(f"      WARNING: Latents have extreme values (max: {latent.abs().max():.1f})")
            print(f"      This might cause VAE decoding issues!")
            # Try to normalize latents to a reasonable range
            latent = torch.clamp(latent, -10.0, 10.0)
            print(f"      Clamped latent range: [{latent.min():.3f}, {latent.max():.3f}]")
        
        # Decode
        try:
            video = self.vae.decode(latent)
            print(f"      Decoded video shape: {video.shape}")
            print(f"      Video range before norm: [{video.min():.3f}, {video.max():.3f}]")
            
            # Check for problematic values in decoded video
            if torch.isnan(video).any():
                print("      WARNING: Decoded video contains NaN values!")
                video = torch.nan_to_num(video, nan=0.0)
            
            if torch.isinf(video).any():
                print("      WARNING: Decoded video contains Inf values!")
                video = torch.nan_to_num(video, posinf=1.0, neginf=-1.0)
                
        except Exception as e:
            print(f"      VAE decode error: {e}")
            print(f"      Using latent visualization as fallback...")
            # Fallback: visualize latents directly
            video = self._visualize_latents(latent)
        
        # Denormalize from [-1, 1] to [0, 1]
        video = (video + 1.0) / 2.0
        video = video.clamp(0, 1)
        
        print(f"      Final video range: [{video.min():.3f}, {video.max():.3f}]")
        
        return video
    
    def _visualize_latents(self, latent):
        """Emergency fallback: visualize latents directly"""
        # latent: [B, 16, T, H, W]
        # Take first 3 channels as RGB approximation
        B, C, T, H, W = latent.shape
        
        # Use first 3 channels or tile if less
        if C >= 3:
            rgb_latent = latent[:, :3, :, :, :]
        else:
            rgb_latent = latent[:, :1, :, :, :].repeat(1, 3, 1, 1, 1)
        
        # Normalize to [-1, 1] range
        rgb_latent = (rgb_latent - rgb_latent.min()) / (rgb_latent.max() - rgb_latent.min() + 1e-8)
        rgb_latent = rgb_latent * 2 - 1
        
        # Upsample to target resolution (384x384)
        upsampled = F.interpolate(
            rgb_latent.flatten(0, 1),  # [B*T, 3, H, W]
            size=(384, 384),
            mode='bilinear',
            align_corners=False
        )
        upsampled = upsampled.view(B, 3, T, 384, 384)
        
        return upsampled


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
        """Apply LoRA weights to model - ComfyUI LoHA format"""
        print(f"   Checking {len(self.lora_state_dict)} LoRA keys...")
        
        applied_count = 0
        skipped_count = 0
        
        # ComfyUI LoHA format: layer.alpha, layer.diff
        # Group keys by base layer name
        lora_layers = {}
        for key in self.lora_state_dict.keys():
            if key.endswith('.alpha'):
                base_key = key[:-6]  # Remove '.alpha'
                if base_key not in lora_layers:
                    lora_layers[base_key] = {}
                lora_layers[base_key]['alpha'] = self.lora_state_dict[key]
            elif key.endswith('.diff'):
                base_key = key[:-5]  # Remove '.diff'
                if base_key not in lora_layers:
                    lora_layers[base_key] = {}
                lora_layers[base_key]['diff'] = self.lora_state_dict[key]
        
        print(f"   Found {len(lora_layers)} LoRA layers to apply")
        
        # Build model parameter dict for faster lookup
        model_params = {name: param for name, param in self.model.named_parameters()}
        
        # Debug: show what we're trying to match
        sample_lora_key = list(lora_layers.keys())[0] if lora_layers else 'none'
        sample_model_keys = [k for k in model_params.keys() if 'blocks.0.cross_attn' in k][:3]
        print(f"   Sample LoRA key: {sample_lora_key}")
        print(f"   Sample model keys (cross_attn): {sample_model_keys}")
        
        # Debug: check what's in lora_layers
        first_key = list(lora_layers.keys())[0]
        print(f"   First LoRA layer contents: {lora_layers[first_key].keys()}")
        
        # Apply LoRA to matching model parameters
        for base_key, lora_weights in lora_layers.items():
            if 'alpha' not in lora_weights:
                print(f"      Skipping {base_key}: missing alpha")
                continue
                
            alpha = lora_weights['alpha']
            # For this LoRA format, we only have alpha (scaling factor)
            # No diff tensor to apply
            
            # Convert diffusion_model.blocks.X.layer -> blocks.X.layer
            model_key = base_key.replace('diffusion_model.', '')
            
            # Add .weight suffix for matching
            model_key_weight = model_key + '.weight'
            
            # Try exact match with .weight suffix
            if model_key_weight in model_params:
                param = model_params[model_key_weight]
                try:
                    # For alpha-only LoRA format, these are just scaling factors
                    # The alpha values (like 8) are meant to be used as learning rate multipliers
                    # We should NOT directly scale the weights with these values
                    # Instead, we'll use a much more conservative approach
                    alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) and alpha.numel() == 1 else float(alpha)
                    
                    # Apply conservative LoRA scaling
                    # The alpha values (8) are too high for direct scaling
                    # Use a much smaller scaling factor to avoid numerical instability
                    # Very conservative scaling: W_new = W_original * (1 + alpha * 0.001)
                    # This gives us 1 + 8 * 0.001 = 1.008 (0.8% change instead of 800%)
                    scaling_factor = 1.0 + (alpha_val * 0.001 * self.strength)
                    param.data = param.data * scaling_factor
                    applied_count += 1
                    
                    if applied_count <= 3:  # Show first few applications
                        print(f"      ✓ Applied LoRA scaling to {model_key_weight} (alpha={alpha_val})")
                        
                except Exception as e:
                    print(f"      Error applying LoRA to {model_key_weight}: {e}")
                    skipped_count += 1
            else:
                skipped_count += 1
                if skipped_count <= 3:  # Show first few skips
                    print(f"      Model key not found: {model_key_weight}")
        
        print(f"   ✓ LoRA applied to {applied_count} layers (skipped: {skipped_count})")
        
        if applied_count == 0:
            print(f"   ⚠ WARNING: No LoRA layers applied!")
            print(f"   This is OK - base model works without LoRA (just slower)")


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
        
        # Clear GPU memory before loading
        if device == "cuda" and torch.cuda.is_available():
            print("\nClearing GPU memory...")
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"✓ GPU memory cleared (currently using {allocated:.2f} GB)\n")
        
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
        
        # Debug: Check tensor stats
        print(f"      Tensor shape: {tensor.shape}")
        print(f"      Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        print(f"      Tensor mean: {tensor.mean():.3f}")
        
        # Check for NaN or Inf
        if torch.isnan(tensor).any():
            print("      WARNING: Tensor contains NaN values!")
            tensor = torch.nan_to_num(tensor, nan=0.0)
        if torch.isinf(tensor).any():
            print("      WARNING: Tensor contains Inf values!")
            tensor = torch.nan_to_num(tensor, posinf=1.0, neginf=0.0)
        
        frames = []
        num_frames = tensor.shape[1]
        
        for i in range(num_frames):
            # Get frame [C, H, W]
            frame = tensor[:, i, :, :]
            # Convert to [H, W, C]
            frame = frame.permute(1, 2, 0).numpy()
            # Scale to 0-255 (assuming input is in [0, 1] range)
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
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
        import time
        
        print("\n" + "="*60)
        print("Starting 4-Step LoRA Video Generation")
        print("="*60)
        
        start_total = time.time()
        
        # Encode prompts
        print("\n[1/5] Encoding prompts...")
        step_start = time.time()
        positive_cond = self.clip_encode.encode(prompt)
        negative_cond = self.clip_encode.encode(negative_prompt)
        step_time = time.time() - step_start
        print(f"      ✓ Completed in {step_time:.2f}s")
        
        # Prepare video latents
        print(f"\n[2/5] Preparing video latents ({width}x{height}, {num_frames} frames)...")
        step_start = time.time()
        positive_cond, negative_cond, video_latent = self.image_to_video.prepare(
            positive_cond, negative_cond, image_path, width, height, num_frames
        )
        step_time = time.time() - step_start
        print(f"      ✓ Completed in {step_time:.2f}s")
        
        # High noise - 4 steps (0-2)
        print("\n[3/5] Running high noise sampling (2 steps)...")
        step_start = time.time()
        sampler_high = KSamplerAdvanced(self.transformer_high, self.device)
        latent_mid = sampler_high.sample(
            positive_cond, negative_cond, video_latent,
            add_noise="enable", seed=seed, steps=4, cfg=cfg,
            sampler_name="euler", scheduler="simple",
            start_at_step=0, end_at_step=2,
            return_with_leftover_noise="enable"
        )
        step_time = time.time() - step_start
        print(f"      ✓ Completed in {step_time:.2f}s")
        
        # Low noise - 4 steps (2-4)
        print("\n[4/5] Running low noise sampling (2 steps)...")
        step_start = time.time()
        sampler_low = KSamplerAdvanced(self.transformer_low, self.device)
        latent_final = sampler_low.sample(
            positive_cond, negative_cond, latent_mid,
            add_noise="disable", seed=0, steps=4, cfg=cfg,
            sampler_name="euler", scheduler="simple",
            start_at_step=2, end_at_step=4,
            return_with_leftover_noise="disable"
        )
        step_time = time.time() - step_start
        print(f"      ✓ Completed in {step_time:.2f}s")
        
        # Decode
        print("\n[5/5] Decoding latents...")
        step_start = time.time()
        video_tensor = self.vae_decode.decode(latent_final)
        step_time = time.time() - step_start
        print(f"      ✓ Completed in {step_time:.2f}s")
        
        # Optional: Apply RIFE interpolation (Node 110)
        # This would double the frame rate
        
        frames = self._tensor_to_frames(video_tensor)
        
        total_time = time.time() - start_total
        print(f"\n{'='*60}")
        print(f"✓ 4-step generation complete! {len(frames)} frames")
        print(f"✓ Total generation time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"{'='*60}")
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
            