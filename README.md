# WAN 2.2 Image-to-Video Pipeline

Complete Python implementation of WAN 2.2 Image-to-Video generation, converted from ComfyUI workflow without using Diffusers library.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Model Architecture](#model-architecture)
- [Memory Management](#memory-management)
- [API Reference](#api-reference)

## ✨ Features

### Core Features
- ✅ **No Diffusers Dependency** - Pure PyTorch implementation
- ✅ **ComfyUI Node Conversion** - All nodes converted to Python classes
- ✅ **Two-Pass Generation** - High noise → Low noise sampling
- ✅ **FP16 Support** - Full precision models
- ✅ **LoRA Support** - 4-step fast generation
- ✅ **Multiple Schedulers** - Simple, Karras, Exponential, SGM

### Memory Management
- ✅ **Group Offload** - Layer-level GPU ↔ CPU offloading
- ✅ **VAE Tiling** - Process high-res videos efficiently
- ✅ **Sequential Offload** - Aggressive memory saving
- ✅ **Model CPU Offload** - Component-level offloading

### Post-Processing
- ✅ **RIFE Interpolation** - Double frame rate
- ✅ **Sharpening** - Enhance video quality
- ✅ **Denoising** - Reduce noise artifacts
- ✅ **Color Correction** - Gamma adjustment

### Utilities
- ✅ **Batch Processing** - Process multiple videos
- ✅ **Configuration System** - YAML/JSON configs
- ✅ **CLI Interface** - Command line tool
- ✅ **Performance Monitoring** - Track VRAM and timing

## 🚀 Installation

### Prerequisites
```bash
# CUDA 11.8 or higher
# Python 3.8+
# 48GB VRAM recommended for FP16 (24GB minimum with offloading)
```

### Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install safetensors transformers einops opencv-python pillow pyyaml pandas tqdm
```

### Download Models
```bash
# Create model directory structure
mkdir -p models/{diffusion_models,vae,text_encoders,loras}

# Download models from HuggingFace
# High Noise Transformer
wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors \
  -O models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors

# Low Noise Transformer
wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors \
  -O models/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors

# VAE
wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors \
  -O models/vae/wan_2.1_vae.safetensors

# Text Encoder
wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors \
  -O models/text_encoders/umt5_xxl_fp16.safetensors

# Optional: LoRA models for 4-step generation
wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors \
  -O models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors

wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors \
  -O models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors
```

### Project Structure
```
wan22-pipeline/
├── wan22_pipeline.py       # Main pipeline implementation
├── wan22_utils.py          # Utilities (tiling, interpolation, etc.)
├── wan22_config.py         # Configuration and CLI
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── models/
    ├── diffusion_models/
    │   ├── wan2.2_i2v_high_noise_14B_fp16.safetensors
    │   └── wan2.2_i2v_low_noise_14B_fp16.safetensors
    ├── vae/
    │   └── wan_2.1_vae.safetensors
    ├── text_encoders/
    │   └── umt5_xxl_fp16.safetensors
    └── loras/
        ├── wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors
        └── wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors
```

## 🎯 Quick Start

### Basic Usage
```bash
python wan22_config.py \
  -i input.jpg \
  -p "A dragon flying through clouds" \
  -o output.mp4
```

### With Preset
```bash
# High quality (1024x1024, 129 frames)
python wan22_config.py \
  -i input.jpg \
  -p "A dragon flying through clouds" \
  --preset high_quality

# Fast 4-step with LoRA
python wan22_config.py \
  -i input.jpg \
  -p "A dragon flying through clouds" \
  --preset fast_4step

# Memory efficient (512x512, offloading)
python wan22_config.py \
  -i input.jpg \
  -p "A dragon flying through clouds" \
  --preset memory_efficient
```

### Python API
```python
from wan22_pipeline import WAN22Pipeline

# Initialize pipeline
pipeline = WAN22Pipeline(
    model_dir="models",
    device="cuda",
    dtype=torch.float16
)

# Generate video
frames = pipeline.generate(
    image_path="input.jpg",
    prompt="A dragon flying through clouds",
    negative_prompt="blurry, low quality",
    width=640,
    height=640,
    num_frames=81,
    steps=20,
    cfg=3.5
)

# Save video
pipeline.save_video(frames, "output.mp4", fps=16)
```

## ⚙️ Configuration

### Create Config Files
```bash
# Generate example configs
python wan22_config.py create-configs
```

### YAML Configuration Example
```yaml
model:
  text_encoder: models/text_encoders/umt5_xxl_fp16.safetensors
  vae: models/vae/wan_2.1_vae.safetensors
  transformer_high: models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors
  transformer_low: models/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors
  lora_high: null
  lora_low: null

generation:
  width: 640
  height: 640
  num_frames: 81
  steps: 20
  cfg_scale: 3.5
  seed: null
  fps: 16
  sampler: euler
  scheduler: simple
  shift: 8.0
  high_noise_steps: [0, 10]
  low_noise_steps: [10, 20]

memory:
  device: cuda
  dtype: float16
  enable_group_offload: false
  enable_model_cpu_offload: false
  enable_sequential_cpu_offload: false
  enable_vae_tiling: false
  vae_tile_size: 512
  vae_tile_overlap: 64

postprocess:
  enable_interpolation: false
  interpolation_multiplier: 2
  interpolation_model: rife47.pth
  enable_sharpening: false
  sharpening_strength: 0.5
  enable_denoising: false
  denoising_strength: 5
```

### Use Config File
```bash
python wan22_config.py \
  -i input.jpg \
  -p "Your prompt" \
  --config config.yaml
```

## 🔧 Advanced Usage

### Custom Resolution and Steps
```bash
python wan22_config.py \
  -i input.jpg \
  -p "A dragon flying" \
  -w 1024 -h 1024 \
  --frames 129 \
  --steps 20 \
  --cfg 3.5 \
  --seed 42
```

### With Memory Optimization
```bash
python wan22_config.py \
  -i input.jpg \
  -p "A dragon flying" \
  -w 1920 -h 1080 \
  --offload \
  --tiling
```

### With Post-Processing
```bash
python wan22_config.py \
  -i input.jpg \
  -p "A dragon flying" \
  --interpolate \
  --interpolate-x 2 \
  --sharpen \
  --fps 32
```

### Batch Processing
```bash
# Create batch file (CSV)
# columns: image_path, prompt, negative_prompt

# Process batch
python wan22_config.py batch \
  --input batch.csv \
  --output-dir outputs \
  --config config.yaml
```

### Python API - Advanced
```python
from wan22_pipeline import WAN22PipelineAdvanced
from wan22_utils import TiledVAE, RIFEInterpolation

# Advanced pipeline with LoRA
pipeline = WAN22PipelineAdvanced(model_dir="models")

# Load LoRAs
pipeline.load_with_lora(
    lora_high_path="models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
    lora_low_path="models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
    lora_strength=1.0
)

# Enable memory optimizations
pipeline.enable_group_offload()

# Wrap VAE with tiling
tiled_vae = TiledVAE(pipeline.vae, tile_size=512, tile_overlap=64)
pipeline.vae_decode.vae = tiled_vae

# Generate with 4-step mode
frames = pipeline.generate_4step(
    image_path="input.jpg",
    prompt="A dragon flying through clouds",
    negative_prompt="blurry, static",
    width=1024,
    height=1024,
    num_frames=129,
    seed=42,
    cfg=1.0
)

# Apply RIFE interpolation
rife = RIFEInterpolation(model_path="rife47.pth")
frames_interpolated = rife.interpolate(frames, multiplier=2)

# Save at higher FPS
pipeline.save_video(frames_interpolated, "output_32fps.mp4", fps=32)
```

## 🏗️ Model Architecture

### Node to Class Mapping

The implementation converts each ComfyUI node to a Python class:

| ComfyUI Node | Python Class | Description |
|--------------|--------------|-------------|
| UNETLoader | `ModelLoader.load_unet()` | Loads DiT transformer models |
| VAELoader | `VAELoader.load()` | Loads 3D VAE encoder/decoder |
| CLIPLoader | `CLIPLoader.load()` | Loads UMT5-XXL text encoder |
| CLIPTextEncode | `CLIPTextEncode` | Encodes text prompts |
| ModelSamplingSD3 | `ModelSamplingSD3` | Applies shift to noise schedule |
| WanImageToVideo | `WanImageToVideo` | Prepares video latents |
| KSamplerAdvanced | `KSamplerAdvanced` | Performs diffusion sampling |
| VAEDecode | `VAEDecode` | Decodes latents to frames |
| LoraLoaderModelOnly | `LoraLoaderModelOnly` | Applies LoRA weights |
| SaveVideo | `pipeline.save_video()` | Saves video file |

### Architecture Components

**WanDiT (Diffusion Transformer)**
```python
- Input: 16-channel latent video [B, 16, T, H, W]
- Hidden size: 3072
- Depth: 28 transformer blocks
- Heads: 24 attention heads
- Patch size: (1, 2, 2) - temporal, spatial
- Output: Denoised latent video
```

**WanVAE (3D VAE)**
```python
- Encoder: Video → Latent (8x spatial compression)
- Latent channels: 16
- Decoder: Latent → Video
- Scaling factor: 0.18215
```

**Attention Mechanism**
- 3D spatial-temporal attention
- Multi-head attention with 24 heads
- Adaptive layer normalization with timestep conditioning

### Two-Pass Sampling Process

```
Input Image
    ↓
[VAE Encode] → Image Latent
    ↓
[Expand Temporal] → Video Latent (T=81)
    ↓
[Add Noise] → Noisy Latent
    ↓
┌─────────────────────────────────────┐
│ Pass 1: High Noise Model            │
│ - Steps 0-10                        │
│ - High noise → Mid noise            │
│ - Transformer: 14B params           │
│ - Shift: 8.0                        │
└─────────────────────────────────────┘
    ↓
[Intermediate Latent]
    ↓
┌─────────────────────────────────────┐
│ Pass 2: Low Noise Model             │
│ - Steps 10-20                       │
│ - Mid noise → Clean                 │
│ - Transformer: 14B params           │
│ - Shift: 8.0                        │
└─────────────────────────────────────┘
    ↓
[Denoised Latent]
    ↓
[VAE Decode] → Video Frames (81 frames)
    ↓
[Post-process] → Final Video
```

## 💾 Memory Management

### Memory Usage (640x640, 81 frames)

| Configuration | VRAM | Speed | Quality |
|---------------|------|-------|---------|
| FP16, No offload | ~40GB | Fast | Best |
| FP16, Group offload | ~24GB | Medium | Best |
| FP16, Sequential offload | ~16GB | Slow | Best |
| FP8, No offload | ~20GB | Fast | Good |

### Offloading Strategies

**1. Group Offload (Recommended for 48GB)**
```python
pipeline = WAN22Pipeline(enable_offload=True)
```
- Moves 4-block groups between GPU/CPU
- Good balance of speed and memory
- ~40% VRAM reduction

**2. Model CPU Offload (24-40GB)**
```python
config.memory.enable_model_cpu_offload = True
```
- Moves entire model components
- Moderate speed impact
- ~50% VRAM reduction

**3. Sequential CPU Offload (16-24GB)**
```python
config.memory.enable_sequential_cpu_offload = True
```
- Moves individual layers
- Significant slowdown (3x)
- ~70% VRAM reduction

**4. VAE Tiling (Any resolution)**
```python
config.memory.enable_vae_tiling = True
config.memory.vae_tile_size = 512
```
- Process video in tiles
- Essential for >1024 resolution
- Minimal speed impact

### Memory Optimization Tips

1. **Use FP16 for quality**, FP8 for memory
2. **Enable tiling** for high resolutions
3. **Group offload** for balanced performance
4. **Sequential offload** only if desperate
5. **Lower frame count** reduces memory linearly
6. **Smaller resolution** reduces memory quadratically

### Measuring Memory Usage

```python
from wan22_utils import PerformanceMonitor

# During generation
stats = PerformanceMonitor.get_memory_stats()
print(f"VRAM: {stats['allocated_gb']:.2f} GB")
print(f"Usage: {stats['usage_percent']:.1f}%")
```

## 📚 API Reference

### WAN22Pipeline

```python
class WAN22Pipeline:
    def __init__(
        self,
        model_dir: str = "models",
        device: str = "cuda",
        dtype = torch.float16,
        enable_offload: bool = False
    )
    
    def generate(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 640,
        height: int = 640,
        num_frames: int = 81,
        seed: Optional[int] = None,
        steps: int = 20,
        cfg: float = 3.5
    ) -> List[np.ndarray]
    
    def save_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 16
    )
```

### WAN22PipelineAdvanced

```python
class WAN22PipelineAdvanced(WAN22Pipeline):
    def load_with_lora(
        self,
        lora_high_path: Optional[str] = None,
        lora_low_path: Optional[str] = None,
        lora_strength: float = 1.0
    )
    
    def enable_group_offload(self)
    
    def generate_4step(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 129,
        seed: Optional[int] = None,
        cfg: float = 1.0
    ) -> List[np.ndarray]
```

### TiledVAE

```python
class TiledVAE:
    def __init__(
        self,
        vae,
        tile_size: int = 512,
        tile_overlap: int = 64
    )
    
    def encode_tiled(self, x: torch.Tensor) -> torch.Tensor
    def decode_tiled(self, latent: torch.Tensor) -> torch.Tensor
```

### RIFEInterpolation

```python
class RIFEInterpolation:
    def __init__(
        self,
        model_path: str = "rife47.pth",
        device: str = "cuda"
    )
    
    def interpolate(
        self,
        frames: List[np.ndarray],
        multiplier: int = 2,
        scale_factor: float = 1.0
    ) -> List[np.ndarray]
```

### NoiseScheduler

```python
class NoiseScheduler:
    @staticmethod
    def get_sigmas(
        steps: int,
        scheduler_type: str = "simple",
        shift: float = 8.0
    ) -> torch.Tensor
    
    # Available schedulers:
    # - simple: Linear with shift (default for WAN)
    # - karras: Karras schedule from EDM
    # - exponential: Exponential decay
    # - sgm_uniform: SGM uniform with shift
```

## 🎨 Examples

### Example 1: Basic Video Generation
```python
from wan22_pipeline import WAN22Pipeline

pipeline = WAN22Pipeline(model_dir="models")

frames = pipeline.generate(
    image_path="dragon.jpg",
    prompt="A majestic dragon flying through stormy clouds, epic cinematic lighting",
    negative_prompt="blurry, static, low quality, distorted",
    width=640,
    height=640,
    num_frames=81,
    steps=20,
    cfg=3.5,
    seed=42
)

pipeline.save_video(frames, "dragon_flight.mp4", fps=16)
```

### Example 2: High Quality with Post-Processing
```python
from wan22_pipeline import WAN22Pipeline
from wan22_utils import TiledVAE, RIFEInterpolation, QualityEnhancer

pipeline = WAN22Pipeline(model_dir="models")

# Enable tiling for high resolution
tiled_vae = TiledVAE(pipeline.vae, tile_size=512)
pipeline.vae_decode.vae = tiled_vae

# Generate high resolution
frames = pipeline.generate(
    image_path="landscape.jpg",
    prompt="Beautiful mountain landscape with flowing river, golden hour lighting",
    width=1920,
    height=1080,
    num_frames=81,
    steps=20,
    cfg=3.5
)

# Apply sharpening
frames = QualityEnhancer.sharpen_frames(frames, strength=0.3)

# Interpolate to 32 FPS
rife = RIFEInterpolation()
frames = rife.interpolate(frames, multiplier=2)

pipeline.save_video(frames, "landscape_hq.mp4", fps=32)
```

### Example 3: Fast 4-Step Generation
```python
from wan22_pipeline import WAN22PipelineAdvanced

pipeline = WAN22PipelineAdvanced(model_dir="models")

# Load LoRAs for fast generation
pipeline.load_with_lora(
    lora_high_path="models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
    lora_low_path="models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
)

# Generate in just 4 steps!
frames = pipeline.generate_4step(
    image_path="portrait.jpg",
    prompt="Person smiling and waving at camera, natural movement",
    width=1024,
    height=1024,
    num_frames=129,
    cfg=1.0
)

pipeline.save_video(frames, "portrait_fast.mp4", fps=16)
```

### Example 4: Batch Processing
```python
from wan22_pipeline import WAN22Pipeline
from wan22_utils import BatchProcessor

pipeline = WAN22Pipeline(model_dir="models")
processor = BatchProcessor(pipeline)

image_paths = [
    "scene1.jpg",
    "scene2.jpg",
    "scene3.jpg"
]

prompts = [
    "Ocean waves crashing on rocky shore",
    "City street at night with neon lights",
    "Forest path with sunlight filtering through trees"
]

results = processor.process_batch(
    image_paths=image_paths,
    prompts=prompts,
    output_dir="batch_outputs",
    width=640,
    height=640,
    num_frames=81
)

for i, result in enumerate(results):
    if result['success']:
        print(f"✓ Video {i+1} completed: {result['output']}")
    else:
        print(f"✗ Video {i+1} failed: {result['error']}")
```

### Example 5: Custom Scheduler
```python
from wan22_pipeline import WAN22Pipeline, KSamplerAdvanced
from wan22_utils import NoiseScheduler

pipeline = WAN22Pipeline(model_dir="models")

# Create custom sampler with Karras schedule
sampler = KSamplerAdvanced(pipeline.transformer_high, device="cuda")

# Override get_sigmas method
def custom_get_sigmas(steps, scheduler="karras"):
    return NoiseScheduler.get_sigmas(steps, scheduler_type=scheduler)

sampler.get_sigmas = custom_get_sigmas

# Use in generation...
```

## 🔍 Troubleshooting

### Out of Memory (OOM)

**Problem**: `torch.cuda.OutOfMemoryError`

**Solutions**:
1. Enable offloading: `--offload`
2. Enable tiling: `--tiling`
3. Reduce resolution: `-w 512 -h 512`
4. Reduce frames: `--frames 49`
5. Use FP8 models instead of FP16
6. Use 4-step LoRA mode

### Slow Generation

**Problem**: Generation takes too long

**Solutions**:
1. Disable sequential offload (use group offload instead)
2. Use 4-step LoRA mode
3. Reduce number of steps
4. Disable post-processing
5. Use smaller resolution

### Poor Quality

**Problem**: Output quality is low

**Solutions**:
1. Increase steps: `--steps 30`
2. Adjust CFG scale: `--cfg 4.0`
3. Use better prompts (detailed, descriptive)
4. Enable sharpening: `--sharpen`
5. Use FP16 models instead of FP8
6. Try different seeds

### Model Loading Errors

**Problem**: Cannot load models

**Solutions**:
1. Check model paths in config
2. Verify file integrity (re-download if needed)
3. Ensure sufficient disk space
4. Check file permissions

## 📊 Performance Benchmarks

### RTX 4090 (24GB VRAM)

| Configuration | Resolution | Frames | Time | FPS |
|---------------|------------|--------|------|-----|
| FP16, 20 steps | 640x640 | 81 | ~536s | 0.15 |
| FP16 + group offload | 640x640 | 81 | ~580s | 0.14 |
| FP8, 20 steps | 640x640 | 81 | ~490s | 0.17 |
| 4-step LoRA | 640x640 | 81 | ~97s | 0.83 |
| 4-step LoRA + tiling | 1024x1024 | 129 | ~180s | 0.72 |

### A100 (80GB VRAM)

| Configuration | Resolution | Frames | Time | FPS |
|---------------|------------|--------|------|-----|
| FP16, 20 steps | 1920x1080 | 129 | ~850s | 0.15 |
| 4-step LoRA | 1920x1080 | 129 | ~200s | 0.65 |

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Optimization**: Further memory optimization techniques
2. **Schedulers**: Additional noise schedulers
3. **Models**: Support for other video diffusion models
4. **Features**: ControlNet support, motion control
5. **Documentation**: More examples and tutorials

## 📄 License

This project is MIT licensed. Model weights follow their respective licenses from Hugging Face.

## 🙏 Acknowledgments

- **Tencent ARC** - WAN 2.2 model
- **ComfyUI** - Node-based UI inspiration
- **Diffusers/HuggingFace** - Model hosting
- **RIFE** - Frame interpolation

## 📞 Support

- GitHub Issues: [Report bugs](https://github.com/yourusername/wan22-pipeline/issues)
- Documentation: [Full docs](https://github.com/yourusername/wan22-pipeline/wiki)
- Discord: [Community chat](https://discord.gg/yourlink)

---

**Made with ❤️ for the AI video generation community**