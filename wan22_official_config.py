"""
WAN 2.2 Official Configuration System
Based on the official Wan2.2 repository configuration structure
"""

import os
import torch
from easydict import EasyDict
from typing import Dict, Any, Optional


# ==================== SHARED CONFIGURATION ====================

def get_shared_config():
    """Get shared Wan2.2 configuration"""
    wan_shared_cfg = EasyDict()
    
    # T5 configuration
    wan_shared_cfg.t5_model = 'umt5_xxl'
    wan_shared_cfg.t5_dtype = torch.bfloat16
    wan_shared_cfg.text_len = 512
    
    # Transformer configuration
    wan_shared_cfg.param_dtype = torch.bfloat16
    
    # Inference configuration
    wan_shared_cfg.num_train_timesteps = 1000
    wan_shared_cfg.sample_fps = 16
    wan_shared_cfg.sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
    wan_shared_cfg.frame_num = 81
    
    return wan_shared_cfg


# ==================== MODEL CONFIGURATIONS ====================

def get_i2v_A14B_config():
    """Get I2V A14B configuration"""
    i2v_A14B = EasyDict(__name__='Config: Wan I2V A14B')
    i2v_A14B.update(get_shared_config())
    
    # T5 configuration
    i2v_A14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
    i2v_A14B.t5_tokenizer = 'google/umt5-xxl'
    
    # VAE configuration
    i2v_A14B.vae_checkpoint = 'Wan2.1_VAE.pth'
    i2v_A14B.vae_stride = (4, 8, 8)
    
    # Transformer configuration
    i2v_A14B.patch_size = (1, 2, 2)
    i2v_A14B.dim = 5120
    i2v_A14B.ffn_dim = 13824
    i2v_A14B.freq_dim = 256
    i2v_A14B.num_heads = 40
    i2v_A14B.num_layers = 40
    i2v_A14B.window_size = (-1, -1)
    i2v_A14B.qk_norm = True
    i2v_A14B.cross_attn_norm = True
    i2v_A14B.eps = 1e-6
    i2v_A14B.low_noise_checkpoint = 'low_noise_model'
    i2v_A14B.high_noise_checkpoint = 'high_noise_model'
    
    # LoRA configuration
    i2v_A14B.lora_enabled = False
    i2v_A14B.lora_dir = 'loras'
    i2v_A14B.lora_files = {
        'low_noise_lora': 'wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors',
        'high_noise_lora': 'wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors'
    }
    i2v_A14B.lora_scale = 1.0
    
    # Inference configuration
    i2v_A14B.sample_shift = 5.0
    i2v_A14B.sample_steps = 40
    i2v_A14B.boundary = 0.900
    i2v_A14B.sample_guide_scale = (3.5, 3.5)  # low noise, high noise
    
    return i2v_A14B


def get_t2v_A14B_config():
    """Get T2V A14B configuration"""
    t2v_A14B = EasyDict(__name__='Config: Wan T2V A14B')
    t2v_A14B.update(get_shared_config())
    
    # T5 configuration
    t2v_A14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
    t2v_A14B.t5_tokenizer = 'google/umt5-xxl'
    
    # VAE configuration
    t2v_A14B.vae_checkpoint = 'Wan2.1_VAE.pth'
    t2v_A14B.vae_stride = (4, 8, 8)
    
    # Transformer configuration
    t2v_A14B.patch_size = (1, 2, 2)
    t2v_A14B.dim = 5120
    t2v_A14B.ffn_dim = 13824
    t2v_A14B.freq_dim = 256
    t2v_A14B.num_heads = 40
    t2v_A14B.num_layers = 40
    t2v_A14B.window_size = (-1, -1)
    t2v_A14B.qk_norm = True
    t2v_A14B.cross_attn_norm = True
    t2v_A14B.eps = 1e-6
    t2v_A14B.low_noise_checkpoint = 'low_noise_model'
    t2v_A14B.high_noise_checkpoint = 'high_noise_model'
    
    # LoRA configuration
    t2v_A14B.lora_enabled = False
    t2v_A14B.lora_dir = 'loras'
    t2v_A14B.lora_files = {
        'low_noise_lora': 'wan2.2_t2v_lightx2v_4steps_lora_v1_low_noise.safetensors',
        'high_noise_lora': 'wan2.2_t2v_lightx2v_4steps_lora_v1_high_noise.safetensors'
    }
    t2v_A14B.lora_scale = 1.0
    
    # Inference configuration
    t2v_A14B.sample_shift = 12.0
    t2v_A14B.sample_steps = 40
    t2v_A14B.boundary = 0.875
    t2v_A14B.sample_guide_scale = (3.0, 4.0)  # low noise, high noise
    
    return t2v_A14B


def get_ti2v_5B_config():
    """Get TI2V 5B configuration"""
    ti2v_5B = EasyDict(__name__='Config: Wan TI2V 5B')
    ti2v_5B.update(get_shared_config())
    
    # T5 configuration
    ti2v_5B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
    ti2v_5B.t5_tokenizer = 'google/umt5-xxl'
    
    # VAE configuration (Wan2.2 VAE for 5B model)
    ti2v_5B.vae_checkpoint = 'Wan2.2_VAE.pth'
    ti2v_5B.vae_stride = (4, 16, 16)
    
    # Transformer configuration
    ti2v_5B.patch_size = (1, 2, 2)
    ti2v_5B.dim = 3072
    ti2v_5B.ffn_dim = 14336
    ti2v_5B.freq_dim = 256
    ti2v_5B.num_heads = 24
    ti2v_5B.num_layers = 30
    ti2v_5B.window_size = (-1, -1)
    ti2v_5B.qk_norm = True
    ti2v_5B.cross_attn_norm = True
    ti2v_5B.eps = 1e-6
    
    # Inference configuration
    ti2v_5B.sample_fps = 24
    ti2v_5B.sample_shift = 5.0
    ti2v_5B.sample_steps = 50
    ti2v_5B.sample_guide_scale = 5.0
    ti2v_5B.frame_num = 121
    
    return ti2v_5B


# ==================== CONFIGURATION REGISTRY ====================

WAN_CONFIGS = {
    'i2v-A14B': get_i2v_A14B_config(),
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '704*1280': (704, 1280),
    '1280*704': (1280, 704)
}

MAX_AREA_CONFIGS = {
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
    '704*1280': 704 * 1280,
    '1280*704': 1280 * 704,
}

SUPPORTED_SIZES = {
    'i2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
}


# ==================== CONFIGURATION UTILITIES ====================

def get_config(model_type: str) -> EasyDict:
    """Get configuration for specified model type"""
    if model_type not in WAN_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}. Available: {list(WAN_CONFIGS.keys())}")
    return WAN_CONFIGS[model_type]


def get_supported_sizes(model_type: str) -> tuple:
    """Get supported sizes for model type"""
    if model_type not in SUPPORTED_SIZES:
        raise ValueError(f"Unsupported model type: {model_type}")
    return SUPPORTED_SIZES[model_type]


def validate_size(model_type: str, size: str) -> bool:
    """Validate if size is supported for model type"""
    return size in SUPPORTED_SIZES.get(model_type, ())


def get_size_config(size: str) -> tuple:
    """Get size configuration"""
    if size not in SIZE_CONFIGS:
        raise ValueError(f"Unsupported size: {size}. Available: {list(SIZE_CONFIGS.keys())}")
    return SIZE_CONFIGS[size]


def get_max_area(size: str) -> int:
    """Get maximum area for size"""
    if size not in MAX_AREA_CONFIGS:
        raise ValueError(f"Unsupported size: {size}")
    return MAX_AREA_CONFIGS[size]


# ==================== MODEL PATH UTILITIES ====================

def get_model_paths(model_dir: str, model_type: str) -> Dict[str, str]:
    """Get model file paths for specified model type"""
    config = get_config(model_type)
    
    paths = {
        'text_encoder': os.path.join(model_dir, config.t5_checkpoint),
        'text_encoder_tokenizer': os.path.join(model_dir, config.t5_tokenizer),
        'vae': os.path.join(model_dir, config.vae_checkpoint),
    }
    
    # Add model-specific paths
    if hasattr(config, 'low_noise_checkpoint'):
        paths['low_noise_model'] = os.path.join(model_dir, config.low_noise_checkpoint)
    if hasattr(config, 'high_noise_checkpoint'):
        paths['high_noise_model'] = os.path.join(model_dir, config.high_noise_checkpoint)
    if not hasattr(config, 'low_noise_checkpoint') and not hasattr(config, 'high_noise_checkpoint'):
        # Single model (like TI2V-5B)
        paths['model'] = model_dir
    
    return paths


def check_model_files(model_dir: str, model_type: str) -> Dict[str, bool]:
    """Check if all required model files exist"""
    paths = get_model_paths(model_dir, model_type)
    return {name: os.path.exists(path) for name, path in paths.items()}


def get_missing_files(model_dir: str, model_type: str) -> List[str]:
    """Get list of missing model files"""
    status = check_model_files(model_dir, model_type)
    return [name for name, exists in status.items() if not exists]


# ==================== LORA UTILITIES ====================

def get_lora_paths(model_dir: str, model_type: str) -> Dict[str, str]:
    """Get LoRA file paths for specified model type"""
    config = get_config(model_type)
    
    if not hasattr(config, 'lora_enabled') or not config.lora_enabled:
        return {}
    
    lora_dir = os.path.join(model_dir, config.lora_dir)
    paths = {}
    
    if hasattr(config, 'lora_files'):
        for lora_type, filename in config.lora_files.items():
            paths[lora_type] = os.path.join(lora_dir, filename)
    
    return paths


def check_lora_files(model_dir: str, model_type: str) -> Dict[str, bool]:
    """Check if LoRA files exist"""
    lora_paths = get_lora_paths(model_dir, model_type)
    return {name: os.path.exists(path) for name, path in lora_paths.items()}


def get_missing_lora_files(model_dir: str, model_type: str) -> List[str]:
    """Get list of missing LoRA files"""
    status = check_lora_files(model_dir, model_type)
    return [name for name, exists in status.items() if not exists]


def enable_lora_for_model(model_type: str) -> EasyDict:
    """Enable LoRA for a model configuration"""
    config = get_config(model_type)
    config.lora_enabled = True
    return config


def disable_lora_for_model(model_type: str) -> EasyDict:
    """Disable LoRA for a model configuration"""
    config = get_config(model_type)
    config.lora_enabled = False
    return config


# ==================== INFERENCE PARAMETERS ====================

def get_default_inference_params(model_type: str) -> Dict[str, Any]:
    """Get default inference parameters for model type"""
    config = get_config(model_type)
    
    params = {
        'steps': config.sample_steps,
        'shift': config.sample_shift,
        'cfg_scale': config.sample_guide_scale,
        'fps': config.sample_fps,
        'frame_num': config.frame_num,
        'negative_prompt': config.sample_neg_prompt
    }
    
    return params


def get_recommended_settings(model_type: str, gpu_memory_gb: float) -> Dict[str, Any]:
    """Get recommended settings based on GPU memory"""
    base_params = get_default_inference_params(model_type)
    
    if gpu_memory_gb >= 80:
        # High-end GPU settings
        return {
            **base_params,
            'offload_model': False,
            'convert_model_dtype': False,
            't5_cpu': False,
            'max_resolution': (1280, 720),
            'max_frames': 81
        }
    elif gpu_memory_gb >= 48:
        # Mid-range GPU settings
        return {
            **base_params,
            'offload_model': True,
            'convert_model_dtype': True,
            't5_cpu': False,
            'max_resolution': (1024, 576),
            'max_frames': 65
        }
    elif gpu_memory_gb >= 24:
        # Lower-end GPU settings
        return {
            **base_params,
            'offload_model': True,
            'convert_model_dtype': True,
            't5_cpu': True,
            'max_resolution': (832, 480),
            'max_frames': 49
        }
    else:
        # Very limited GPU memory
        return {
            **base_params,
            'offload_model': True,
            'convert_model_dtype': True,
            't5_cpu': True,
            'max_resolution': (640, 360),
            'max_frames': 33
        }


# ==================== CONFIGURATION VALIDATION ====================

def validate_config(config: EasyDict, model_type: str) -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []
    
    # Check required attributes
    required_attrs = [
        't5_checkpoint', 't5_tokenizer', 'vae_checkpoint',
        'dim', 'ffn_dim', 'num_heads', 'num_layers',
        'sample_steps', 'sample_shift', 'sample_guide_scale'
    ]
    
    for attr in required_attrs:
        if not hasattr(config, attr):
            issues.append(f"Missing required attribute: {attr}")
    
    # Check MoE models
    if model_type in ['t2v-A14B', 'i2v-A14B']:
        if not hasattr(config, 'low_noise_checkpoint'):
            issues.append("Missing low_noise_checkpoint for MoE model")
        if not hasattr(config, 'high_noise_checkpoint'):
            issues.append("Missing high_noise_checkpoint for MoE model")
        if not hasattr(config, 'boundary'):
            issues.append("Missing boundary for MoE model")
    
    # Check guide scale format
    if hasattr(config, 'sample_guide_scale'):
        if isinstance(config.sample_guide_scale, tuple) and len(config.sample_guide_scale) != 2:
            issues.append("sample_guide_scale tuple must have exactly 2 elements")
        elif not isinstance(config.sample_guide_scale, (int, float, tuple)):
            issues.append("sample_guide_scale must be number or tuple")
    
    return issues


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example usage
    print("Available model types:", list(WAN_CONFIGS.keys()))
    
    # Get configuration
    config = get_config('i2v-A14B')
    print(f"I2V A14B config: {config.__name__}")
    print(f"Model dimension: {config.dim}")
    print(f"Number of layers: {config.num_layers}")
    print(f"Sample steps: {config.sample_steps}")
    
    # Check model files
    model_dir = "./models"
    if os.path.exists(model_dir):
        status = check_model_files(model_dir, 'i2v-A14B')
        print(f"Model files status: {status}")
        
        missing = get_missing_files(model_dir, 'i2v-A14B')
        if missing:
            print(f"Missing files: {missing}")
        else:
            print("All model files present!")
    
    # Get recommended settings
    settings = get_recommended_settings('i2v-A14B', 48.0)
    print(f"Recommended settings for 48GB GPU: {settings}")
