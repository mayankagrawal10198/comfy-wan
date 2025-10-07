#!/usr/bin/env python3
"""
Inspect WAN 2.2 model architecture from safetensors file
This will show us the actual layer structure
"""

from safetensors.torch import load_file
import sys

def inspect_model(model_path):
    """Inspect model structure"""
    print(f"Loading: {model_path}")
    print("="*80)
    
    state_dict = load_file(model_path)
    
    print(f"\nTotal keys: {len(state_dict)}")
    print(f"\nFirst 20 keys:")
    for i, key in enumerate(list(state_dict.keys())[:20]):
        tensor = state_dict[key]
        print(f"  {i+1:2d}. {key:60s} {str(tensor.shape):30s} {tensor.dtype}")
    
    # Analyze structure
    print(f"\n" + "="*80)
    print("Architecture Analysis:")
    print("="*80)
    
    # Count different layer types
    layer_types = {}
    for key in state_dict.keys():
        parts = key.split('.')
        if len(parts) > 1:
            layer_type = '.'.join(parts[:3])  # First 3 parts
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    print("\nLayer type counts:")
    for layer_type, count in sorted(layer_types.items())[:30]:
        print(f"  {layer_type:50s}: {count} params")
    
    # Check for specific patterns
    print("\n" + "="*80)
    print("Key Patterns:")
    print("="*80)
    
    has_blocks = any('blocks' in k for k in state_dict.keys())
    has_cross_attn = any('cross_attn' in k for k in state_dict.keys())
    has_self_attn = any('self_attn' in k or 'attn' in k for k in state_dict.keys())
    has_ffn = any('ffn' in k or 'mlp' in k for k in state_dict.keys())
    
    print(f"  Has 'blocks': {has_blocks}")
    print(f"  Has 'cross_attn': {has_cross_attn}")
    print(f"  Has 'self_attn': {has_self_attn}")
    print(f"  Has 'ffn/mlp': {has_ffn}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
    
    inspect_model(model_path)
