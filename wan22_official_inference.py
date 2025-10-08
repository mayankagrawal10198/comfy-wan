"""
WAN 2.2 Official Inference Script
Based on the official Wan2.2 repository implementation
"""

import argparse
import logging
import os
import sys
import random
import torch
from PIL import Image
from typing import Optional, Tuple

# Import official implementation
try:
    from wan22_official_pipeline import OfficialWan22Pipeline, create_official_pipeline
    from wan22_official_config import (
        WAN_CONFIGS, get_config, get_supported_sizes, 
        validate_size, get_recommended_settings, check_model_files
    )
    OFFICIAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import official implementation: {e}")
    OFFICIAL_AVAILABLE = False


# ==================== ARGUMENT PARSING ====================

def create_argument_parser():
    """Create argument parser for inference script"""
    parser = argparse.ArgumentParser(
        description="WAN 2.2 Official Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image-to-Video generation
  python wan22_official_inference.py --image input.jpg --prompt "A dragon flying" --output output.mp4
  
  # With LoRA (faster, 4 steps)
  python wan22_official_inference.py --image input.jpg --prompt "A dragon flying" --use-lora --output output.mp4
  
  # With custom parameters
  python wan22_official_inference.py --image input.jpg --prompt "A dragon flying" --width 1280 --height 720 --steps 40 --cfg 3.5 --seed 42
        """
    )
    
    # Task selection (I2V only)
    parser.add_argument(
        '--task',
        type=str,
        default='i2v-A14B',
        choices=['i2v-A14B'],
        help='Task type: i2v-A14B (Image-to-Video only)'
    )
    
    # Input/Output
    parser.add_argument('--image', type=str, required=True, help='Input image path (required for I2V)')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--negative', type=str, default='', help='Negative prompt')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video path')
    
    # Model configuration
    parser.add_argument('--model-dir', type=str, default='./models', help='Model directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32', 'bfloat16'], help='Data type')
    
    # Generation parameters
    parser.add_argument('--width', type=int, help='Video width')
    parser.add_argument('--height', type=int, help='Video height')
    parser.add_argument('--frames', type=int, help='Number of frames')
    parser.add_argument('--steps', type=int, help='Sampling steps')
    parser.add_argument('--cfg', type=float, help='CFG scale')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed (-1 for random)')
    parser.add_argument('--shift', type=float, help='Noise schedule shift')
    parser.add_argument('--solver', type=str, default='unipc', choices=['unipc', 'dpm++'], help='Sampling solver')
    parser.add_argument('--fps', type=int, help='Output FPS')
    
    # Memory management
    parser.add_argument('--offload', action='store_true', help='Enable model offloading')
    parser.add_argument('--convert-dtype', action='store_true', help='Convert model dtype')
    parser.add_argument('--t5-cpu', action='store_true', help='Place T5 on CPU')
    
    # LoRA configuration
    parser.add_argument('--use-lora', action='store_true', help='Enable LoRA for faster inference (4 steps)')
    parser.add_argument('--lora-dir', type=str, default='loras', help='LoRA directory')
    parser.add_argument('--lora-high', type=str, help='High noise LoRA file path')
    parser.add_argument('--lora-low', type=str, help='Low noise LoRA file path')
    parser.add_argument('--lora-scale', type=float, default=1.0, help='LoRA strength/scale (0.0-2.0)')
    
    # Auto-configuration
    parser.add_argument('--auto-config', action='store_true', help='Auto-configure based on GPU memory')
    parser.add_argument('--gpu-memory', type=float, help='GPU memory in GB (for auto-config)')
    
    # Logging
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    
    return parser


# ==================== CONFIGURATION HELPERS ====================

def get_auto_config(args, model_type: str) -> dict:
    """Get automatic configuration based on GPU memory"""
    if args.gpu_memory:
        gpu_memory = args.gpu_memory
    else:
        # Try to detect GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            gpu_memory = 8.0  # Default to 8GB if CUDA not available
    
    settings = get_recommended_settings(model_type, gpu_memory)
    
    # Override with command line arguments if provided
    config = {}
    if args.width is None:
        config['width'] = settings['max_resolution'][0]
    if args.height is None:
        config['height'] = settings['max_resolution'][1]
    if args.frames is None:
        config['frames'] = settings['max_frames']
    if args.steps is None:
        config['steps'] = settings['steps']
    if args.cfg is None:
        config['cfg'] = settings['cfg_scale']
    if args.shift is None:
        config['shift'] = settings['shift']
    if args.fps is None:
        config['fps'] = settings['fps']
    
    # Memory management
    config['offload_model'] = settings['offload_model']
    config['convert_model_dtype'] = settings['convert_model_dtype']
    config['t5_cpu'] = settings['t5_cpu']
    
    return config


def get_default_config(model_type: str) -> dict:
    """Get default configuration for model type"""
    config = get_config(model_type)
    
    # Get default sizes
    supported_sizes = get_supported_sizes(model_type)
    default_size = supported_sizes[0]  # Use first supported size
    
    return {
        'width': SIZE_CONFIGS[default_size][0],
        'height': SIZE_CONFIGS[default_size][1],
        'frames': config.frame_num,
        'steps': config.sample_steps,
        'cfg': config.sample_guide_scale[0] if isinstance(config.sample_guide_scale, tuple) else config.sample_guide_scale,
        'shift': config.sample_shift,
        'fps': config.sample_fps,
        'offload_model': True,
        'convert_model_dtype': True,
        't5_cpu': False
    }


# ==================== VALIDATION ====================

def validate_arguments(args) -> list:
    """Validate command line arguments"""
    issues = []
    
    # Check if official implementation is available
    if not OFFICIAL_AVAILABLE:
        issues.append("Official Wan2.2 implementation not available")
        return issues
    
    # Check task-specific requirements (I2V requires image)
    if not args.image:
        issues.append("Image path required for I2V task")
    
    # Check model directory
    if not os.path.exists(args.model_dir):
        issues.append(f"Model directory does not exist: {args.model_dir}")
    else:
        # Check model files
        missing_files = get_missing_files(args.model_dir, args.task)
        if missing_files:
            issues.append(f"Missing model files: {missing_files}")
        
        # Check LoRA files if LoRA is enabled
        if args.use_lora:
            missing_lora = get_missing_lora_files(args.model_dir, args.task)
            if missing_lora:
                issues.append(f"Missing LoRA files: {missing_lora}")
    
    # Check size validation
    if args.width and args.height:
        size_str = f"{args.height}*{args.width}"
        if not validate_size(args.task, size_str):
            supported = get_supported_sizes(args.task)
            issues.append(f"Unsupported size {size_str} for {args.task}. Supported: {supported}")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        issues.append("CUDA not available but device set to cuda")
    
    return issues


# ==================== MAIN INFERENCE FUNCTION ====================

def run_inference(args):
    """Run inference with given arguments"""
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # Validate arguments
    issues = validate_arguments(args)
    if issues:
        print("Validation errors:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    
    # Get configuration
    if args.auto_config:
        config = get_auto_config(args, args.task)
    else:
        config = get_default_config(args.task)
    
    # Override with command line arguments
    if args.width:
        config['width'] = args.width
    if args.height:
        config['height'] = args.height
    if args.frames:
        config['frames'] = args.frames
    if args.steps:
        config['steps'] = args.steps
    if args.cfg:
        config['cfg'] = args.cfg
    if args.shift:
        config['shift'] = args.shift
    if args.fps:
        config['fps'] = args.fps
    
    # Memory management overrides
    if args.offload:
        config['offload_model'] = True
    if args.convert_dtype:
        config['convert_model_dtype'] = True
    if args.t5_cpu:
        config['t5_cpu'] = True
    
    # LoRA configuration
    config['use_lora'] = args.use_lora
    if args.use_lora:
        config['lora_dir'] = args.lora_dir
        config['lora_scale'] = args.lora_scale
        config['lora_high'] = args.lora_high
        config['lora_low'] = args.lora_low
        
        # If LoRA is enabled, reduce steps for faster inference
        if not args.steps:  # Only override if steps not explicitly set
            config['steps'] = 4  # LoRA typically uses 4 steps
    
    # Setup seed
    if args.seed < 0:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = args.seed
    
    # Print configuration
    if not args.quiet:
        print("="*60)
        print("WAN 2.2 Official Inference")
        print("="*60)
        print(f"Task: {args.task}")
        print(f"Model Directory: {args.model_dir}")
        print(f"Device: {args.device}")
        print(f"Data Type: {args.dtype}")
        print(f"Resolution: {config['width']}x{config['height']}")
        print(f"Frames: {config['frames']}")
        print(f"Steps: {config['steps']}")
        print(f"CFG Scale: {config['cfg']}")
        print(f"Seed: {seed}")
        if config.get('use_lora', False):
            print(f"LoRA: Enabled (Scale: {config.get('lora_scale', 1.0)})")
            if config.get('lora_high'):
                print(f"LoRA High: {config['lora_high']}")
            if config.get('lora_low'):
                print(f"LoRA Low: {config['lora_low']}")
        else:
            print("LoRA: Disabled")
        print(f"Prompt: {args.prompt}")
        if args.image:
            print(f"Image: {args.image}")
        print(f"Output: {args.output}")
        print("="*60)
    
    try:
        # Create pipeline
        if not args.quiet:
            print("Initializing pipeline...")
        
        pipeline = create_official_pipeline(
            model_dir=args.model_dir,
            model_type=args.task,
            device=args.device,
            dtype=args.dtype,
            offload_model=config['offload_model'],
            convert_model_dtype=config['convert_model_dtype'],
            t5_cpu=config['t5_cpu'],
            use_lora=config.get('use_lora', False),
            lora_dir=config.get('lora_dir', 'loras'),
            lora_high=config.get('lora_high'),
            lora_low=config.get('lora_low'),
            lora_scale=config.get('lora_scale', 1.0)
        )
        
        if not args.quiet:
            print("Pipeline initialized successfully!")
            print("Generating video...")
        
        # Generate video
        video = pipeline.generate(
            prompt=args.prompt,
            image_path=args.image,
            negative_prompt=args.negative,
            width=config['width'],
            height=config['height'],
            num_frames=config['frames'],
            steps=config['steps'],
            cfg_scale=config['cfg'],
            seed=seed,
            shift=config['shift'],
            sample_solver=args.solver,
            output_path=args.output,
            fps=config['fps']
        )
        
        if not args.quiet:
            print(f"✓ Video generation successful!")
            print(f"✓ Output saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


# ==================== BATCH PROCESSING ====================

def run_batch_inference(batch_file: str, args):
    """Run batch inference from file"""
    import json
    import csv
    
    # Load batch file
    if batch_file.endswith('.json'):
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
    elif batch_file.endswith('.csv'):
        with open(batch_file, 'r') as f:
            reader = csv.DictReader(f)
            batch_data = list(reader)
    else:
        print("Batch file must be JSON or CSV")
        return 1
    
    # Create output directory
    output_dir = os.path.dirname(args.output) or "batch_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for i, item in enumerate(batch_data):
        print(f"\nProcessing {i+1}/{len(batch_data)}")
        
        # Update arguments
        batch_args = argparse.Namespace(**vars(args))
        batch_args.prompt = item['prompt']
        batch_args.image = item.get('image', None)
        batch_args.negative = item.get('negative', '')
        batch_args.output = os.path.join(output_dir, f"batch_{i:04d}.mp4")
        
        # Run inference
        result = run_inference(batch_args)
        results.append({
            'index': i,
            'status': 'success' if result == 0 else 'failed',
            'output': batch_args.output
        })
    
    # Save results
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBatch processing complete!")
    print(f"Results saved to: {results_file}")
    
    return 0


# ==================== MAIN ENTRY POINT ====================

def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Check if batch processing
    if hasattr(args, 'batch') and args.batch:
        return run_batch_inference(args.batch, args)
    
    # Run single inference
    return run_inference(args)


if __name__ == "__main__":
    sys.exit(main())
