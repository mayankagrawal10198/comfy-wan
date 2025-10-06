"""
WAN 2.2 Configuration Manager and Inference Script
Complete configuration management and CLI interface
"""

import yaml
import json
import argparse
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import torch


# ==================== CONFIGURATION CLASSES ====================

@dataclass
class ModelConfig:
    """Model file paths configuration"""
    text_encoder: str = "models/text_encoders/umt5_xxl_fp16.safetensors"
    vae: str = "models/vae/wan_2.1_vae.safetensors"
    transformer_high: str = "models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
    transformer_low: str = "models/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
    lora_high: Optional[str] = None
    lora_low: Optional[str] = None


@dataclass
class GenerationConfig:
    """Generation parameters"""
    width: int = 640
    height: int = 640
    num_frames: int = 81
    steps: int = 20
    cfg_scale: float = 3.5
    seed: Optional[int] = None
    fps: int = 16
    
    # Sampling parameters
    sampler: str = "euler"
    scheduler: str = "simple"
    shift: float = 8.0
    
    # Two-pass parameters
    high_noise_steps: tuple = (0, 10)
    low_noise_steps: tuple = (10, 20)


@dataclass
class MemoryConfig:
    """Memory management settings"""
    device: str = "cuda"
    dtype: str = "float16"
    enable_group_offload: bool = False
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_vae_tiling: bool = False
    vae_tile_size: int = 512
    vae_tile_overlap: int = 64


@dataclass
class PostProcessConfig:
    """Post-processing settings"""
    enable_interpolation: bool = False
    interpolation_multiplier: int = 2
    interpolation_model: str = "rife47.pth"
    enable_sharpening: bool = False
    sharpening_strength: float = 0.5
    enable_denoising: bool = False
    denoising_strength: int = 5


@dataclass
class WAN22Config:
    """Complete WAN 2.2 configuration"""
    model: ModelConfig
    generation: GenerationConfig
    memory: MemoryConfig
    postprocess: PostProcessConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary"""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            generation=GenerationConfig(**config_dict.get('generation', {})),
            memory=MemoryConfig(**config_dict.get('memory', {})),
            postprocess=PostProcessConfig(**config_dict.get('postprocess', {}))
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load config from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: str):
        """Load config from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'model': asdict(self.model),
            'generation': asdict(self.generation),
            'memory': asdict(self.memory),
            'postprocess': asdict(self.postprocess)
        }
    
    def save_yaml(self, yaml_path: str):
        """Save config to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, json_path: str):
        """Save config to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ==================== PRESET CONFIGURATIONS ====================

class ConfigPresets:
    """Pre-defined configuration presets"""
    
    @staticmethod
    def standard_quality():
        """Standard quality preset (640x640, 20 steps)"""
        return WAN22Config(
            model=ModelConfig(),
            generation=GenerationConfig(
                width=640,
                height=640,
                num_frames=81,
                steps=20,
                cfg_scale=3.5,
                fps=16
            ),
            memory=MemoryConfig(
                enable_group_offload=False
            ),
            postprocess=PostProcessConfig()
        )
    
    @staticmethod
    def high_quality():
        """High quality preset (1024x1024, 20 steps)"""
        return WAN22Config(
            model=ModelConfig(),
            generation=GenerationConfig(
                width=1024,
                height=1024,
                num_frames=129,
                steps=20,
                cfg_scale=3.5,
                fps=16
            ),
            memory=MemoryConfig(
                enable_vae_tiling=True,
                vae_tile_size=512
            ),
            postprocess=PostProcessConfig(
                enable_sharpening=True,
                sharpening_strength=0.3
            )
        )
    
    @staticmethod
    def fast_4step():
        """Fast 4-step LoRA preset"""
        return WAN22Config(
            model=ModelConfig(
                lora_high="models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
                lora_low="models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
            ),
            generation=GenerationConfig(
                width=384,  # Further reduced for 48GB VRAM
                height=384,  # Further reduced for 48GB VRAM
                num_frames=49,  # Reduced from 81 for memory efficiency
                steps=4,
                cfg_scale=1.0,
                fps=16,
                high_noise_steps=(0, 2),
                low_noise_steps=(2, 4)
            ),
            memory=MemoryConfig(
                enable_group_offload=True,  # Enable by default
                enable_vae_tiling=True,
                vae_tile_size=192  # Smaller tiles for 384x384
            ),
            postprocess=PostProcessConfig(
                enable_interpolation=True,
                interpolation_multiplier=2
            )
        )
    
    @staticmethod
    def memory_efficient():
        """Memory efficient preset for limited VRAM"""
        return WAN22Config(
            model=ModelConfig(),
            generation=GenerationConfig(
                width=512,
                height=512,
                num_frames=49,
                steps=20,
                cfg_scale=3.5,
                fps=12
            ),
            memory=MemoryConfig(
                enable_group_offload=True,
                enable_vae_tiling=True,
                vae_tile_size=256
            ),
            postprocess=PostProcessConfig()
        )


# ==================== INFERENCE ENGINE ====================

class WAN22Inference:
    """Main inference engine with configuration support"""
    
    def __init__(self, config: WAN22Config):
        self.config = config
        self.pipeline = None
        self.monitor = None
        
    def initialize(self):
        """Initialize pipeline with configuration"""
        from wan22_pipeline import WAN22Pipeline, WAN22PipelineAdvanced
        from wan22_utils import (
            TiledVAE, RIFEInterpolation, PerformanceMonitor,
            QualityEnhancer
        )
        
        print("\n" + "="*60)
        print("Initializing WAN 2.2 Inference Engine")
        print("="*60)
        
        # Initialize performance monitor
        self.monitor = PerformanceMonitor()
        self.monitor.start("initialization")
        
        # Get dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        dtype = dtype_map.get(self.config.memory.dtype, torch.float16)
        
        # Initialize pipeline
        if self.config.model.lora_high or self.config.model.lora_low:
            print("Using LoRA-enhanced pipeline...")
            self.pipeline = WAN22PipelineAdvanced(
                model_dir=os.path.dirname(self.config.model.transformer_high),
                device=self.config.memory.device,
                dtype=dtype
            )
            
            # Load LoRAs
            if self.config.model.lora_high:
                self.pipeline.load_with_lora(
                    lora_high_path=self.config.model.lora_high,
                    lora_low_path=self.config.model.lora_low,
                    lora_strength=1.0
                )
        else:
            print("Using standard pipeline...")
            self.pipeline = WAN22Pipeline(
                model_dir=os.path.dirname(self.config.model.transformer_high),
                device=self.config.memory.device,
                dtype=dtype,
                enable_offload=self.config.memory.enable_group_offload
            )
        
        # Setup memory management
        if self.config.memory.enable_group_offload:
            print("Enabling group offload...")
            if hasattr(self.pipeline, 'enable_group_offload'):
                self.pipeline.enable_group_offload()
        
        # Setup VAE tiling
        if self.config.memory.enable_vae_tiling:
            print("Enabling VAE tiling...")
            tiled_vae = TiledVAE(
                self.pipeline.vae,
                tile_size=self.config.memory.vae_tile_size,
                tile_overlap=self.config.memory.vae_tile_overlap
            )
            self.pipeline.vae_decode.vae = tiled_vae
        
        # Setup post-processing
        self.postprocessors = {}
        if self.config.postprocess.enable_interpolation:
            print("Loading RIFE interpolation...")
            self.postprocessors['rife'] = RIFEInterpolation(
                model_path=self.config.postprocess.interpolation_model,
                device=self.config.memory.device
            )
        
        self.monitor.end("initialization")
        print(f"✓ Initialization complete in {self.monitor.metrics['initialization']['duration']:.2f}s")
        
    def generate(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str = "",
        output_path: str = "output.mp4"
    ):
        """Generate video with full pipeline"""
        if self.pipeline is None:
            self.initialize()
        
        self.monitor.start("total_generation")
        
        # Prepare seed
        seed = self.config.generation.seed
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        print(f"\nUsing seed: {seed}")
        
        # Generation
        self.monitor.start("generation")
        
        if hasattr(self.pipeline, 'generate_4step') and self.config.generation.steps == 4:
            # Use 4-step LoRA generation
            frames = self.pipeline.generate_4step(
                image_path=image_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=self.config.generation.width,
                height=self.config.generation.height,
                num_frames=self.config.generation.num_frames,
                seed=seed,
                cfg=self.config.generation.cfg_scale
            )
        else:
            # Use standard generation
            frames = self.pipeline.generate(
                image_path=image_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=self.config.generation.width,
                height=self.config.generation.height,
                num_frames=self.config.generation.num_frames,
                seed=seed,
                steps=self.config.generation.steps,
                cfg=self.config.generation.cfg_scale
            )
        
        self.monitor.end("generation")
        
        # Post-processing
        frames = self._postprocess(frames)
        
        # Save video
        self.monitor.start("saving")
        fps = self.config.generation.fps
        if self.config.postprocess.enable_interpolation:
            fps *= self.config.postprocess.interpolation_multiplier
        
        self.pipeline.save_video(frames, output_path, fps=fps)
        self.monitor.end("saving")
        
        self.monitor.end("total_generation")
        
        # Print report
        self._print_report(output_path, len(frames), seed)
        
        return frames
    
    def _postprocess(self, frames):
        """Apply post-processing"""
        from wan22_utils import QualityEnhancer
        
        # Frame interpolation
        if self.config.postprocess.enable_interpolation:
            self.monitor.start("interpolation")
            print(f"\nApplying {self.config.postprocess.interpolation_multiplier}x interpolation...")
            rife = self.postprocessors.get('rife')
            if rife:
                frames = rife.interpolate(
                    frames,
                    multiplier=self.config.postprocess.interpolation_multiplier
                )
            self.monitor.end("interpolation")
        
        # Sharpening
        if self.config.postprocess.enable_sharpening:
            self.monitor.start("sharpening")
            print("Applying sharpening...")
            frames = QualityEnhancer.sharpen_frames(
                frames,
                strength=self.config.postprocess.sharpening_strength
            )
            self.monitor.end("sharpening")
        
        # Denoising
        if self.config.postprocess.enable_denoising:
            self.monitor.start("denoising")
            print("Applying denoising...")
            frames = QualityEnhancer.denoise_frames(
                frames,
                strength=self.config.postprocess.denoising_strength
            )
            self.monitor.end("denoising")
        
        return frames
    
    def _print_report(self, output_path, num_frames, seed):
        """Print generation report"""
        from wan22_utils import PerformanceMonitor
        
        print("\n" + "="*60)
        print("Generation Report")
        print("="*60)
        print(f"Output: {output_path}")
        print(f"Frames: {num_frames}")
        print(f"Resolution: {self.config.generation.width}x{self.config.generation.height}")
        print(f"Seed: {seed}")
        print(f"Steps: {self.config.generation.steps}")
        print(f"CFG Scale: {self.config.generation.cfg_scale}")
        
        # Timing breakdown
        print("\nTiming Breakdown:")
        for name, data in self.monitor.metrics.items():
            if 'duration' in data:
                print(f"  {name:.<30} {data['duration']:.2f}s")
        
        # Memory stats
        mem_stats = PerformanceMonitor.get_memory_stats()
        if mem_stats:
            print(f"\nVRAM Usage:")
            print(f"  Allocated: {mem_stats['allocated_gb']:.2f} GB")
            print(f"  Usage: {mem_stats['usage_percent']:.1f}%")
        
        print("="*60)


# ==================== COMMAND LINE INTERFACE ====================

def create_cli():
    """Create command line interface"""
    parser = argparse.ArgumentParser(
        description="WAN 2.2 Image-to-Video Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python inference.py -i input.jpg -p "A dragon flying" -o output.mp4
  
  # With config file
  python inference.py -i input.jpg -p "A dragon flying" --config config.yaml
  
  # Use preset
  python inference.py -i input.jpg -p "A dragon flying" --preset high_quality
  
  # Custom resolution and steps
  python inference.py -i input.jpg -p "A dragon flying" -w 1024 -h 1024 --steps 20
  
  # 4-step LoRA mode
  python inference.py -i input.jpg -p "A dragon flying" --preset fast_4step
        """
    )
    
    # Input/Output
    parser.add_argument('-i', '--image', required=True, help='Input image path')
    parser.add_argument('-p', '--prompt', required=True, help='Text prompt')
    parser.add_argument('-n', '--negative', default="", help='Negative prompt')
    parser.add_argument('-o', '--output', default='output.mp4', help='Output video path')
    
    # Configuration
    parser.add_argument('--config', help='Configuration file (YAML or JSON)')
    parser.add_argument('--preset', choices=['standard', 'high_quality', 'fast_4step', 'memory_efficient'],
                       help='Use preset configuration')
    parser.add_argument('--save-config', help='Save current config to file')
    
    # Generation parameters
    gen_group = parser.add_argument_group('Generation Parameters')
    gen_group.add_argument('-w', '--width', type=int, help='Video width')
    gen_group.add_argument('-H', '--height', type=int, help='Video height')
    gen_group.add_argument('-f', '--frames', type=int, help='Number of frames')
    gen_group.add_argument('-s', '--steps', type=int, help='Sampling steps')
    gen_group.add_argument('--cfg', type=float, help='CFG scale')
    gen_group.add_argument('--seed', type=int, help='Random seed')
    gen_group.add_argument('--fps', type=int, help='Output FPS')
    
    # Memory management
    mem_group = parser.add_argument_group('Memory Management')
    mem_group.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    mem_group.add_argument('--dtype', choices=['float16', 'float32', 'bfloat16'], help='Data type')
    mem_group.add_argument('--offload', action='store_true', help='Enable group offload')
    mem_group.add_argument('--tiling', action='store_true', help='Enable VAE tiling')
    
    # Post-processing
    post_group = parser.add_argument_group('Post-processing')
    post_group.add_argument('--interpolate', action='store_true', help='Enable frame interpolation')
    post_group.add_argument('--interpolate-x', type=int, default=2, help='Interpolation multiplier')
    post_group.add_argument('--sharpen', action='store_true', help='Enable sharpening')
    post_group.add_argument('--denoise', action='store_true', help='Enable denoising')
    
    return parser


def main():
    """Main entry point"""
    parser = create_cli()
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        # Load from file
        if args.config.endswith('.yaml') or args.config.endswith('.yml'):
            config = WAN22Config.from_yaml(args.config)
        elif args.config.endswith('.json'):
            config = WAN22Config.from_json(args.config)
        else:
            raise ValueError("Config file must be .yaml or .json")
    elif args.preset:
        # Use preset
        preset_map = {
            'standard': ConfigPresets.standard_quality,
            'high_quality': ConfigPresets.high_quality,
            'fast_4step': ConfigPresets.fast_4step,
            'memory_efficient': ConfigPresets.memory_efficient
        }
        config = preset_map[args.preset]()
    else:
        # Create default config
        config = ConfigPresets.standard_quality()
    
    # Override with command line arguments
    if args.width:
        config.generation.width = args.width
    if args.height:
        config.generation.height = args.height
    if args.frames:
        config.generation.num_frames = args.frames
    if args.steps:
        config.generation.steps = args.steps
    if args.cfg:
        config.generation.cfg_scale = args.cfg
    if args.seed:
        config.generation.seed = args.seed
    if args.fps:
        config.generation.fps = args.fps
    
    if args.device:
        config.memory.device = args.device
    if args.dtype:
        config.memory.dtype = args.dtype
    if args.offload:
        config.memory.enable_group_offload = True
    if args.tiling:
        config.memory.enable_vae_tiling = True
    
    if args.interpolate:
        config.postprocess.enable_interpolation = True
        config.postprocess.interpolation_multiplier = args.interpolate_x
    if args.sharpen:
        config.postprocess.enable_sharpening = True
    if args.denoise:
        config.postprocess.enable_denoising = True
    
    # Save config if requested
    if args.save_config:
        if args.save_config.endswith('.yaml') or args.save_config.endswith('.yml'):
            config.save_yaml(args.save_config)
        elif args.save_config.endswith('.json'):
            config.save_json(args.save_config)
        print(f"Configuration saved to {args.save_config}")
    
    # Initialize inference engine
    engine = WAN22Inference(config)
    
    # Generate video
    try:
        engine.generate(
            image_path=args.image,
            prompt=args.prompt,
            negative_prompt=args.negative,
            output_path=args.output
        )
        print(f"\n✓ Video generation successful!")
        print(f"✓ Output saved to: {args.output}")
        
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


# ==================== BATCH PROCESSING SCRIPT ====================

def batch_process_cli():
    """Batch processing from CSV or JSON"""
    parser = argparse.ArgumentParser(description="Batch process multiple videos")
    parser.add_argument('--input', required=True, help='Input CSV or JSON file')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--config', help='Configuration file')
    parser.add_argument('--preset', help='Configuration preset')
    
    args = parser.parse_args()
    
    # Load batch file
    import pandas as pd
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    elif args.input.endswith('.json'):
        df = pd.read_json(args.input)
    else:
        raise ValueError("Input must be CSV or JSON")
    
    # Load config
    if args.config:
        config = WAN22Config.from_yaml(args.config)
    elif args.preset:
        preset_map = {
            'standard': ConfigPresets.standard_quality,
            'high_quality': ConfigPresets.high_quality,
            'fast_4step': ConfigPresets.fast_4step,
            'memory_efficient': ConfigPresets.memory_efficient
        }
        config = preset_map[args.preset]()
    else:
        config = ConfigPresets.standard_quality()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize engine
    engine = WAN22Inference(config)
    engine.initialize()
    
    # Process batch
    results = []
    for idx, row in df.iterrows():
        print(f"\n{'='*60}")
        print(f"Processing {idx+1}/{len(df)}")
        print(f"{'='*60}")
        
        try:
            output_path = os.path.join(args.output_dir, f"video_{idx:04d}.mp4")
            
            engine.generate(
                image_path=row['image_path'],
                prompt=row['prompt'],
                negative_prompt=row.get('negative_prompt', ''),
                output_path=output_path
            )
            
            results.append({
                'index': idx,
                'status': 'success',
                'output': output_path
            })
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'index': idx,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Success: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
    print(f"{'='*60}")


# ==================== EXAMPLE CONFIGS ====================

def create_example_configs():
    """Create example configuration files"""
    
    # Standard config
    config = ConfigPresets.standard_quality()
    config.save_yaml("config_standard.yaml")
    config.save_json("config_standard.json")
    
    # High quality config
    config = ConfigPresets.high_quality()
    config.save_yaml("config_high_quality.yaml")
    
    # Fast 4-step config
    config = ConfigPresets.fast_4step()
    config.save_yaml("config_fast_4step.yaml")
    
    # Memory efficient config
    config = ConfigPresets.memory_efficient()
    config.save_yaml("config_memory_efficient.yaml")
    
    print("Example configuration files created:")
    print("  - config_standard.yaml")
    print("  - config_standard.json")
    print("  - config_high_quality.yaml")
    print("  - config_fast_4step.yaml")
    print("  - config_memory_efficient.yaml")


# ==================== ENTRY POINTS ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create-configs":
        create_example_configs()
    elif len(sys.argv) > 1 and sys.argv[1] == "batch":
        sys.argv.pop(1)  # Remove 'batch' argument
        batch_process_cli()
    else:
        sys.exit(main())
