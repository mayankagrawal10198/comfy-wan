"""
Update Script: Migrate to Official Wan2.2 Implementation
This script helps migrate from the custom implementation to the official Wan2.2 implementation
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import json


class Wan22OfficialUpdater:
    """Helper class for updating to official Wan2.2 implementation"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.wan22_dir = self.project_root / "Wan2.2"
        self.models_dir = self.project_root / "models"
        
    def check_requirements(self) -> Dict[str, bool]:
        """Check if all requirements are met"""
        requirements = {
            'wan22_repo': self.wan22_dir.exists(),
            'models_dir': self.models_dir.exists(),
            'python_packages': self._check_python_packages(),
            'cuda_available': self._check_cuda(),
            'sufficient_memory': self._check_memory()
        }
        return requirements
    
    def _check_python_packages(self) -> bool:
        """Check if required Python packages are installed"""
        required_packages = [
            'torch', 'torchvision', 'transformers', 'diffusers',
            'safetensors', 'einops', 'tqdm', 'PIL', 'numpy'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"Missing packages: {missing_packages}")
            return False
        return True
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _check_memory(self) -> bool:
        """Check if sufficient GPU memory is available"""
        try:
            import torch
            if torch.cuda.is_available():
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return memory_gb >= 8.0  # Minimum 8GB
            return False
        except:
            return False
    
    def setup_wan22_repo(self) -> bool:
        """Setup the official Wan2.2 repository"""
        if self.wan22_dir.exists():
            print("Wan2.2 repository already exists")
            return True
        
        print("Setting up Wan2.2 repository...")
        try:
            # Clone the repository
            subprocess.run([
                'git', 'clone', 
                'https://github.com/Wan-Video/Wan2.2.git',
                str(self.wan22_dir)
            ], check=True)
            
            # Install dependencies
            requirements_file = self.wan22_dir / "requirements.txt"
            if requirements_file.exists():
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
                ], check=True)
            
            print("✓ Wan2.2 repository setup complete")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Error setting up Wan2.2 repository: {e}")
            return False
    
    def setup_model_structure(self) -> bool:
        """Setup proper model directory structure"""
        print("Setting up model directory structure...")
        
        # Create model subdirectories
        subdirs = [
            'diffusion_models',
            'text_encoders', 
            'vae',
            'loras'
        ]
        
        for subdir in subdirs:
            (self.models_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        print("✓ Model directory structure created")
        return True
    
    def create_model_config(self) -> bool:
        """Create model configuration file"""
        print("Creating model configuration...")
        
        config = {
            "model_types": {
                "i2v-A14B": {
                    "description": "Image-to-Video A14B model",
                    "files": {
                        "text_encoder": "text_encoders/umt5_xxl_fp16.safetensors",
                        "vae": "vae/wan_2.1_vae.safetensors",
                        "low_noise_model": "diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
                        "high_noise_model": "diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
                    },
                    "lora_files": {
                        "low_noise_lora": "loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
                        "high_noise_lora": "loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
                    }
                }
            },
            "download_links": {
                "i2v-A14B": {
                    "huggingface": "https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B",
                    "modelscope": "https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B"
                }
            }
        }
        
        config_file = self.project_root / "model_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✓ Model configuration created")
        return True
    
    def create_migration_script(self) -> bool:
        """Create migration script for existing code"""
        print("Creating migration script...")
        
        migration_script = """#!/usr/bin/env python3
\"\"\"
Migration Script: Update existing code to use official Wan2.2 implementation
\"\"\"

import os
import sys
from pathlib import Path

def migrate_imports():
    \"\"\"Update import statements\"\"\"
    print("Updating import statements...")
    
    # Add Wan2.2 to Python path
    wan22_path = Path(__file__).parent / "Wan2.2"
    if wan22_path.exists():
        sys.path.insert(0, str(wan22_path))
        print("✓ Added Wan2.2 to Python path")
    else:
        print("✗ Wan2.2 directory not found")

def migrate_pipeline_usage():
    \"\"\"Show how to migrate pipeline usage\"\"\"
    print("\\nMigration Guide:")
    print("="*50)
    print("OLD (Custom Implementation):")
    print("from wan22_pipeline import WAN22Pipeline")
    print("pipeline = WAN22Pipeline(model_dir='./models')")
    print("video = pipeline.generate(image_path='input.jpg', prompt='A dragon')")
    print()
    print("NEW (Official Implementation):")
    print("from wan22_official_pipeline import OfficialWan22Pipeline")
    print("pipeline = OfficialWan22Pipeline(model_dir='./models', model_type='i2v-A14B')")
    print("video = pipeline.generate(image_path='input.jpg', prompt='A dragon')")
    print("="*50)

if __name__ == "__main__":
    migrate_imports()
    migrate_pipeline_usage()
"""
        
        script_path = self.project_root / "migrate_to_official.py"
        with open(script_path, 'w') as f:
            f.write(migration_script)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        print("✓ Migration script created")
        return True
    
    def create_example_usage(self) -> bool:
        """Create example usage scripts"""
        print("Creating example usage scripts...")
        
        # Basic usage example
        basic_example = """#!/usr/bin/env python3
\"\"\"
Basic Usage Example: Official Wan2.2 Implementation
\"\"\"

from wan22_official_pipeline import OfficialWan22Pipeline

def main():
    # Initialize pipeline
    pipeline = OfficialWan22Pipeline(
        model_dir="./models",
        model_type="i2v-A14B",
        device="cuda",
        offload_model=True
    )
    
    # Generate video
    video = pipeline.generate(
        image_path="input.jpg",
        prompt="A dragon flying through the sky",
        negative_prompt="blurry, low quality",
        width=1280,
        height=720,
        num_frames=81,
        steps=40,
        cfg_scale=3.5,
        seed=42,
        output_path="output.mp4"
    )
    
    print("Video generation complete!")

if __name__ == "__main__":
    main()
"""
        
        example_path = self.project_root / "example_official_usage.py"
        with open(example_path, 'w') as f:
            f.write(basic_example)
        
        # Make executable
        os.chmod(example_path, 0o755)
        
        print("✓ Example usage script created")
        return True
    
    def create_comparison_guide(self) -> bool:
        """Create comparison guide between old and new implementations"""
        print("Creating comparison guide...")
        
        comparison_guide = """# Wan2.2 Implementation Comparison Guide

## Key Differences

### 1. Architecture
- **Old Implementation**: Custom architecture with simplified components
- **New Implementation**: Official MoE (Mixture of Experts) architecture with separate high-noise and low-noise models

### 2. Model Loading
- **Old**: Single model loading with custom components
- **New**: Proper model loading with official WanModel class and MoE switching

### 3. VAE Implementation
- **Old**: Simplified VAE implementation
- **New**: Official Wan2.1 VAE for A14B models, Wan2.2 VAE for 5B models

### 4. Attention Mechanism
- **Old**: Basic attention implementation
- **New**: RoPE (Rotary Position Embedding), RMSNorm, and proper modulation

### 5. Schedulers
- **Old**: Basic schedulers
- **New**: Official Flow Matching schedulers (UniPC, DPM++)

## Migration Steps

### Step 1: Install Official Repository
```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
```

### Step 2: Update Imports
```python
# Old
from wan22_pipeline import WAN22Pipeline

# New
from wan22_official_pipeline import OfficialWan22Pipeline
```

### Step 3: Update Pipeline Initialization
```python
# Old
pipeline = WAN22Pipeline(model_dir="./models")

# New
pipeline = OfficialWan22Pipeline(
    model_dir="./models",
    model_type="i2v-A14B",  # Specify model type
    device="cuda",
    offload_model=True
)
```

### Step 4: Update Generation Calls
```python
# Old
video = pipeline.generate(
    image_path="input.jpg",
    prompt="A dragon",
    width=1280,
    height=720,
    steps=20
)

# New
video = pipeline.generate(
    image_path="input.jpg",
    prompt="A dragon",
    width=1280,
    height=720,
    num_frames=81,  # More explicit
    steps=40,       # Official default
    cfg_scale=3.5,  # More explicit
    seed=42,        # More explicit
    sample_solver="unipc"  # New parameter
)
```

## Benefits of Official Implementation

1. **Better Quality**: Official MoE architecture provides better results
2. **Proper Memory Management**: Better GPU memory usage with model offloading
3. **Multiple Model Support**: Support for T2V, I2V, and TI2V models
4. **Official Schedulers**: Proper flow matching schedulers
5. **Distributed Training**: Support for FSDP and sequence parallel
6. **Regular Updates**: Access to official updates and improvements

## Performance Comparison

| Aspect | Old Implementation | New Implementation |
|--------|-------------------|-------------------|
| Quality | Good | Excellent |
| Memory Usage | High | Optimized |
| Speed | Fast | Similar |
| Flexibility | Limited | High |
| Maintenance | Custom | Official |

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure Wan2.2 repository is properly installed
2. **Model Loading**: Check that all model files are in correct locations
3. **Memory Issues**: Use model offloading and dtype conversion
4. **CUDA Issues**: Ensure proper CUDA installation and compatibility

### Solutions

1. **Install Dependencies**: `pip install -r Wan2.2/requirements.txt`
2. **Check Model Files**: Verify all required model files are present
3. **Use Auto-Config**: Use `--auto-config` flag for automatic configuration
4. **Memory Management**: Enable `--offload` and `--convert-dtype` flags
"""
        
        guide_path = self.project_root / "MIGRATION_GUIDE.md"
        with open(guide_path, 'w') as f:
            f.write(comparison_guide)
        
        print("✓ Comparison guide created")
        return True
    
    def run_full_update(self) -> bool:
        """Run complete update process"""
        print("Starting Wan2.2 Official Implementation Update")
        print("="*60)
        
        # Check requirements
        requirements = self.check_requirements()
        print("Requirements Check:")
        for req, status in requirements.items():
            status_str = "✓" if status else "✗"
            print(f"  {status_str} {req}")
        
        if not all(requirements.values()):
            print("\\nSome requirements are not met. Please address them before continuing.")
            return False
        
        # Setup steps
        steps = [
            ("Setting up Wan2.2 repository", self.setup_wan22_repo),
            ("Setting up model structure", self.setup_model_structure),
            ("Creating model configuration", self.create_model_config),
            ("Creating migration script", self.create_migration_script),
            ("Creating example usage", self.create_example_usage),
            ("Creating comparison guide", self.create_comparison_guide)
        ]
        
        for step_name, step_func in steps:
            print(f"\\n{step_name}...")
            if not step_func():
                print(f"✗ Failed: {step_name}")
                return False
            print(f"✓ Completed: {step_name}")
        
        print("\\n" + "="*60)
        print("✓ Wan2.2 Official Implementation Update Complete!")
        print("="*60)
        print("\\nNext Steps:")
        print("1. Download the required model files")
        print("2. Update your code to use the new implementation")
        print("3. Test with the example scripts")
        print("4. Refer to MIGRATION_GUIDE.md for detailed instructions")
        
        return True


def main():
    """Main entry point"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    updater = Wan22OfficialUpdater(project_root)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        # Just check requirements
        requirements = updater.check_requirements()
        print("Requirements Check:")
        for req, status in requirements.items():
            status_str = "✓" if status else "✗"
            print(f"  {status_str} {req}")
        return 0 if all(requirements.values()) else 1
    
    # Run full update
    success = updater.run_full_update()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
