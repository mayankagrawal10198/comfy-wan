"""
WAN 2.2 Utilities and Extensions
Additional components for the pipeline including:
- VAE Tiling for memory efficiency
- RIFE Frame Interpolation
- Advanced schedulers
- Batch processing
- Model patching utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional
from tqdm import tqdm
import gc


# ==================== VAE TILING ====================

class TiledVAE:
    """
    VAE with tiling support for processing large videos
    Critical for handling high-resolution content within memory limits
    """
    def __init__(self, vae, tile_size=512, tile_overlap=64):
        self.vae = vae
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        
    def _get_tiles(self, height, width):
        """Calculate tile positions"""
        tiles = []
        stride = self.tile_size - self.tile_overlap
        
        for h in range(0, height, stride):
            for w in range(0, width, stride):
                h_end = min(h + self.tile_size, height)
                w_end = min(w + self.tile_size, width)
                h_start = max(0, h_end - self.tile_size)
                w_start = max(0, w_end - self.tile_size)
                tiles.append((h_start, h_end, w_start, w_end))
                
        return tiles
        
    @torch.no_grad()
    def encode_tiled(self, x):
        """Encode with tiling"""
        B, C, T, H, W = x.shape
        tiles = self._get_tiles(H, W)
        
        # Process first tile to get latent dimensions
        h_start, h_end, w_start, w_end = tiles[0]
        tile = x[:, :, :, h_start:h_end, w_start:w_end]
        latent_tile = self.vae.encode(tile)
        
        # Calculate latent dimensions
        _, LC, LT, LH, LW = latent_tile.shape
        latent_h_ratio = LH / (h_end - h_start)
        latent_w_ratio = LW / (w_end - w_start)
        
        # Initialize output
        latent_h = int(H * latent_h_ratio)
        latent_w = int(W * latent_w_ratio)
        latent = torch.zeros((B, LC, LT, latent_h, latent_w), device=x.device)
        count = torch.zeros((B, LC, LT, latent_h, latent_w), device=x.device)
        
        # Process all tiles
        for h_start, h_end, w_start, w_end in tqdm(tiles, desc="Encoding tiles"):
            tile = x[:, :, :, h_start:h_end, w_start:w_end]
            latent_tile = self.vae.encode(tile)
            
            # Calculate latent positions
            lh_start = int(h_start * latent_h_ratio)
            lh_end = int(h_end * latent_h_ratio)
            lw_start = int(w_start * latent_w_ratio)
            lw_end = int(w_end * latent_w_ratio)
            
            # Accumulate with overlap blending
            latent[:, :, :, lh_start:lh_end, lw_start:lw_end] += latent_tile
            count[:, :, :, lh_start:lh_end, lw_start:lw_end] += 1
            
        # Average overlapping regions
        latent = latent / count.clamp(min=1)
        return latent
        
    @torch.no_grad()
    def decode_tiled(self, latent):
        """Decode with tiling - critical for video generation"""
        B, C, T, H, W = latent.shape
        
        # Decode frame by frame to save memory
        frames = []
        for t in tqdm(range(T), desc="Decoding frames"):
            frame_latent = latent[:, :, t:t+1, :, :]
            
            # Decode with spatial tiling if needed
            if H * W > (self.tile_size // 8) ** 2:
                frame = self._decode_frame_tiled(frame_latent)
            else:
                frame = self.vae.decode(frame_latent)
                
            frames.append(frame)
            
            # Clear cache
            torch.cuda.empty_cache()
            
        return torch.cat(frames, dim=2)
        
    def _decode_frame_tiled(self, frame_latent):
        """Decode single frame with spatial tiling"""
        B, C, T, H, W = frame_latent.shape
        tiles = self._get_tiles(H, W)
        
        # Get output dimensions
        h_start, h_end, w_start, w_end = tiles[0]
        tile = frame_latent[:, :, :, h_start:h_end, w_start:w_end]
        decoded_tile = self.vae.decode(tile)
        
        _, DC, DT, DH, DW = decoded_tile.shape
        out_h_ratio = DH / (h_end - h_start)
        out_w_ratio = DW / (w_end - w_start)
        
        out_h = int(H * out_h_ratio)
        out_w = int(W * out_w_ratio)
        output = torch.zeros((B, DC, DT, out_h, out_w), device=frame_latent.device)
        count = torch.zeros((B, DC, DT, out_h, out_w), device=frame_latent.device)
        
        for h_start, h_end, w_start, w_end in tiles:
            tile = frame_latent[:, :, :, h_start:h_end, w_start:w_end]
            decoded = self.vae.decode(tile)
            
            oh_start = int(h_start * out_h_ratio)
            oh_end = int(h_end * out_h_ratio)
            ow_start = int(w_start * out_w_ratio)
            ow_end = int(w_end * out_w_ratio)
            
            output[:, :, :, oh_start:oh_end, ow_start:ow_end] += decoded
            count[:, :, :, oh_start:oh_end, ow_start:ow_end] += 1
            
        output = output / count.clamp(min=1)
        return output


# ==================== FRAME INTERPOLATION (RIFE) ====================

class RIFEInterpolation:
    """
    RIFE (Real-Time Intermediate Flow Estimation) for frame interpolation
    Node 110 from ComfyUI workflow
    """
    def __init__(self, model_path="rife47.pth", device="cuda"):
        self.device = device
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """Load RIFE model"""
        try:
            from rife.RIFE_HDv3 import Model
            model = Model()
            model.load_model(model_path, -1)
            model.eval()
            model.device()
            return model
        except ImportError:
            print("Warning: RIFE not available. Install from https://github.com/megvii-research/ECCV2022-RIFE")
            return None
            
    @torch.no_grad()
    def interpolate(
        self,
        frames: List[np.ndarray],
        multiplier: int = 2,
        scale_factor: float = 1.0
    ):
        """
        Interpolate frames to increase FPS
        multiplier: Number of intermediate frames (2 = double FPS)
        """
        if self.model is None:
            print("RIFE model not available, returning original frames")
            return frames
            
        interpolated = []
        
        for i in tqdm(range(len(frames) - 1), desc=f"Interpolating {multiplier}x"):
            frame1 = self._prepare_frame(frames[i])
            frame2 = self._prepare_frame(frames[i + 1])
            
            # Add original frame
            interpolated.append(frames[i])
            
            # Generate intermediate frames
            for j in range(1, multiplier):
                timestep = j / multiplier
                
                # RIFE inference
                mid_frame = self.model.inference(frame1, frame2, timestep, scale_factor)
                mid_frame = self._postprocess_frame(mid_frame)
                
                interpolated.append(mid_frame)
                
        # Add last frame
        interpolated.append(frames[-1])
        
        return interpolated
        
    def _prepare_frame(self, frame):
        """Convert numpy frame to tensor"""
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return frame.to(self.device)
        
    def _postprocess_frame(self, tensor):
        """Convert tensor back to numpy"""
        frame = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return frame


# ==================== ADVANCED SCHEDULERS ====================

class NoiseScheduler:
    """Advanced noise schedulers for better quality"""
    
    @staticmethod
    def get_sigmas(steps: int, scheduler_type: str = "simple", shift: float = 8.0):
        """Get noise schedule based on type"""
        if scheduler_type == "simple":
            return NoiseScheduler.simple_schedule(steps, shift)
        elif scheduler_type == "karras":
            return NoiseScheduler.karras_schedule(steps)
        elif scheduler_type == "exponential":
            return NoiseScheduler.exponential_schedule(steps)
        elif scheduler_type == "sgm_uniform":
            return NoiseScheduler.sgm_uniform_schedule(steps, shift)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    @staticmethod
    def simple_schedule(steps: int, shift: float = 8.0):
        """Simple linear schedule with shift (default for WAN)"""
        sigmas = torch.linspace(1.0, 0.0, steps + 1)
        # Apply shift: sigma' = shift * sigma / (1 + (shift - 1) * sigma)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        return sigmas
        
    @staticmethod
    def karras_schedule(steps: int, sigma_min=0.0292, sigma_max=14.6146, rho=7.0):
        """Karras schedule from EDM paper"""
        ramp = torch.linspace(0, 1, steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        sigmas = torch.cat([sigmas, torch.zeros(1)])
        return sigmas
        
    @staticmethod
    def exponential_schedule(steps: int, sigma_min=0.0292, sigma_max=14.6146):
        """Exponential schedule"""
        sigmas = torch.exp(torch.linspace(
            np.log(sigma_max),
            np.log(sigma_min),
            steps
        ))
        sigmas = torch.cat([sigmas, torch.zeros(1)])
        return sigmas
        
    @staticmethod
    def sgm_uniform_schedule(steps: int, shift: float = 8.0):
        """SGM uniform schedule with shift"""
        timesteps = torch.linspace(1, 0, steps + 1)
        sigmas = timesteps * shift / (1 + (shift - 1) * timesteps)
        return sigmas


# ==================== BATCH PROCESSING ====================

class BatchProcessor:
    """Process multiple videos in batch"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def process_batch(
        self,
        image_paths: List[str],
        prompts: List[str],
        output_dir: str = "outputs",
        **kwargs
    ):
        """Process multiple images to videos"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, (image_path, prompt) in enumerate(zip(image_paths, prompts)):
            print(f"\n{'='*60}")
            print(f"Processing batch {i+1}/{len(image_paths)}")
            print(f"{'='*60}")
            
            try:
                frames = self.pipeline.generate(
                    image_path=image_path,
                    prompt=prompt,
                    **kwargs
                )
                
                output_path = f"{output_dir}/video_{i:04d}.mp4"
                self.pipeline.save_video(frames, output_path)
                
                results.append({
                    'success': True,
                    'output': output_path,
                    'frames': len(frames)
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'success': False,
                    'error': str(e)
                })
                
            # Clean up memory
            gc.collect()
            torch.cuda.empty_cache()
            
        return results


# ==================== MODEL PATCHING ====================

class ModelPatcher:
    """
    Utilities for patching models (LoRA, ControlNet, etc.)
    Based on ComfyUI's model_patcher.py
    """
    
    def __init__(self, model):
        self.model = model
        self.patches = {}
        self.backup = {}
        
    def add_patch(self, key: str, patch: torch.Tensor, strength: float = 1.0):
        """Add a patch to the model"""
        if key not in self.patches:
            self.patches[key] = []
        self.patches[key].append((patch, strength))
        
    def apply_patches(self):
        """Apply all patches to model"""
        for key, patches in self.patches.items():
            # Get original parameter
            param = self._get_parameter(self.model, key)
            if param is None:
                continue
                
            # Backup original
            if key not in self.backup:
                self.backup[key] = param.data.clone()
                
            # Apply patches
            modified = param.data.clone()
            for patch, strength in patches:
                modified = modified + patch * strength
                
            param.data = modified
            
    def unpatch(self):
        """Remove all patches"""
        for key, original in self.backup.items():
            param = self._get_parameter(self.model, key)
            if param is not None:
                param.data = original
                
        self.patches.clear()
        self.backup.clear()
        
    @staticmethod
    def _get_parameter(model, key):
        """Get parameter by key"""
        try:
            for name, param in model.named_parameters():
                if name == key:
                    return param
        except:
            pass
        return None


# ==================== CONDITIONING UTILITIES ====================

class ConditioningTools:
    """Tools for manipulating conditioning"""
    
    @staticmethod
    def concat_conditioning(cond1, cond2, axis=1):
        """Concatenate two conditioning tensors"""
        return torch.cat([cond1, cond2], dim=axis)
        
    @staticmethod
    def interpolate_conditioning(cond1, cond2, alpha=0.5):
        """Interpolate between two conditionings"""
        return cond1 * (1 - alpha) + cond2 * alpha
        
    @staticmethod
    def scale_conditioning(cond, scale=1.0):
        """Scale conditioning strength"""
        return cond * scale
        
    @staticmethod
    def add_motion_conditioning(cond, motion_vector):
        """Add motion information to conditioning"""
        # Placeholder - would add motion embeddings
        return cond


# ==================== QUALITY ENHANCEMENT ====================

class QualityEnhancer:
    """Post-processing for quality enhancement"""
    
    @staticmethod
    def sharpen_frames(frames: List[np.ndarray], strength: float = 0.5):
        """Apply sharpening to frames"""
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) * strength
        
        sharpened = []
        for frame in frames:
            sharp = cv2.filter2D(frame, -1, kernel)
            sharp = np.clip(sharp, 0, 255).astype(np.uint8)
            sharpened.append(sharp)
            
        return sharpened
        
    @staticmethod
    def denoise_frames(frames: List[np.ndarray], strength: int = 5):
        """Apply denoising"""
        denoised = []
        for frame in frames:
            clean = cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)
            denoised.append(clean)
        return denoised
        
    @staticmethod
    def color_correction(frames: List[np.ndarray], gamma: float = 1.0):
        """Apply color correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        
        corrected = []
        for frame in frames:
            corrected.append(cv2.LUT(frame, table))
        return corrected


# ==================== PERFORMANCE MONITORING ====================

class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        
    def start(self, name: str):
        """Start timing"""
        import time
        self.metrics[name] = {'start': time.time()}
        
    def end(self, name: str):
        """End timing"""
        import time
        if name in self.metrics:
            self.metrics[name]['end'] = time.time()
            self.metrics[name]['duration'] = self.metrics[name]['end'] - self.metrics[name]['start']
            
    def report(self):
        """Print performance report"""
        print("\n" + "="*60)
        print("Performance Report")
        print("="*60)
        for name, data in self.metrics.items():
            if 'duration' in data:
                print(f"{name:.<40} {data['duration']:.2f}s")
        print("="*60)
        
    @staticmethod
    def get_memory_stats():
        """Get current memory statistics"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'usage_percent': (allocated / total) * 100
            }
        return None


# ==================== USAGE EXAMPLES ====================

def example_tiled_vae():
    """Example: Using tiled VAE for high-res generation"""
    from wan22_pipeline import WAN22Pipeline
    
    pipeline = WAN22Pipeline(model_dir="models")
    
    # Wrap VAE with tiling
    tiled_vae = TiledVAE(pipeline.vae, tile_size=512, tile_overlap=64)
    pipeline.vae_decode.vae = tiled_vae
    
    # Now can handle larger resolutions
    frames = pipeline.generate(
        image_path="input.jpg",
        prompt="...",
        width=1920,  # High resolution
        height=1080,
        num_frames=81
    )
    

def example_frame_interpolation():
    """Example: Using RIFE to double FPS"""
    from wan22_pipeline import WAN22Pipeline
    
    pipeline = WAN22Pipeline(model_dir="models")
    
    # Generate at 16 FPS
    frames = pipeline.generate(
        image_path="input.jpg",
        prompt="...",
        num_frames=81
    )
    
    # Interpolate to 32 FPS
    rife = RIFEInterpolation(model_path="rife47.pth")
    frames_interpolated = rife.interpolate(frames, multiplier=2)
    
    # Save at higher FPS
    pipeline.save_video(frames_interpolated, "output_32fps.mp4", fps=32)


def example_batch_processing():
    """Example: Batch process multiple videos"""
    from wan22_pipeline import WAN22Pipeline
    
    pipeline = WAN22Pipeline(model_dir="models")
    processor = BatchProcessor(pipeline)
    
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    prompts = [
        "A dragon flying through clouds",
        "Ocean waves crashing on beach",
        "City lights at night"
    ]
    
    results = processor.process_batch(
        image_paths=image_paths,
        prompts=prompts,
        output_dir="batch_outputs",
        width=640,
        height=640,
        num_frames=81
    )
    
    print(f"Processed {len(results)} videos")


def example_custom_scheduler():
    """Example: Using custom noise scheduler"""
    from wan22_pipeline import KSamplerAdvanced
    
    # Create sampler with custom scheduler
    sampler = KSamplerAdvanced(model, device="cuda")
    
    # Get Karras schedule instead of simple
    sigmas = NoiseScheduler.get_sigmas(20, scheduler_type="karras")
    
    # Use in sampling...


if __name__ == "__main__":
    print("WAN 2.2 Utilities Module")
    print("Import these utilities in your main pipeline")
    print("\nAvailable utilities:")
    print("  - TiledVAE: Memory-efficient VAE processing")
    print("  - RIFEInterpolation: Frame interpolation")
    print("  - NoiseScheduler: Advanced scheduling")
    print("  - BatchProcessor: Batch video generation")
    print("  - ModelPatcher: Dynamic model patching")
    print("  - QualityEnhancer: Post-processing")
    print("  - PerformanceMonitor: Performance tracking")
