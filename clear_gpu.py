#!/usr/bin/env python3
"""
Script to clear GPU memory and check memory status.
Run this if you need to free up GPU memory manually.
"""

import torch
import gc

def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    if torch.cuda.is_available():
        print("Clearing GPU memory...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✓ GPU memory cleared!")
    else:
        print("CUDA not available")

def check_gpu_memory():
    """Check available GPU memory and print status."""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3  # GB
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3  # GB
        free_memory = total_memory - allocated_memory
        
        print(f"\nGPU Memory Status:")
        print(f"  Total: {total_memory:.2f} GB")
        print(f"  Allocated: {allocated_memory:.2f} GB")
        print(f"  Cached: {cached_memory:.2f} GB")
        print(f"  Free: {free_memory:.2f} GB")
        print(f"  Usage: {(allocated_memory/total_memory)*100:.1f}%")
        
        return free_memory
    else:
        print("CUDA not available")
        return 0

if __name__ == "__main__":
    print("="*60)
    print("GPU Memory Management Tool")
    print("="*60)
    
    # Check before clearing
    print("\nBefore clearing:")
    check_gpu_memory()
    
    # Clear memory
    print()
    clear_gpu_memory()
    
    # Check after clearing
    print("\nAfter clearing:")
    check_gpu_memory()
    
    print("\n" + "="*60)
