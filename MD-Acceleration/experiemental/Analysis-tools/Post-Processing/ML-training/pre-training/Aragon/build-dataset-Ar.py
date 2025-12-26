import ase.io
import random
import os

def sample_and_save(source_files, output_name, n_samples):
    print(f"Building {output_name}...")
    all_frames = []
    
    for f in source_files:
        print(f"  Reading {f}...") 
        frames = ase.io.read(f, index=":") 
        all_frames.extend(frames)
        
    print(f"  Total pool size: {len(all_frames)} frames.")
    
    if len(all_frames) < n_samples:
        print(f"  WARNING: Requested {n_samples}, but only have {len(all_frames)}.")
        n_samples = len(all_frames)
        
    selected_frames = random.sample(all_frames, n_samples)
    
    ase.io.write(output_name, selected_frames, format="extxyz")
    print(f"  -> Wrote {n_samples} frames to {output_name}")

train_sources = [
    "dataset_build/temp_production_run_22000.extxyz",
    "dataset_build/temp_production_run_24000.extxyz",
    "dataset_build/temp_production_run_38000.extxyz"
]

val_sources = [
    "dataset_build/temp_production_run_58000.extxyz"
]

test_sources = [
    "dataset_build/temp_production_run_61000.extxyz"
]

sample_and_save(train_sources, "train_argon_5fs.extxyz", 5000)
sample_and_save(val_sources,   "val_argon_5fs.extxyz",   1250)
sample_and_save(test_sources,  "test_argon_5fs.extxyz",  1000)
