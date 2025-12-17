#!/usr/bin/env python3
"""
Build datasets from NVE trajectories and save in .extxyz format.
"""

import os
import numpy as np
import random
import ase.io

def preprocess_trajectory(traj_file, output_dir, dt_fs=5.0,
                         system_type="interface", frame_interval=2.5):
    """
    Preprocess a single NVE trajectory and save in .extxyz format.
    
    Args:
        traj_file: Path to LAMMPS trajectory file
        output_dir: Directory to save processed data
        dt_fs: Prediction horizon Δt in fs
        system_type: "interface", "silica",..etc
        frame_interval: Time between frames in trajectory (0.5 fs if dumped every step)
    """
    
    print(f"Preprocessing: {traj_file}")
    print(f"  Δt = {dt_fs} fs, frame interval = {frame_interval} fs")
    
    # Type mapping - ADJUST BASED ON YOUR SYSTEM!
    type_mapping = {
        1: 14,   # Si -> atomic number 14
        2:  8,   #  O -> atomic number  8
        3:  1,   #  H -> atomic number  1
    }
    
    # Import trajcast only for preprocessing
    from trajcast.data.trajectory import ASETrajectory
    from trajcast.data._keys import (
        POSITIONS_KEY, ATOMIC_NUMBERS_KEY, VELOCITIES_KEY,
        PBC_KEY, CELL_KEY, DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY
    )
    
    # Load trajectory
    traj = ASETrajectory.read_from_file(
        root=os.path.dirname(traj_file),
        filename=os.path.basename(traj_file),
        frame_interval=frame_interval,
        wrapper="lammps",
        wrapper_kwargs={
            "lammps_units": "real",
            "desired_units": "real",
            "type_mapping": type_mapping,
        },
        index=":",  # Read all frames
        apply_unwrapping=True,  # Important for periodic systems
    )
    
    print(f"  Loaded {traj.n_frames} frames")
    
    # Compute target variables (displacements and velocities after Δt)
    frames_to_forward = int(dt_fs / frame_interval)
    
    traj.compute_additional_fields(
        add_fields={DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY},
        time_step=frames_to_forward,
        truncate=True,  # Remove last frame without future
    )
    
    print(f"  After processing: {traj.n_frames} frames")
    print(f"  Available fields: {list(traj.available_fields)}")
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(traj_file))[0]
    
    output_file = os.path.join(output_dir, f"processed_{base_name}.extxyz")
    
    # Write using TrajCast's built-in method
    traj.write_to_file(
        root=output_dir,
        filename_prefix=f"processed_{base_name}",
        chosen_fields={
            POSITIONS_KEY,
            ATOMIC_NUMBERS_KEY,
            VELOCITIES_KEY,
            DISPLACEMENTS_KEY,
            UPDATE_VELOCITIES_KEY,
            CELL_KEY,
            PBC_KEY,
        },
        format="extxyz"
    )
    
    print(f"  ✓ Saved to {output_file}")
    
    # Also save metadata separately
    metadata = {
        'dt_fs': dt_fs,
        'frame_interval': frame_interval,
        'n_frames': traj.n_frames,
        'system_type': system_type,
        'source_file': traj_file
    }
    
    import json
    with open(os.path.join(output_dir, f"processed_{base_name}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
        
    return output_file


def sample_frames_from_extxyz(extxyz_files, n_samples, output_file, random_seed=42):
    """
    Sample frames from .extxyz files and save as a new .extxyz file.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"Sampling {n_samples} frames from {len(extxyz_files)} files...")
    
    # Load all frames from all files
    all_atoms = []
    
    for extxyz_file in extxyz_files:
        print(f"  Loading {extxyz_file}...")
        atoms_list = ase.io.read(extxyz_file, index=':')
        all_atoms.extend(atoms_list)
        print(f"    Frames: {len(atoms_list)}")
    
    total_frames = len(all_atoms)
    
    if n_samples > total_frames:
        print(f"Warning: Requested {n_samples} samples but only {total_frames} available")
        n_samples = total_frames
    
    # Sample random indices
    indices = random.sample(range(total_frames), n_samples)
    indices.sort()
    
    # Create sampled atoms list
    sampled_atoms = [all_atoms[i] for i in indices]
    
    # Write to output file
    ase.io.write(output_file, sampled_atoms, format='extxyz')
    
    print(f"  Selected {len(sampled_atoms)} frames")
    print(f"  ✓ Saved sampled dataset to {output_file}")
    
    # Save sampling info
    sampling_info = {
        'n_samples': len(sampled_atoms),
        'source_files': [os.path.basename(f) for f in extxyz_files],
        'random_seed': random_seed,
        'sampling_method': 'random_without_replacement'
    }
    
    import json
    info_file = output_file.replace('.extxyz', '_info.json')
    with open(info_file, 'w') as f:
        json.dump(sampling_info, f, indent=2)
    
    return output_file


def build_datasets_extxyz():
    """
    Main function to build datasets in .extxyz format.
    """
    SYSTEM = "interface"
    
    # Prediction horizon Δt
    if SYSTEM == "interface":
        DT_FS = 5.0  # fs
        N_TRAIN = 2000  # From paper: 5,000 for quartz
    else:
        raise ValueError(f"Unknown system: {SYSTEM}")
    
    N_VAL = N_TRAIN // 4  # Validation: 1/4 of training size
    N_TEST = 1000  # Fixed test set size
    
    # NVE trajectory files (adjust paths)
    NVE_TRAJECTORIES = [
        "nve_1.lammpstrj",
        "nve_2.lammpstrj",
        "nve_3.lammpstrj",
        "nve_4.lammpstrj",
        "nve_5.lammpstrj"
    ]
    
    # Assign trajectories to datasets
    TRAIN_FILES = NVE_TRAJECTORIES[:3]  # First 3 for training
    VAL_FILE = NVE_TRAJECTORIES[3]      # 4th for validation
    TEST_FILE = NVE_TRAJECTORIES[4]     # 5th for test
    
    # Output directories
    PROCESSED_DIR = "./processed_trajectories/"
    DATASETS_DIR = "./trajcast_datasets_extxyz/"
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    print("="*70)
    print(f"Building TrajCast Datasets for {SYSTEM} (EXTXYZ format)")
    print("="*70)
    print(f"Training: {len(TRAIN_FILES)} trajectories -> {N_TRAIN} samples")
    print(f"Validation: {VAL_FILE} -> {N_VAL} samples")
    print(f"Test: {TEST_FILE} -> {N_TEST} samples")
    print(f"Prediction horizon Δt: {DT_FS} fs")
    print("="*70)
    
    print("\nStep 1: Preprocessing trajectories to .extxyz format...")
    
    processed_train_files = []
    for traj_file in TRAIN_FILES:
        processed_file = preprocess_trajectory(
            traj_file, PROCESSED_DIR, DT_FS, SYSTEM, frame_interval=2.5
        )
        processed_train_files.append(processed_file)
        
    print(f"\nPreprocessing validation trajectory: {VAL_FILE}")
    processed_val_file = preprocess_trajectory(
        VAL_FILE, PROCESSED_DIR, DT_FS, SYSTEM, frame_interval=2.5
    )
    
    print(f"\nPreprocessing test trajectory: {TEST_FILE}")
    processed_test_file = preprocess_trajectory(
        TEST_FILE, PROCESSED_DIR, DT_FS, SYSTEM, frame_interval=2.5
    )
    
    print("\n" + "="*70)
    print("Step 2: Building training dataset...")

    train_output = os.path.join(DATASETS_DIR, f"{SYSTEM}_train.extxyz")
    sample_frames_from_extxyz(
        processed_train_files, N_TRAIN, train_output, random_seed=42
    )
    
    print("\n" + "="*70)
    print("Step 3: Building validation dataset...")

    val_output = os.path.join(DATASETS_DIR, f"{SYSTEM}_val.extxyz")
    sample_frames_from_extxyz(
        [processed_val_file], N_VAL, val_output, random_seed=43
    )
    
    print("\n" + "="*70)
    print("Step 4: Building test dataset...")

    test_output = os.path.join(DATASETS_DIR, f"{SYSTEM}_test.extxyz")
    sample_frames_from_extxyz(
        [processed_test_file], N_TEST, test_output, random_seed=44
    )
    
    print("\n" + "="*70)
    print("Dataset Summary")
    print("="*70)
    
    # Count frames in each dataset
    def count_frames(extxyz_file):
        frames = ase.io.read(extxyz_file, index=':')
        return len(frames)
    
    train_frames = count_frames(train_output)
    val_frames = count_frames(val_output)
    test_frames = count_frames(test_output)
    
    summary = f"""
Dataset Configuration for {SYSTEM} (EXTXYZ format):
------------------------------------------------
Prediction horizon (Δt): {DT_FS} fs

Training:
  Source trajectories: {TRAIN_FILES}
  Samples: {train_frames} (from paper: {N_TRAIN})
  File: {train_output}

Validation:
  Source trajectory: {VAL_FILE}
  Samples: {val_frames} (0.25 × training = {N_VAL})
  File: {val_output}

Test:
  Source trajectory: {TEST_FILE}
  Samples: {test_frames} (fixed from paper: {N_TEST})
  File: {test_output}

Dataset Files:
  {train_output}
  {val_output}
  {test_output}
"""

    print(summary)
    
    with open(os.path.join(DATASETS_DIR, "dataset_summary.txt"), "w") as f:
        f.write(summary)
        
    print(f"\nSummary saved to: {os.path.join(DATASETS_DIR, 'dataset_summary.txt')}")
    print("="*70)
    print("Dataset construction complete!")
    print("="*70)

def verify_extxyz_datasets():
    """Verify the created .extxyz datasets."""
    print("\n" + "="*70)
    print("Verifying EXTXYZ Datasets")
    print("="*70)
    DATASETS_DIR = "./trajcast_datasets_extxyz/"
    
    for system in ["interface"]:
        for split in ["train", "val", "test"]:
            file_path = os.path.join(DATASETS_DIR, f"{system}_{split}.extxyz")
            
            if os.path.exists(file_path):
                print(f"\nVerifying: {file_path}")
                
                # Load a few frames to verify
                try:
                    atoms_list = ase.io.read(file_path, index=':')
                    
                    print(f"  ✓ Loaded successfully")
                    print(f"  Frames: {len(atoms_list)}")
                    
                    if len(atoms_list) > 0:
                        first_atoms = atoms_list[0]
                        print(f"  Atoms per frame: {len(first_atoms)}")
                        
                        # Check what arrays are present
                        arrays = list(first_atoms.arrays.keys())
                        print(f"  Arrays present: {arrays}")
                        
                        # Check if velocities are stored as velocities or momenta
                        if 'velocities' in first_atoms.arrays:
                            print(f"  ✓ Velocities stored as 'velocities' array")
                        elif 'momenta' in first_atoms.arrays:
                            print(f"  Note: Velocities stored as 'momenta' (ASE convention)")
                            print(f"  To convert to velocities: velocities = momenta / mass")
                        
                        # Check required fields
                        required_arrays = ['displacements', 'update_velocities']
                        for arr in required_arrays:
                            if arr in arrays:
                                print(f"  ✓ {arr} present")
                            else:
                                print(f"  ✗ {arr} missing")
                                
                except Exception as e:
                    print(f"  ✗ Error loading: {e}")
                    
if __name__ == "__main__":
    build_datasets_extxyz()
    verify_extxyz_datasets()