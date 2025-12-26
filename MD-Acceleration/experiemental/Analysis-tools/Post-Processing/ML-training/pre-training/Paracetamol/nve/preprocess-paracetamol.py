#!/usr/bin/env python3
"""
Step 1: Convert raw LAMMPS trajectories to .extxyz with calculated displacements and updated velocities.
"""

import os
import glob
from trajcast.data.trajectory import ASETrajectory
from trajcast.data._keys import (
    POSITIONS_KEY, ATOMIC_NUMBERS_KEY, VELOCITIES_KEY,
    PBC_KEY, CELL_KEY, DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY
)

def preprocess_trajectory(traj_file, output_dir, dt_fs=5.0, frame_interval=0.5):
    """
    Args:
        traj_file: Path to LAMMPS dump
        dt_fs: Prediction horizon
        frame_interval: Time between dump frames in fs
    """ 
    base_name = os.path.splitext(os.path.basename(traj_file))[0]
    print(f"--- Processing {base_name} ---")
    traj = ASETrajectory.read_from_file(
        root=os.path.dirname(traj_file),
        filename=os.path.basename(traj_file),
        frame_interval=frame_interval,
        wrapper="lammps",
        wrapper_kwargs={
            "lammps_units": "real", 
            "desired_units": "real",
        },
        index=":", # Read all frames
        apply_unwrapping=True, 
    )

    frames_to_forward = int(dt_fs / frame_interval)
    
    print(f"  Computing targets for dt={dt_fs}fs ({frames_to_forward} frames ahead)...")
    
    traj.compute_additional_fields(
        add_fields={DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY},
        time_step=frames_to_forward,
        truncate=True, 
    )

    output_file = os.path.join(output_dir, f"temp_{base_name}.extxyz")
    
    traj.write_to_file(
        root=output_dir,
        filename_prefix=f"temp_{base_name}",
        chosen_fields={
            POSITIONS_KEY, ATOMIC_NUMBERS_KEY, VELOCITIES_KEY,
            DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY,
            CELL_KEY, PBC_KEY,
        },
        format="extxyz"
    )
    print(f"  Saved full processed traj to {output_file}")
    return output_file

if __name__ == "__main__":
    os.makedirs("dataset_build", exist_ok=True)
    files = glob.glob("fixed_production_run_*.lammpstrj")
    for f in files:
        preprocess_trajectory(f, "dataset_build", dt_fs=5.0, frame_interval=0.5)
