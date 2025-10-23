from trajectory_processor import TrajectoryProcessor
import numpy as np

# Initialize processor
tp = TrajectoryProcessor(backend='ase')  # Force ASE backend

# Load trajectory
tp.load('dump.lammpstrj', fmt='lammpstrj')

# Process with 10 fs time step
tp.process(
    recompute_velocities=True,
    dt=10.0,  # 10 fs
    remove_com_motion=True,
    handle_charges=True
)

# Save in multiple formats
tp.save_hdf5('full_data.h5')  # All data in HDF5
tp.save_npz('compact_data.npz')  # Compact NPZ format
tp.save_extxyz('visualization.xyz',  # For visualization
               use_unwrapped=True,
               include_velocities=True,
               include_charges=True)

# Access processed data directly
print(f"Number of frames: {tp.batch.pos_unwrapped.shape[0]}")
print(f"Number of atoms: {tp.batch.pos_unwrapped.shape[1]}")
print(f"Displacements shape: {tp.batch.meta['displacements'].shape}")

# Check if charges were processed
if tp.batch.atom_charges_corrected is not None:
    print(f"Charges available: {tp.batch.atom_charges_corrected.shape}")
    
# Check velocities
if tp.batch.vel_corrected is not None:
    print(f"Velocities available: {tp.batch.vel_corrected.shape}")
    print(f"Average velocity magnitude: {np.mean(np.linalg.norm(tp.batch.vel_corrected, axis=2))}")
