import h5py
import numpy as np

# Load the processed HDF5 file to verify all data is there
with h5py.File('full_data.h5', 'r') as f:
    print("Available datasets:")
    for key in f.keys():
        print(f"  {key}: {f[key].shape}")
    
    # Access specific data
    unwrapped_positions = f['positions_unwrapped'][:]
    velocities = f['velocities'][:] if 'velocities' in f else None
    charges = f['charges_corrected'][:] if 'charges_corrected' in f else None
    displacements = f['displacements'][:] if 'displacements' in f else None
    
    print(f"\nUnwrapped positions shape: {unwrapped_positions.shape}")
    if velocities is not None:
        print(f"Velocities shape: {velocities.shape}")
    if charges is not None:
        print(f"Charges shape: {charges.shape}")
    if displacements is not None:
        print(f"Displacements shape: {displacements.shape}")
    
    # Print metadata
    print("\nMetadata:")
    for key, value in f.attrs.items():
        print(f"  {key}: {value}")
