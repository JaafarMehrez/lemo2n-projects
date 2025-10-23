from trajectory_processor import TrajectoryProcessor

# Initialize the processor
tp = TrajectoryProcessor(backend='ase')

# Load a trajectory file (adjust the path and format as needed)
tp.load('dump.lammpstrj', fmt='lammps')

# Process the trajectory with a 10 fs time step
# This will:
# - Unwrap positions 
# - Compute displacements
# - Recompute velocities from unwrapped positions using 10 fs time step
# - Remove center-of-mass motion
# - Handle charges
tp.process(
    recompute_velocities=True,  # Recompute velocities from positions
    dt=10.0,                    # 10 femtosecond time step
    remove_com_motion=True,      # Remove center-of-mass motion
    handle_charges=True         # Process charges
)

# Save in HDF5 format (includes all data)
tp.save('processed_trajectory.h5', format='hdf5')

# Also save in extended XYZ format for visualization
tp.save('processed_trajectory.extxyz', format='extxyz', 
        use_unwrapped=True,      # Use unwrapped positions
        include_velocities=True, # Include velocities
        include_charges=True,    # Include charges
        include_box=True)        # Include box information

print("Processing complete!")
