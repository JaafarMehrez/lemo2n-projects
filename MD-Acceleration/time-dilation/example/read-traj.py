import os
import numpy as np
from trajectory_processor import ASETrajectory
from _keys import (
    POSITIONS_KEY, ATOMIC_NUMBERS_KEY, VELOCITIES_KEY, PARTIAL_CHARGES_KEY, PBC_KEY, CELL_KEY, DISPLACEMENTS_KEY, 
    UPDATE_PARTIAL_CHARGES_KEY, UPDATE_VELOCITIES_KEY
)


time_between_frames = 50  # time interval between frames in fs
timestep_forward = 100 # forward timestep

ROOT = "./data/" 
FILENAME = "dump-100.lammpstrj"
OUT_PREFIX = "traj_with_displacement"
FRAME_DT_FS = None

traj = ASETrajectory.read_from_file(
    root=ROOT,
    filename=FILENAME,
    frame_interval = time_between_frames,
    wrapper="lammps",
    wrapper_kwargs={"lammps_units":"real", "desired_units":"real", "type_mapping":{1:14,2:8}},
    index=":",
    apply_unwrapping=False,
)
traj.compute_additional_fields(
    add_fields={DISPLACEMENTS_KEY, PARTIAL_CHARGES_KEY, UPDATE_PARTIAL_CHARGES_KEY, UPDATE_VELOCITIES_KEY},
    time_step=int(timestep_forward / time_between_frames),
    truncate=True,
)

print(traj.available_fields)

traj.write_to_file(
    root=ROOT,
    filename_prefix=OUT_PREFIX,
    chosen_fields={POSITIONS_KEY, ATOMIC_NUMBERS_KEY, PARTIAL_CHARGES_KEY, PBC_KEY, CELL_KEY, 
                   VELOCITIES_KEY, DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY, UPDATE_PARTIAL_CHARGES_KEY},
    format="extxyz"
)


# if you chaoose timestep_forward  < time_between_frames, you will get the following error
'''
The error happens in compute_atomic_displacement_vectors()
ValueError: shapes (0,) and (3,3) not aligned: 0 (dim 0) != 3 (dim 0)


- You should also be careful when you choose time_between_frames and timestep_forward
'''