import os
import numpy as np
from trajectory_processor import ASETrajectory
from _keys import (
    POSITIONS_KEY, PARTIAL_CHARGES_KEY, PBC_KEY, CELL_KEY, DISPLACEMENTS_KEY,UPDATE_PARTIAL_CHARGES_KEY
)

ROOT = "./" 
FILENAME = "dump.lammpstrj"
OUT_PREFIX = "traj_with_displacement"
FRAME_DT_FS = None

traj = ASETrajectory.read_from_file(
    root=ROOT,
    filename=FILENAME,
    wrapper="lammps",
    wrapper_kwargs={"lammps_units":"metal", "desired_units":"metal", "type_mapping":{1:14,2:8}},
    index=":",
    apply_unwrapping=False,
)
traj.compute_additional_fields(
    add_fields={DISPLACEMENTS_KEY,PARTIAL_CHARGES_KEY,UPDATE_PARTIAL_CHARGES_KEY},
    time_step=1,                   
    time_step_in_fs=64,           
    truncate=True
)

traj.write_to_file(
    root=ROOT,
    filename_prefix=OUT_PREFIX,
    chosen_fields={POSITIONS_KEY,PARTIAL_CHARGES_KEY, PBC_KEY, CELL_KEY, DISPLACEMENTS_KEY,UPDATE_PARTIAL_CHARGES_KEY},
    format="extxyz"
)
