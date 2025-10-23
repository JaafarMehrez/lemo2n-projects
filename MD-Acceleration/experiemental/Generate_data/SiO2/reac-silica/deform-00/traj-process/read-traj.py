import os
import numpy as np
from trajcast.data.trajectory import ASETrajectory
from trajcast.data._keys import (
    POSITIONS_KEY, ATOMIC_NUMBERS_KEY, PBC_KEY, CELL_KEY, DISPLACEMENTS_KEY, PARTIAL_CHARGES_KEY, UPDATE_PARTIAL_CHARGES_KEY
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
    apply_unwrapping=False, # that would only work for trajectories generated from NVT ensemble
                            # if the trajectory is coming from NPT or deform, then the unwrapping will not work
                            # That would mean any property that depends on the coordinates might be questionalble
)


#try:
   # print("Available canonical fields:", traj.available_fields)
#except Exception:
    #print("Trajectory object keys/attributes:", [a for a in dir(traj) if not a.startswith("_")])


traj.compute_additional_fields(
    add_fields={DISPLACEMENTS_KEY, UPDATE_PARTIAL_CHARGES_KEY},
    time_step=1,                   
    time_step_in_fs=7,           
    truncate=True
)

traj.write_to_file(
    root=ROOT,
    filename_prefix=OUT_PREFIX,
    chosen_fields={POSITIONS_KEY, ATOMIC_NUMBERS_KEY, PBC_KEY, CELL_KEY, DISPLACEMENTS_KEY, PARTIAL_CHARGES_KEY, UPDATE_PARTIAL_CHARGES_KEY},
    format="extxyz"
)
