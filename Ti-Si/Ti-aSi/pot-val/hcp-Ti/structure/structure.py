from ase import Atoms
from ase.io import write
import numpy as np

# Standard HCP Ti parameters
a = 2.95
c = 4.68

# Define the Orthorhombic HCP unit cell (4 atoms)
# The box dimensions are: a, a*sqrt(3), c
cell = [a, a * np.sqrt(3), c]

# Scaled positions for a standard 4-atom orthorhombic HCP cell
positions = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.0, 1.0/3.0, 0.5],
    [0.5, 5.0/6.0, 0.5]
]

# Create the unit cell
ti_unit = Atoms('Ti4', scaled_positions=positions, cell=cell, pbc=True)

# Create supercell to reach ~2nm in each direction
# Unit cell is approx 2.95 x 5.11 x 4.68 Angstroms
# 7x4x5 reps results in: 20.65 x 20.44 x 23.40 Angstroms
reps = (7, 4, 5)
ti_block = ti_unit * reps

print(f"Structure generated: {len(ti_block)} atoms")
print(f"Cell Dimensions: {ti_block.get_cell().lengths()} Å")

# Save for LAMMPS
write('Ti_pure.data', ti_block, format='lammps-data')

# Save for GPUMD (extxyz)
write('Ti_pure.xyz', ti_block, format='extxyz')

