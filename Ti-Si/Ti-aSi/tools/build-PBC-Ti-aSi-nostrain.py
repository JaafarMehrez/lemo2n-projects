"""
Ti-aSi Periodic multilayer generator (relaxed Ti).
Build relaxed Ti first, then scale a-Si to fit Ti.

Author: Jaafar Mehrez
Modified for: Stability of HCP phase
"""

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list

target_si_height_nm = 2.0
target_ti_height_nm = 3.0
interface_gap = 2.4        # (Angstroms)
input_asi_file = '1000.xyz' 
use_cropping = True         

# experiemental values
ti_a = 2.95
ti_c = 4.68

def get_material_stats(atoms, species_name, mass_amu):
    pos = atoms.get_positions()
    if len(pos) == 0: return 0, 0, 0, 0
    height = np.max(pos[:, 2]) - np.min(pos[:, 2])
    cell = atoms.get_cell()
    area = np.linalg.det(cell[:2, :2])
    volume_a3 = area * height
    density = (len(atoms) * mass_amu * 1.66054) / volume_a3
    return height, density, len(atoms)

# Ti layer
ti_ortho_box = [ti_a, ti_a * np.sqrt(3), ti_c]
ti_coords = [[0,0,0], [0.5,0.5,0], [0.5,1/6.,0.5], [0.0,2/3.,0.5]]
ti_unit = Atoms('Ti4', scaled_positions=ti_coords, cell=ti_ortho_box, pbc=True)

# Load a-Si base to see how many Ti repetitions we need to match the area
asi_base = read(input_asi_file)
Lx_target = asi_base.get_cell()[0,0]
Ly_target = asi_base.get_cell()[1,1]

# Find integer repetitions of Ti that get closest to a-Si dimensions
nx = int(np.round(Lx_target / ti_ortho_box[0]))
ny = int(np.round(Ly_target / ti_ortho_box[1]))
nz = max(1, int(np.round((target_ti_height_nm * 10.0) / ti_c)))

ti_layer = ti_unit * (nx, ny, nz)
Lx_final = ti_layer.get_cell()[0,0]
Ly_final = ti_layer.get_cell()[1,1]
h_ti_actual = np.max(ti_layer.get_positions()[:, 2]) - np.min(ti_layer.get_positions()[:, 2])

# a-Si layer (Scale it to match Ti)
base_Lz = asi_base.get_cell()[2,2]
H_si_target_A = target_si_height_nm * 10.0

if use_cropping:
    reps_z_si = int(np.ceil(H_si_target_A / base_Lz))
    asi_layer = asi_base * (1, 1, reps_z_si)
    asi_layer.translate([0, 0, -np.min(asi_layer.get_positions()[:, 2])])
    mask = asi_layer.get_positions()[:, 2] <= H_si_target_A
    asi_layer = asi_layer[mask]
else:
    reps_z_si = max(1, int(np.round(H_si_target_A / base_Lz)))
    asi_layer = asi_base * (1, 1, reps_z_si)

asi_layer.set_cell([Lx_final, Ly_final, asi_layer.get_cell()[2,2]], scale_atoms=True)
asi_layer.translate([0, 0, -np.min(asi_layer.get_positions()[:, 2])])
h_si_actual = np.max(asi_layer.get_positions()[:, 2]) - np.min(asi_layer.get_positions()[:, 2])

# Calculate the center of the box in X and Y
center_x = Lx_final / 2.0
center_y = Ly_final / 2.0

# Center a-Si in X and Y
asi_cm = asi_layer.get_center_of_mass()
asi_layer.translate([center_x - asi_cm[0], center_y - asi_cm[1], 0])
# Ensure a-Si starts at Z=0
asi_layer.translate([0, 0, -np.min(asi_layer.get_positions()[:, 2])])

# Center Ti in X and Y
ti_cm = ti_layer.get_center_of_mass()
ti_layer.translate([center_x - ti_cm[0], center_y - ti_cm[1], 0])
# Move Ti to be on top of Si with the gap
z_si_max = np.max(asi_layer.get_positions()[:, 2])
ti_layer.translate([0, 0, z_si_max + interface_gap - np.min(ti_layer.get_positions()[:,2])])

# Combine
interface = asi_layer + ti_layer

total_z_box = h_si_actual + h_ti_actual + (2 * interface_gap)
interface.set_cell([Lx_final, Ly_final, total_z_box])
interface.center(axis=2)
interface.pbc = True

# Wrap atoms so they all stay inside [0, L]
interface.wrap()

# Stats
h_si, dens_si, n_si = get_material_stats(asi_layer, 'Si', 28.085)
h_ti, dens_ti, n_ti = get_material_stats(ti_layer, 'Ti', 47.867)

print("\n" + "="*55)
print(f"{'RELAXED Ti MULTILAYER REPORT':^55}")
print("="*55)
print(f"Total Atoms: {len(interface)} ({n_si} Si, {n_ti} Ti)")
print(f"Cell Size:   {Lx_final:.2f} x {Ly_final:.2f} x {total_z_box:.2f} Å")
print("-" * 55)
print(f"Ti Layer:    {h_ti:.2f} Å (Relaxed HCP)")
print(f"a-Si Layer:  {h_si:.2f} Å (Scaled to Ti)")
print(f"a-Si Density change: {((dens_si/2.33)-1)*100:+.2f}% (relative to bulk Si)")
print("-" * 55)
print(f"Stacking:    [Bottom: a-Si] -> [Gap: {interface_gap}Å] -> [Top: Ti]")
print("="*55)

write('Ti_aSi.data', interface, format='lammps-data')
