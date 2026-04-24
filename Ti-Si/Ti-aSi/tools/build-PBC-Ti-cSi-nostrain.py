"""
Ti-cSi periodic multilayer generator

Builds a crystalline Si/hcp-Ti periodic multilayer with exact target
thicknesses along z.

Author: Jaafar Mehrez
Email:  jaafarmehrez@sjtu.edu.cn / jaafar@hpqc.org
Date:   April 2026
"""

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write

lateral_width_nm = 3.2     
target_si_height_nm = 2.0
target_ti_height_nm = 3.0

interface_gap = 2.4 #(Angstroms)

# Lattice Constants
ti_a = 2.95                # HCP Ti a
ti_c = 4.68                # HCP Ti c
si_a = 5.431               # Diamond Cubic Si a

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

nx_ti = int(np.round((lateral_width_nm * 10.0) / ti_ortho_box[0]))
ny_ti = int(np.round((lateral_width_nm * 10.0) / ti_ortho_box[1]))
nz_ti = max(1, int(np.round((target_ti_height_nm * 10.0) / ti_c)))

ti_layer = ti_unit * (nx_ti, ny_ti, nz_ti)
Lx_final = ti_layer.get_cell()[0,0]
Ly_final = ti_layer.get_cell()[1,1]
h_ti_actual = np.max(ti_layer.get_positions()[:, 2]) - np.min(ti_layer.get_positions()[:, 2])

# c-Si layer
si_unit = bulk('Si', 'diamond', a=si_a, cubic=True)

# We must match the Ti Lx and Ly to keep the interface periodic
nx_si = int(np.round(Lx_final / si_a))
ny_si = int(np.round(Ly_final / si_a))
nz_si = max(1, int(np.round((target_si_height_nm * 10.0) / si_a)))

si_layer = si_unit * (nx_si, ny_si, nz_si)

# Scale the crystalline Si X and Y to match Ti exactly
si_layer.set_cell([Lx_final, Ly_final, si_layer.get_cell()[2,2]], scale_atoms=True)
si_layer.translate([0, 0, -np.min(si_layer.get_positions()[:, 2])])
h_si_actual = np.max(si_layer.get_positions()[:, 2]) - np.min(si_layer.get_positions()[:, 2])

# Center for alignment
center_x = Lx_final / 2.0
center_y = Ly_final / 2.0

si_cm = si_layer.get_center_of_mass()
si_layer.translate([center_x - si_cm[0], center_y - si_cm[1], 0])

ti_cm = ti_layer.get_center_of_mass()
ti_layer.translate([center_x - ti_cm[0], center_y - ti_cm[1], 0])

# Move Ti to be on top of Si with the gap
z_si_max = np.max(si_layer.get_positions()[:, 2])
ti_layer.translate([0, 0, z_si_max + interface_gap - np.min(ti_layer.get_positions()[:,2])])

# Combine
interface = si_layer + ti_layer

# Final box setup
total_z_box = h_si_actual + h_ti_actual + (2 * interface_gap)
interface.set_cell([Lx_final, Ly_final, total_z_box])
interface.center(axis=2)
interface.pbc = True
interface.wrap()

# Stats
h_si, dens_si, n_si = get_material_stats(si_layer, 'Si', 28.085)
h_ti, dens_ti, n_ti = get_material_stats(ti_layer, 'Ti', 47.867)

print("\n" + "="*55)
print(f"{'CRYSTALLINE Ti-Si MULTILAYER REPORT':^55}")
print("="*55)
print(f"Total Atoms: {len(interface)} ({n_si} Si, {n_ti} Ti)")
print(f"Lateral (X/Y): {Lx_final:.2f} x {Ly_final:.2f} Å")
print(f"Total Height (Z): {total_z_box:.2f} Å")
print("-" * 55)
print(f"Ti Thickness:    {h_ti:.2f} Å")
print(f"c-Si Thickness:  {h_si_actual:.2f} Å")
print("-" * 55)
print(f"Stacking: [c-Si] -> [Gap: {interface_gap}Å] -> [Ti]")
print("="*55)

write('Ti_cSi.data', interface, format='lammps-data')
