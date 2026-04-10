"""
Ti-aSi Periodic multilayer generator (FIXED STACKING)

Author: Jaafar Mehrez
Email:  jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
Date:   March 2026
"""

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list

# Target thickness in (nm) 
target_si_height_nm = 2.0   
target_ti_height_nm = 3.0   

interface_gap = 1.75        # Distance between materials (Angstroms)
input_asi_file = '1000.xyz' # this structure was adopted from 'https://gitlab.com/Kazongogit/MTPu/-/tree/main/ART/Si'
use_cropping = True         # TRUE: Slices material at exact nm height
                            # FALSE: Keeps whole blocks (rounds to nearest block)         


def get_material_stats(atoms, species_name, mass_amu):
    pos = atoms.get_positions()
    if len(pos) == 0: return 0, 0, 0, 0
    height = np.max(pos[:, 2]) - np.min(pos[:, 2])
    cell = atoms.get_cell()
    area = np.linalg.det(cell[:2, :2])
    volume_a3 = area * height
    density = (len(atoms) * mass_amu * 1.66054) / volume_a3
    cutoff = 3.0 if species_name == 'Si' else 3.2
    i, j, d = neighbor_list('ijd', atoms, cutoff)
    avg_bond = np.mean(d) if len(d) > 0 else 0
    return height, density, len(atoms), avg_bond

# (a-Si)
asi_base = read(input_asi_file)
asi_base.wrap()
base_cell = asi_base.get_cell()
Lx, Ly, base_Lz = base_cell[0,0], base_cell[1,1], base_cell[2,2]
H_si_target_A = target_si_height_nm * 10.0

if use_cropping:
    reps_z_si = int(np.ceil(H_si_target_A / base_Lz))
    asi_layer = asi_base * (1, 1, reps_z_si)
    # Start Si at Z=0
    asi_layer.translate([0, 0, -np.min(asi_layer.get_positions()[:, 2])])
    mask = asi_layer.get_positions()[:, 2] <= H_si_target_A
    asi_layer = asi_layer[mask]
else:
    reps_z_si = max(1, int(np.round(H_si_target_A / base_Lz)))
    asi_layer = asi_base * (1, 1, reps_z_si)
    asi_layer.translate([0, 0, -np.min(asi_layer.get_positions()[:, 2])])

h_si_actual = np.max(asi_layer.get_positions()[:, 2]) - np.min(asi_layer.get_positions()[:, 2])

# Ti layer (HCP Orthogonal)
a, c = 2.95, 4.68
ti_ortho_box = [a, a * np.sqrt(3), c]
ti_coords = [[0,0,0], [0.5,0.5,0], [0.5,1/6.,0.5], [0.0,2/3.,0.5]]
ti_unit = Atoms('Ti4', scaled_positions=ti_coords, cell=ti_ortho_box, pbc=True)
H_ti_target_A = target_ti_height_nm * 10.0

nx = int(np.round(Lx / ti_ortho_box[0]))
ny = int(np.round(Ly / ti_ortho_box[1]))
nz = max(1, int(np.round(H_ti_target_A / c)))
ti_layer = ti_unit * (nx, ny, nz)

# Strain Ti to match Si X/Y
ti_layer.set_cell([Lx, Ly, ti_layer.get_cell()[2,2]], scale_atoms=True)
h_ti_actual = np.max(ti_layer.get_positions()[:, 2]) - np.min(ti_layer.get_positions()[:, 2])

# Align and stack
# Center X/Y
si_cm = asi_layer.get_center_of_mass()[:2]
ti_cm = ti_layer.get_center_of_mass()[:2]
ti_layer.translate([si_cm[0] - ti_cm[0], si_cm[1] - ti_cm[1], 0])

# Translate Ti vertically to be on top of Si with the gap
z_si_max = np.max(asi_layer.get_positions()[:, 2])
z_ti_min = np.min(ti_layer.get_positions()[:, 2])
# New Ti position = Si_top + gap
ti_layer.translate([0, 0, (z_si_max + interface_gap) - z_ti_min])

# periodic cell (NO VACUUM)
# Total Z box = height of both materials + two gaps (middle + boundary)
total_z_box = h_si_actual + h_ti_actual + (2 * interface_gap)

interface = asi_layer + ti_layer
interface.set_cell([Lx, Ly, total_z_box])

# Center the whole assembly in the box (distributes space to gaps)
interface.center(axis=2)
interface.pbc = [True, True, True]

h_si, dens_si, n_si, bond_si = get_material_stats(asi_layer, 'Si', 28.085)
h_ti, dens_ti, n_ti, bond_ti = get_material_stats(ti_layer, 'Ti', 47.867)

print("\n" + "="*55)
print(f"{'FIXED PERIODIC MULTILAYER REPORT':^55}")
print("="*55)
print(f"Total Atoms: {len(interface)} ({n_si} Si, {n_ti} Ti)")
print(f"Cell Size:   {interface.get_cell()[0,0]:.2f} x {interface.get_cell()[1,1]:.2f} x {interface.get_cell()[2,2]:.2f} Å")
print("-" * 55)
print(f"MATERIAL 1 (a-Si):  {h_si:.2f} Å height | Density: {dens_si:.3f} g/cm³")
print(f"MATERIAL 2 (Ti):    {h_ti:.2f} Å height | Density: {dens_ti:.3f} g/cm³")
print("-" * 55)
print(f"INTERFACES:")
print(f"  - Center Gap:      {interface_gap} Å")
print(f"  - Boundary Gap:    {interface_gap} Å (Periodic Z)")
print("="*55)

# Export
write('Ti_aSi_Multilayer.xyz', interface)
write('Ti_aSi_Multilayer.data', interface, format='lammps-data')
