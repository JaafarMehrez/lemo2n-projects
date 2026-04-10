"""
Script: Ti-aSi Interface Generator
Author: Jaafar Mehrez
Email:  jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
Date:   March 2026

Description:
    This script constructs a crystalline Titanium (HCP) / Amorphous Silicon (a-Si) 
    bilayer for Molecular Dynamics simulations. It utilizes high-quality 
    amorphous silicon models generated via ARTn-MTP [Zongo et al., Phys. Rev. B 
    111, 214209 (2025)] and scales them to study the low-temperature self-limited 
    silicidation process [Liao et al., IEEE Electron Device Lett. 46, 10 (2025)].

Key Features:
    - Transforms hexagonal HCP-Ti into an orthogonal representation for cubic cells.
    - Supports user-defined material thicknesses in nanometers (nm).
    - Optional "Cropping" feature to allow exact slicing of amorphous networks.
    - Automated structural analysis: Density (g/cm³), bond distances, and heights.
    - Exports to .xyz (visualization) and .data (LAMMPS input) formats.

Requirements: 
    - Python 3.x, NumPy, ASE (Atomic Simulation Environment)
"""

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list

# Target thickness in (nm) 
target_si_height_nm = 2.0
target_ti_height_nm = 3.0  

use_cropping = True        # TRUE: Slices material at exact nm height
                           # FALSE: Keeps whole blocks (rounds to nearest block)

interface_gap = 2.1        # Distance between materials (Angstroms)
input_asi_file = '1000.xyz' # this structure was adopted from 'https://gitlab.com/Kazongogit/MTPu/-/tree/main/ART/Si'

def get_material_stats(atoms, species_name, mass_amu):
    pos = atoms.get_positions()
    if len(pos) == 0: return 0, 0, 0, 0
    
    # Material height (top atom to bottom atom)
    height = np.max(pos[:, 2]) - np.min(pos[:, 2])
    
    # Density calculation
    cell = atoms.get_cell()
    area = np.linalg.det(cell[:2, :2])
    volume_a3 = area * height
    density = (len(atoms) * mass_amu * 1.66054) / volume_a3
    
    # Avg Bond Distance
    cutoff = 3.0 if species_name == 'Si' else 3.2
    i, j, d = neighbor_list('ijd', atoms, cutoff)
    avg_bond = np.mean(d) if len(d) > 0 else 0
    
    return height, density, len(atoms), avg_bond

# a-Si
asi_base = read(input_asi_file)
asi_base.wrap()
base_cell = asi_base.get_cell()
Lx, Ly, base_Lz = base_cell[0,0], base_cell[1,1], base_cell[2,2]
H_si_target_A = target_si_height_nm * 10.0

if use_cropping:
    # Tile and slice
    reps_z_si = int(np.ceil(H_si_target_A / base_Lz))
    asi_layer = asi_base * (1, 1, reps_z_si)
    # Align bottom to 0 
    asi_layer.translate([0, 0, -np.min(asi_layer.get_positions()[:, 2])])
    mask = asi_layer.get_positions()[:, 2] <= H_si_target_A
    asi_layer = asi_layer[mask]
else:
    # Just tile to the nearest whole block count
    reps_z_si = int(np.round(H_si_target_A / base_Lz))
    if reps_z_si < 1: reps_z_si = 1
    asi_layer = asi_base * (1, 1, reps_z_si)

# Ti layer (HCP Orthogonal)
a, c = 2.95, 4.68
ti_ortho_box = [a, a * np.sqrt(3), c]
ti_coords = [[0,0,0], [0.5,0.5,0], [0.5,1/6.,0.5], [0.0,2/3.,0.5]]
ti_unit = Atoms('Ti4', scaled_positions=ti_coords, cell=ti_ortho_box, pbc=True)
H_ti_target_A = target_ti_height_nm * 10.0

nx = int(np.round(Lx / ti_ortho_box[0]))
ny = int(np.round(Ly / ti_ortho_box[1]))
# preserve symmetry
nz = int(np.round(H_ti_target_A / c))
if nz < 1: nz = 1
ti_layer = ti_unit * (nx, ny, nz)

# Strain Ti to match Si X/Y
ti_layer.set_cell([Lx, Ly, ti_layer.get_cell()[2,2]], scale_atoms=True)

# Align and stack
# Center X/Y
asi_center = asi_layer.get_center_of_mass()[:2]
ti_center = ti_layer.get_center_of_mass()[:2]
ti_layer.translate([asi_center[0] - ti_center[0], asi_center[1] - ti_center[1], 0])

# Precise Z-Gap
max_z_si = np.max(asi_layer.get_positions()[:, 2])
min_z_ti = np.min(ti_layer.get_positions()[:, 2])
ti_layer.translate([0, 0, (max_z_si + interface_gap) - min_z_ti])

# Summary
h_si, dens_si, n_si, bond_si = get_material_stats(asi_layer, 'Si', 28.085)
h_ti, dens_ti, n_ti, bond_ti = get_material_stats(ti_layer, 'Ti', 47.867)

interface = asi_layer + ti_layer
interface.center(vacuum=10.0, axis=2)
interface.wrap()

print("\n" + "="*55)
print(f"{'INTERFACE BUILDER REPORT':^55}")
print(f"{'(Cropping: ' + str(use_cropping) + ')':^55}")
print("="*55)
print(f"Total Atoms: {len(interface)} ({n_si} Si, {n_ti} Ti)")
print(f"Cell Size:   {interface.get_cell()[0,0]:.2f} x {interface.get_cell()[1,1]:.2f} x {interface.get_cell()[2,2]:.2f} Å")
print("-" * 55)
print(f"AMORPHOUS SILICON (a-Si):")
print(f"  - Actual Height:    {h_si:.2f} Å (~{h_si/10:.2f} nm)")
print(f"  - Density:          {dens_si:.3f} g/cm³")
print(f"  - Avg Si-Si Bond:   {bond_si:.3f} Å")
print("-" * 55)
print(f"TITANIUM (HCP Ti):")
print(f"  - Actual Height:    {h_ti:.2f} Å (~{h_ti/10:.2f} nm)")
print(f"  - Density:          {dens_ti:.3f} g/cm³")
print(f"  - Avg Ti-Ti Bond:   {bond_ti:.3f} Å")
print("-" * 55)
print(f"INTERFACE:")
print(f"  - Set Gap:          {interface_gap} Å")
print("="*55)

write('Ti_aSi_Interface.xyz', interface)
write('Ti_aSi_Interface.data', interface, format='lammps-data')
