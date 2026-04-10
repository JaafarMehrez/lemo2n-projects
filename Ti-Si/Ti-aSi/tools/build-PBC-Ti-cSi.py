"""
Ti-cSi periodic multilayer generator

Builds a crystalline Si/hcp-Ti periodic multilayer with exact target
thicknesses along z. The Ti layer is first tiled with the closest orthorhombic
hcp supercell and then strained in-plane to match the c-Si cell.

Author: Jaafar Mehrez
Email:  jaafarmehrez@sjtu.edu.cn / jaafar@hpqc.org
Date:   April 2026
"""

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write
from ase.neighborlist import neighbor_list

# Target thickness in nm
target_si_height_nm = 2.0
target_ti_height_nm = 2.0

target_lateral_size_a = 27.344798
max_si_repeats_xy = 10

interface_gap = 1.75  # Angstrom
use_cropping = True   # Slice slabs to the exact thickness along z

# Lattice parameters
si_a = 5.431          # Diamond cubic Si (Angstrom)
ti_a = 2.95           # HCP Ti (Angstrom)
ti_c = 4.68           # HCP Ti (Angstrom)


def get_material_stats(atoms, species_name, mass_amu):
    pos = atoms.get_positions()
    if len(pos) == 0:
        return 0, 0, 0, 0

    height = np.max(pos[:, 2]) - np.min(pos[:, 2])
    cell = atoms.get_cell()
    area = np.linalg.det(cell[:2, :2])
    volume_a3 = area * height
    density = (len(atoms) * mass_amu * 1.66054) / volume_a3
    cutoff = 3.0 if species_name == "Si" else 3.2
    _, _, d = neighbor_list("ijd", atoms, cutoff)
    avg_bond = np.mean(d) if len(d) > 0 else 0
    return height, density, len(atoms), avg_bond


def crop_slab_to_height(atoms, target_height_a):
    slab = atoms.copy()
    slab.translate([0, 0, -np.min(slab.get_positions()[:, 2])])
    mask = slab.get_positions()[:, 2] <= target_height_a
    slab = slab[mask]
    slab.translate([0, 0, -np.min(slab.get_positions()[:, 2])])
    return slab


def best_repeat(target_length_a, unit_length_a):
    ratio = target_length_a / unit_length_a
    candidates = {
        max(1, int(np.floor(ratio))),
        max(1, int(np.ceil(ratio))),
    }
    return min(candidates, key=lambda n: abs(target_length_a - (n * unit_length_a)))


def choose_inplane_match(target_length_a, si_a0, ti_ax, ti_ay, max_repeats):
    best = None
    best_score = None

    for nx_si in range(1, max_repeats + 1):
        for ny_si in range(1, max_repeats + 1):
            Lx = nx_si * si_a0
            Ly = ny_si * si_a0

            if Lx < 0.75 * target_length_a or Ly < 0.75 * target_length_a:
                continue
            if Lx > 1.8 * target_length_a or Ly > 1.8 * target_length_a:
                continue

            nx_ti = best_repeat(Lx, ti_ax)
            ny_ti = best_repeat(Ly, ti_ay)
            ti_Lx = nx_ti * ti_ax
            ti_Ly = ny_ti * ti_ay

            ex = (Lx - ti_Lx) / ti_Lx
            ey = (Ly - ti_Ly) / ti_Ly
            size_penalty = abs(Lx - target_length_a) / target_length_a + abs(Ly - target_length_a) / target_length_a
            aspect_penalty = abs(Lx - Ly) / target_length_a
            score = 8.0 * (abs(ex) + abs(ey)) + 0.8 * size_penalty + 0.4 * aspect_penalty

            if best_score is None or score < best_score:
                best_score = score
                best = {
                    "nx_si": nx_si,
                    "ny_si": ny_si,
                    "Lx": Lx,
                    "Ly": Ly,
                    "nx_ti": nx_ti,
                    "ny_ti": ny_ti,
                    "ti_raw_Lx": ti_Lx,
                    "ti_raw_Ly": ti_Ly,
                    "ex": ex,
                    "ey": ey,
                }

    if best is None:
        raise RuntimeError("Could not find a reasonable c-Si / Ti in-plane supercell match.")

    return best


# c-Si layer: conventional cubic diamond cell oriented along [001]
si_unit = bulk("Si", "diamond", a=si_a, cubic=True)
H_si_target_A = target_si_height_nm * 10.0
match = choose_inplane_match(
    target_lateral_size_a,
    si_a,
    ti_a,
    ti_a * np.sqrt(3.0),
    max_si_repeats_xy,
)
nx_si = match["nx_si"]
ny_si = match["ny_si"]
nz_si = int(np.ceil(H_si_target_A / si_a))
si_layer = si_unit * (nx_si, ny_si, nz_si)

if use_cropping:
    si_layer = crop_slab_to_height(si_layer, H_si_target_A)
else:
    si_layer.translate([0, 0, -np.min(si_layer.get_positions()[:, 2])])

Lx = match["Lx"]
Ly = match["Ly"]
h_si_actual = np.max(si_layer.get_positions()[:, 2]) - np.min(si_layer.get_positions()[:, 2])
si_layer.set_cell([Lx, Ly, h_si_actual], scale_atoms=False)
si_layer.pbc = [True, True, True]

# Ti layer: orthorhombic representation of hcp Ti
ti_ortho_box = [ti_a, ti_a * np.sqrt(3.0), ti_c]
ti_coords = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 1.0 / 6.0, 0.5],
    [0.0, 2.0 / 3.0, 0.5],
]
ti_unit = Atoms("Ti4", scaled_positions=ti_coords, cell=ti_ortho_box, pbc=True)

H_ti_target_A = target_ti_height_nm * 10.0
nx_ti = match["nx_ti"]
ny_ti = match["ny_ti"]
nz_ti = max(1, int(np.ceil(H_ti_target_A / ti_c)))

ti_layer = ti_unit * (nx_ti, ny_ti, nz_ti)
ti_raw_Lx = match["ti_raw_Lx"]
ti_raw_Ly = match["ti_raw_Ly"]

if use_cropping:
    ti_layer = crop_slab_to_height(ti_layer, H_ti_target_A)
else:
    ti_layer.translate([0, 0, -np.min(ti_layer.get_positions()[:, 2])])

# Strain Ti only in-plane. Keep z coordinates unchanged so the requested
# interface spacing is preserved exactly when stacking along z.
ti_positions = ti_layer.get_positions()
ti_positions[:, 0] *= Lx / ti_raw_Lx
ti_positions[:, 1] *= Ly / ti_raw_Ly
ti_layer.set_positions(ti_positions)

h_ti_actual = np.max(ti_layer.get_positions()[:, 2]) - np.min(ti_layer.get_positions()[:, 2])
ti_layer.set_cell([Lx, Ly, h_ti_actual], scale_atoms=False)
ti_layer.pbc = [True, True, True]

ti_strain_x_pct = 100.0 * match["ex"]
ti_strain_y_pct = 100.0 * match["ey"]

# Align and stack
si_cm = si_layer.get_center_of_mass()[:2]
ti_cm = ti_layer.get_center_of_mass()[:2]
ti_layer.translate([si_cm[0] - ti_cm[0], si_cm[1] - ti_cm[1], 0])

z_si_max = np.max(si_layer.get_positions()[:, 2])
z_ti_min = np.min(ti_layer.get_positions()[:, 2])
ti_layer.translate([0, 0, (z_si_max + interface_gap) - z_ti_min])

total_z_box = h_si_actual + h_ti_actual + (2.0 * interface_gap)
interface = si_layer + ti_layer
interface.set_cell([Lx, Ly, total_z_box])
interface.center(axis=2)
interface.pbc = [True, True, True]
interface.wrap()

h_si, dens_si, n_si, bond_si = get_material_stats(si_layer, "Si", 28.085)
h_ti, dens_ti, n_ti, bond_ti = get_material_stats(ti_layer, "Ti", 47.867)

print("\n" + "=" * 60)
print(f"{'PERIODIC c-Si / hcp-Ti MULTILAYER REPORT':^60}")
print("=" * 60)
print(f"Total Atoms: {len(interface)} ({n_si} Si, {n_ti} Ti)")
print(
    f"Cell Size:   {interface.get_cell()[0,0]:.2f} x "
    f"{interface.get_cell()[1,1]:.2f} x {interface.get_cell()[2,2]:.2f} A"
)
print("-" * 60)
print(f"Target lateral size: ~{target_lateral_size_a:.2f} A")
print(f"c-Si repeats (x, y, z before crop): ({nx_si}, {ny_si}, {nz_si})")
print(f"c-Si height: {h_si:.2f} A | Density: {dens_si:.3f} g/cm^3 | Avg bond: {bond_si:.3f} A")
print("-" * 60)
print(f"Ti repeats (x, y, z before crop):   ({nx_ti}, {ny_ti}, {nz_ti})")
print(f"Ti raw in-plane box: {ti_raw_Lx:.2f} x {ti_raw_Ly:.2f} A")
print(
    f"Ti in-plane strain: ex = {ti_strain_x_pct:+.2f}% | "
    f"ey = {ti_strain_y_pct:+.2f}%"
)
print(f"Ti height:  {h_ti:.2f} A | Density: {dens_ti:.3f} g/cm^3 | Avg bond: {bond_ti:.3f} A")
print("-" * 60)
print(f"Center gap:   {interface_gap:.2f} A")
print(f"Boundary gap: {interface_gap:.2f} A (Periodic Z)")
print("=" * 60)

write("Ti_cSi_Multilayer.xyz", interface)
write("Ti_cSi_Multilayer.data", interface, format="lammps-data")
