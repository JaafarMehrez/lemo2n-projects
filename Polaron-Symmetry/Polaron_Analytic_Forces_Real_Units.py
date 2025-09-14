"""
Author: Jaafar Mehrez, jaafarmehrez@sjtu.edu.cn
Polaron relaxation scan with analytic forces (Hellmann-Feynman)
Real-unit mapping example included (eV, Angstrom, amu)
"""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# Output directory
outdir = "data-Analytic-Polaron"
os.makedirs(outdir, exist_ok=True)


# Physical constants (for unit conversions)
eV_to_J = 1.602176634e-19
amu_to_kg = 1.66053906660e-27
angstrom_to_m = 1e-10
pi = np.pi

# Example mapping for m-HfO2 
# Choose vibrating species mass (amu)
mass_O_amu = 15.999
M_kg = mass_O_amu * amu_to_kg

# pick a representative optical phonon frequency (Hz). Example: 10 THz
f_thz = 10.0  # in THz
omega = 2.0 * pi * (f_thz * 1e12)  # rad/s

# spring constant in SI: K_SI = M * omega^2 (N/m)
K_SI = M_kg * omega**2

# convert to eV / Angstrom^2:
# Energy (J) = 0.5 * K_SI * (x_m)^2. For x in Angstroms: x_m = x_A * 1e-10
# effective K (eV/Å^2) = K_SI * 1e-20 (J/Å^2) / (1.602e-19 J/eV)
K_eV_per_A2 = K_SI * 1e-20 / eV_to_J

# choose hopping (eV) and a sample classical deformation potential g (eV/Angstrom)
t_eV = 1.0            # set an order-of-magnitude hopping energy in eV
g_example = 5.0       # eV per Angstrom, illustrative deformation potential

print("=== Example mapping (m-HfO2, ILLUSTRATIVE) ===")
print(f"Mass (O): {mass_O_amu} amu -> {M_kg:.3e} kg")
print(f"Chosen optical phonon freq: {f_thz} THz -> omega = {omega:.3e} rad/s")
print(f"K (SI) = M*omega^2 = {K_SI:.3e} N/m")
print(f"K (convert) = {K_eV_per_A2:.6f} eV/Å^2")
print(f"t = {t_eV} eV (chosen); example g = {g_example} eV/Å")
print("=============================================\n")

# Model functions (analytic gradient)
def tb_hamiltonian_from_onsite(onsite, t, pbc=False):
    Nloc = len(onsite)
    H = np.zeros((Nloc, Nloc), dtype=float)
    np.fill_diagonal(H, onsite)
    for i in range(Nloc - 1):
        H[i, i+1] = -t
        H[i+1, i] = -t
    if pbc and Nloc > 1:
        H[0, -1] = -t
        H[-1, 0] = -t
    return H

def electronic_groundstate_for_u(u, t, g, pbc=False):
    onsite = g * u  # eV (g in eV/Angstrom, u in Angstrom)
    H = tb_hamiltonian_from_onsite(onsite, t, pbc)
    vals, vecs = eigh(H)
    return float(vals[0]), vecs[:,0]

def total_energy_and_grad(u, t, g, K, pbc=False):
    """
    Returns:
      E_total = E_elec(u) + 0.5 * K * sum(u^2)
      grad_i = dE/du_i = g * |psi_i|^2 + K * u_i   (Hellmann-Feynman)
    Units: E in eV, u in Angstrom, g in eV/Angstrom, K in eV/Angstrom^2
    """
    e_elec, psi = electronic_groundstate_for_u(u, t, g, pbc)
    E_total = e_elec + 0.5 * K * np.sum(u**2)
    dens = np.abs(psi)**2
    grad = g * dens + K * u
    return E_total, grad

def energy_only(u, t, g, K, pbc=False):
    E, _ = total_energy_and_grad(u, t, g, K, pbc)
    return E

def energy_and_gradient_for_minimizer(u, t, g, K, pbc=False):
    E, grad = total_energy_and_grad(u, t, g, K, pbc)
    return E, grad

# Relaxation scan using analytic gradient (L-BFGS-B)
def relax_scan_analytic(N, t, K, g_values, pbc=False):
    u_prev = np.zeros(N)
    u_stars = np.zeros((len(g_values), N))
    ipr_vals = np.zeros(len(g_values))
    E_vals = np.zeros(len(g_values))
    psi_stars = np.zeros((len(g_values), N), dtype=complex)
    success = np.zeros(len(g_values), dtype=bool)

    for i, g in enumerate(g_values):
        res = minimize(lambda u: energy_only(u, t, g, K, pbc=pbc),
                       u_prev, method='L-BFGS-B',
                       jac=lambda u: energy_and_gradient_for_minimizer(u, t, g, K, pbc=pbc)[1],
                       options={'ftol':1e-12, 'gtol':1e-8, 'maxiter':1000})
        u_star = res.x
        E_star, grad = total_energy_and_grad(u_star, t, g, K, pbc=pbc)
        e_elec, psi = electronic_groundstate_for_u(u_star, t, g, pbc=pbc)
        psi = psi / np.linalg.norm(psi)
        ipr = float(np.sum(np.abs(psi)**4))

        u_stars[i,:] = u_star
        ipr_vals[i] = ipr
        E_vals[i] = E_star
        psi_stars[i,:] = psi
        success[i] = res.success

        u_prev = u_star  # continuation

    return {"g":np.array(g_values), "u_stars":u_stars, "ipr":ipr_vals, "E":E_vals, "psi":psi_stars, "success":success}


# Run the scan (real units)
N = 20
# choose a g grid in eV/Å: center around the illustrative g_example and g_c estimate
g_grid = np.linspace(0.0, 8.0, 41)  # eV/Å

res_obc = relax_scan_analytic(N=N, t=t_eV, K=K_eV_per_A2, g_values=g_grid, pbc=False)
res_pbc = relax_scan_analytic(N=N, t=t_eV, K=K_eV_per_A2, g_values=g_grid, pbc=True)

# Save data
np.savetxt(os.path.join(outdir,"mapping_info.txt"), 
           ["m_O_amu="+str(mass_O_amu), "f_thz="+str(f_thz), f"K_eV_per_A2={K_eV_per_A2:.6f}", f"t_eV={t_eV}", f"g_example={g_example}"], fmt="%s")
np.savetxt(os.path.join(outdir,"g_grid.csv"), g_grid, delimiter=",", header="g (eV/Å)", comments='')
np.savetxt(os.path.join(outdir,"ipr_obc.csv"), np.vstack([g_grid, res_obc['ipr']]).T, delimiter=",", header="g,IPR", comments='')
np.savetxt(os.path.join(outdir,"ipr_pbc.csv"), np.vstack([g_grid, res_pbc['ipr']]).T, delimiter=",", header="g,IPR", comments='')
np.savetxt(os.path.join(outdir,"u_stars_obc.csv"), res_obc['u_stars'], delimiter=",")
np.savetxt(os.path.join(outdir,"u_stars_pbc.csv"), res_pbc['u_stars'], delimiter=",")

# Plotting and saving figures
plt.rcParams.update({"font.size":10})

# IPR vs g
plt.figure(figsize=(6,4))
plt.plot(res_obc['g'], res_obc['ipr'], '-o', label='OBC (analytic grad)', color='black')
plt.plot(res_pbc['g'], res_pbc['ipr'], '-s', label='PBC (analytic grad)', color='blue')
plt.xlabel('g (eV/Å)', fontname="Georgia", fontsize=12)
plt.ylabel('IPR at relaxed u*', fontname="Georgia", fontsize=12)
plt.title(f'IPR vs g (N={N})',fontname="Georgia")
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0
)
plt.grid(False)
plt.tight_layout()
f_ipr = os.path.join(outdir,"ipr_vs_g_real_units.png")
plt.savefig(f_ipr, dpi=400); plt.close()

# sample u* plots (three representative g's: start, mid (max slope), end)
def pick_inds(ipr_arr):
    idx_min = 0; idx_max = len(ipr_arr)-1
    idx_mid = int(np.argmax(np.abs(np.gradient(ipr_arr))))
    return [idx_min, idx_mid, idx_max]

inds = pick_inds(res_obc['ipr'])
plt.figure(figsize=(8,6))
for ii, idx in enumerate(inds):
    plt.subplot(3,1,ii+1)
    plt.plot(np.arange(1,N+1), res_obc['u_stars'][idx], '-o', label=f'OBC u* (g={res_obc["g"][idx]:.3f})', color='black')
    plt.plot(np.arange(1,N+1), res_pbc['u_stars'][idx], '-s', label=f'PBC u* (g={res_pbc["g"][idx]:.3f})', color='blue')
    plt.ylabel('u* (Å)', fontname="Georgia", fontsize=12)
    plt.legend(
        prop={'family': 'Georgia', 'size': 10},
        frameon=True,
        edgecolor='black',
        framealpha=1.0
    )
    plt.grid(False)
    plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.xlabel('site', fontname="Georgia", fontsize=12)
plt.suptitle('Sample relaxed displacements u*(g)', fontname="Georgia", fontsize=12)
plt.tight_layout(rect=[0,0,1,0.96])
f_u = os.path.join(outdir,"sample_u_stars_real_units.png")
plt.savefig(f_u, dpi=400); plt.close()

print("Saved results in:", os.path.abspath(outdir))
print("Files: ", f_ipr, f_u)

