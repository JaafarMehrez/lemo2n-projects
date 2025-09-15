"""
Author: Jaafar Mehrez, jaafarmehrez@sjtu.edu.cn
Polaron IPR vs g (relaxed lattice) demo
- Tight-binding 1D chain, one electron
- Onsite electron-phonon coupling (Holstein-like, adiabatic limit)
- Minimize total energy E(u) = E_elec(u) + 0.5 K sum_i u_i^2 at each g
- Compute IPR = sum_i |psi_i|^4 of the electronic ground-state at u^*(g)
- Treat both OBC and PBC.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# ---------------------------
# Model / numerical settings
# ---------------------------
N = 50                    # number of lattice sites
t = 1.0                   # hopping amplitude
K = 1.0                   # spring constant for classical phonons
g_min, g_max = 0.0, 6.0   # scan range for electron-phonon coupling g
n_g = 81                  # number of g points
g_values = np.linspace(g_min, g_max, n_g)

# Minimizer settings
minimizer_method = "BFGS"   # good default for smooth problems; can try "L-BFGS-B"
min_opts = {"gtol": 1e-8, "maxiter": 2000}

# Continuation and initial perturbation
initial_noise = 1e-4       # tiny random seed for first g to break exact symmetry if desired
delta_scale = 1e-3         # small amplitude when seeding a symmetry-breaking perturbation (if you want)

# Output paths
outdir = "data-Polaron-IPR"
os.makedirs(outdir, exist_ok=True)

# ---------------------------
# Helper functions
# ---------------------------
def tb_hamiltonian_from_onsite(onsite, t, pbc=False):
    """
    Construct TB Hamiltonian for a 1D chain.
    Parameter 'onsite' should be an array of length N that already contains the onsite energies
    (for our model we will pass onsite = g * u).
    """
    Nloc = len(onsite)
    H = np.zeros((Nloc, Nloc), dtype=float)
    # onsite
    np.fill_diagonal(H, onsite)
    # nearest-neighbour hopping
    for i in range(Nloc - 1):
        H[i, i+1] = -t
        H[i+1, i] = -t
    if pbc and Nloc > 1:
        H[0, -1] = -t
        H[-1, 0] = -t
    return H

def electronic_groundstate_for_u(u, t, g, pbc=False):
    """
    For given displacement vector u (length N), return the lowest eigenvalue and eigenvector
    of the Hamiltonian H = -t sum(c_i^\dagger c_{i+1} + h.c.) + sum_i (g u_i) n_i.
    """
    onsite = g * u  # onsite energies
    H = tb_hamiltonian_from_onsite(onsite, t, pbc)
    vals, vecs = eigh(H)
    return float(vals[0]), vecs[:, 0]   # ground-state energy and wavefunction (lowest eigenpair)

def total_energy(u, t, g, K, pbc=False):
    """Adiabatic total energy: electronic ground-state energy + 0.5 * K * sum u_i^2"""
    e_elec, _ = electronic_groundstate_for_u(u, t, g, pbc=pbc)
    return e_elec + 0.5 * K * np.sum(u**2)

def compute_ipr_from_wavefunction(psi):
    """Inverse participation ratio: sum_i |psi_i|^4 (psi assumed normalized)"""
    p = np.abs(psi)**2
    return float(np.sum(p**2))

# ---------------------------
# Main routine: relax for many g with continuation
# ---------------------------
def relax_scan(N, t, K, g_values, pbc=False,
                init_noise=1e-4, delta_scale=1e-3,
                minimizer_method="BFGS", min_opts=None):
    """
    For each g in g_values:
      - minimize total_energy(u) starting from previous u_star (continuation)
      - compute electronic ground state at u_star and its IPR
    Returns dict with arrays: u_stars (n_g x N), ipr (n_g), E_star (n_g), success flags
    """
    if min_opts is None:
        min_opts = {}

    n_g = len(g_values)
    u_prev = np.zeros(N)
    # small random noise at first step to help reach asymmetric minima if they exist
    if init_noise and init_noise > 0:
        rng = np.random.default_rng(12345)
        u_prev += init_noise * rng.standard_normal(N)

    u_stars = np.zeros((n_g, N))
    ipr_vals = np.zeros(n_g)
    E_vals = np.zeros(n_g)
    success_flags = np.zeros(n_g, dtype=bool)
    psi_stars = np.zeros((n_g, N), dtype=complex)

    for i, g in enumerate(g_values):
        # use previous relaxed u as starting guess (continuation)
        x0 = u_prev.copy()
        # optionally add a tiny directed seed for early g if you want:
        # if i == 0:
        #     x0 += delta_scale * np.sin(np.linspace(0, np.pi, N))  # e.g. small sine distortion

        res = minimize(lambda u: total_energy(u, t, g, K, pbc=pbc),
                       x0, method=minimizer_method, options=min_opts)
        u_star = res.x
        E_star = total_energy(u_star, t, g, K, pbc=pbc)
        e_elec, psi = electronic_groundstate_for_u(u_star, t, g, pbc=pbc)
        # normalize psi just in case (eigh returns normalized but be safe)
        psi = psi / np.linalg.norm(psi)
        ipr = compute_ipr_from_wavefunction(psi)

        u_stars[i, :] = u_star
        ipr_vals[i] = ipr
        E_vals[i] = E_star
        success_flags[i] = res.success
        psi_stars[i, :] = psi

        # continuation
        u_prev = u_star

    return {
        "g": np.array(g_values),
        "u_stars": u_stars,
        "ipr": ipr_vals,
        "E": E_vals,
        "success": success_flags,
        "psi": psi_stars
    }

# ---------------------------
# Run for OBC and PBC
# ---------------------------
print("Starting relaxation scan (this can take some seconds)...")
res_obc = relax_scan(N=N, t=t, K=K, g_values=g_values, pbc=False,
                     init_noise=initial_noise, delta_scale=delta_scale,
                     minimizer_method=minimizer_method, min_opts=min_opts)
res_pbc = relax_scan(N=N, t=t, K=K, g_values=g_values, pbc=True,
                     init_noise=initial_noise, delta_scale=delta_scale,
                     minimizer_method=minimizer_method, min_opts=min_opts)
print("Done scans.")

# Save CSVs
np.savetxt(os.path.join(outdir, "g_values.csv"), g_values, delimiter=",", header="g")
np.savetxt(os.path.join(outdir, "ipr_obc.csv"), np.vstack([g_values, res_obc["ipr"]]).T, delimiter=",", header="g,IPR")
np.savetxt(os.path.join(outdir, "ipr_pbc.csv"), np.vstack([g_values, res_pbc["ipr"]]).T, delimiter=",", header="g,IPR")
np.savetxt(os.path.join(outdir, "E_obc.csv"), np.vstack([g_values, res_obc["E"]]).T, delimiter=",", header="g,E")
np.savetxt(os.path.join(outdir, "E_pbc.csv"), np.vstack([g_values, res_pbc["E"]]).T, delimiter=",", header="g,E")
np.savetxt(os.path.join(outdir, "u_stars_obc.csv"), res_obc["u_stars"], delimiter=",")
np.savetxt(os.path.join(outdir, "u_stars_pbc.csv"), res_pbc["u_stars"], delimiter=",")

# ---------------------------
# Plotting
# ---------------------------
# 1) IPR vs g (both BC)
plt.figure(figsize=(6.0, 4.5))
plt.plot(res_obc["g"], res_obc["ipr"], color="black",marker='+', label="OBC (relaxed)", linewidth=1)
plt.plot(res_pbc["g"], res_pbc["ipr"], color="blue", marker=None, label="PBC (relaxed)", linewidth=1)
plt.xlabel("electron-phonon coupling g", fontname="Georgia", fontsize=12)
plt.ylabel("IPR (sum |ψ|^4) at relaxed u*", fontname="Georgia", fontsize=12)
plt.title(f"IPR vs g (N={N}) — relaxed lattice (continuation)", fontname="Georgia" )
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0
)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "ipr_vs_g_relaxed.png"), dpi=400)


# 2) choose three sample g: before threshold, near threshold, after threshold
# find approximate 'threshold' as significant increase in IPR: compute derivative
def pick_sample_indices(ipr_array):
    # pick min, mid (where slope is largest), and max indices
    idx_min = 0
    idx_max = len(ipr_array) - 1
    # approximate slope and pick index where slope magnitude is maximal
    slopes = np.abs(np.gradient(ipr_array))
    idx_mid = int(np.argmax(slopes))
    return [idx_min, idx_mid, idx_max]

inds = pick_sample_indices(res_obc["ipr"])
sample_gs = res_obc["g"][inds]

# Multi-panel: for each sample g, show u* (OBC & PBC) and |psi|^2 (OBC & PBC)
fig, axes = plt.subplots(3, 2, figsize=(10, 8), constrained_layout=True)
for row, idx in enumerate(inds):
    gval = res_obc["g"][idx]
    # u* plots
    u_obc = res_obc["u_stars"][idx]
    u_pbc = res_pbc["u_stars"][idx]
    ax_u = axes[row, 0]
    ax_u.plot(np.arange(1, N+1), u_obc, marker=None, label=f"OBC u* (g={gval:.3f})", color='black')
    ax_u.plot(np.arange(1, N+1), u_pbc, marker=None, label=f"PBC u* (g={gval:.3f})", color='blue' )
    ax_u.set_xlabel("site", fontname="Georgia", fontsize=12)
    ax_u.set_ylabel("u*", fontname="Georgia", fontsize=12)
    ax_u.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
    if row == 0:
        ax_u.set_title("Relaxed displacements u*(g)", fontname="Georgia")
    ax_u.grid(False)
    ax_u.legend(
        prop={'family': 'Georgia', 'size': 10},
        frameon=True,
        edgecolor='black',
        framealpha=1.0
    )

    # density plots
    ax_psi = axes[row, 1]
    psi_obc = res_obc["psi"][idx]
    psi_pbc = res_pbc["psi"][idx]
    ax_psi.bar(np.arange(1, N+1) - 0.2, np.abs(psi_obc)**2, width=0.4, label="OBC |ψ|^2", color='black')
    ax_psi.bar(np.arange(1, N+1) + 0.2, np.abs(psi_pbc)**2, width=0.4, label="PBC |ψ|^2", color='blue'  )
    ax_psi.set_xlabel("site",fontname="Georgia", fontsize=12)
    ax_psi.set_ylabel("|ψ|^2",fontname="Georgia", fontsize=12)
    ax_psi.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
    if row == 0:
        ax_psi.set_title("Electronic density at u*(g)",fontname="Georgia")
    ax_psi.grid(False)
    ax_psi.legend(
        prop={'family': 'Georgia', 'size': 10},
        frameon=True,
        edgecolor='black',
        framealpha=1.0
    )

plt.suptitle(f"Sample relaxed displacements and densities (N={N})", y=1.02,fontname="Georgia" )
plt.savefig(os.path.join(outdir, "sample_u_and_densities.png"), dpi=400, bbox_inches="tight")


# 3) Optional: plot energy vs g curves to see energy lowering trend
plt.figure(figsize=(6.0, 4.5))
plt.plot(res_obc["g"], res_obc["E"], marker='+', label="E*(g) OBC", color='black')
plt.plot(res_pbc["g"], res_pbc["E"], marker=None, label="E*(g) PBC", color='blue')
plt.xlabel("g",fontname="Georgia", fontsize=12)
plt.ylabel("Total energy E(u*(g))", fontname="Georgia", fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.title("Energy of relaxed states vs g",fontname="Georgia")
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
        frameon=True,
        edgecolor='black',
        framealpha=1.0
)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "E_vs_g_relaxed.png"), dpi=400)

print("All done. Results are saved in:", outdir)

