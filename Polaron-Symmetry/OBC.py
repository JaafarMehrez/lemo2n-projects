# Author: Jaafar Mehrez, jaafarmehrez@sjtu.edu.cn
# Adiabatic tight-binding + classical-phonon model.
# Plots:
# - Smallest Hessian eigenvalue vs g for N=6
# - Comparison N=2 and N=6 with analytic g_c
# - Eigenvector (mode shape) at the first g where instability appears (N=6)
# - Electronic site density after minimizing E(u) starting along unstable mode
# - Inverse participation ratio vs g for N=6 (measure of localization) 

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

outdir = "./data-OBC/polaron_demo"
os.makedirs(outdir, exist_ok=True)

def tb_hamiltonian(u, t):
    N = len(u)
    H = np.zeros((N,N))
    for i in range(N):
        H[i,i] = u[i]
    for i in range(N-1):
        H[i,i+1] = -t
        H[i+1,i] = -t
    return H

def electronic_groundstate(u, t, g):
    H = tb_hamiltonian(g*u, t)
    vals, vecs = eigh(H)
    psi0 = vecs[:,0]
    energy = vals[0]
    return energy, psi0

def total_energy(u, t, g, K):
    energy, _ = electronic_groundstate(u, t, g)
    return energy + 0.5*K*np.sum(u**2)

def numeric_hessian(u0, t, g, K, eps=1e-6):
    N = len(u0)
    Hmat = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            ei = np.zeros(N); ej = np.zeros(N)
            ei[i] = eps; ej[j] = eps
            E_pp = total_energy(u0 + ei + ej, t, g, K)
            E_pm = total_energy(u0 + ei - ej, t, g, K)
            E_mp = total_energy(u0 - ei + ej, t, g, K)
            E_mm = total_energy(u0 - ei - ej, t, g, K)
            Hmat[i,j] = (E_pp - E_pm - E_mp + E_mm) / (4*eps*eps)
    return Hmat

def ipr(psi):
    # psi normalized
    p = np.abs(psi)**2
    return np.sum(p**2)

# Parameters
t  = 1
K  = 1
N6 = 6
g_values = np.linspace(0.0, 4.0, 81)
min_eigvals_N6 = []
ipr_vals = []
psis = []

for g in g_values:
    Hh = numeric_hessian(np.zeros(N6), t, g, K, eps=1e-6)
    eigs, vecs = eigh(Hh)
    min_eigvals_N6.append(eigs[0])
    
    # compute ground-state psi at u=0
    e0, psi0 = electronic_groundstate(np.zeros(N6), t, g)
    ipr_vals.append(ipr(psi0))
    psis.append(psi0)

min_eigvals_N6 = np.array(min_eigvals_N6)
ipr_vals = np.array(ipr_vals)

# Find first g where eigenvalue crosses zero (instability)
cross_idx = np.where(min_eigvals_N6 < 0)[0]
first_unstable_g = None
unstable_mode = None
if len(cross_idx) > 0:
    idx = cross_idx[0]
    first_unstable_g = g_values[idx]
    Hh = numeric_hessian(np.zeros(N6), t, first_unstable_g, K, eps=1e-6)
    eigs, vecs = eigh(Hh)
    unstable_mode = vecs[:,0]  # eigenvector corresponding to smallest eigenvalue
    
# Compute N=2 numeric curve for comparison and analytic g_c
N2 = 2
min_eigvals_N2 = []
g_vals_fine = np.linspace(0,4,161)
for g in g_vals_fine:
    Hh = numeric_hessian(np.zeros(N2), t, g, K, eps=1e-6)
    eigs = np.linalg.eigvalsh(Hh)
    min_eigvals_N2.append(eigs.min())
min_eigvals_N2 = np.array(min_eigvals_N2)
g_c_analytic = np.sqrt(2.0*t*K)

# Plot 1: smallest Hessian eigenvalue vs g for N=6
plt.figure(figsize=(4.5, 4))
plt.plot(g_values, min_eigvals_N6, label=f"N={N6}", color='black')
plt.axhline(0.0, linestyle='--', color='black')
if first_unstable_g is not None:
    plt.axvline(first_unstable_g, linestyle=':', label=f"First Unstable g ≈ {first_unstable_g:.3f}", color='black')
plt.title("Smallest Hessian Eigenvalue vs g (Adiabatic Model)", fontname="cmr10")
plt.xlabel("Electron-Phonon Coupling (g)", fontname="cmr10", fontsize=12)
plt.ylabel("Smallest Hessian Eigenvalue", fontname="cmr10", fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0)
plt.grid(False)
plt.tight_layout()
p1 = os.path.join(outdir, "hessian_vs_g_N6.png")
plt.savefig(p1, dpi=400)
plt.close()

# Plot 2: comparison N=2 and N=6 with analytic g_c
plt.figure(figsize=(4.5,4))
plt.plot(g_values, min_eigvals_N6, label=f"N={N6}", color='black')
plt.plot(g_vals_fine, min_eigvals_N2, label="N=2 (Numeric)", color='blue')
plt.axvline(g_c_analytic, linestyle='--', label=f"Analytic gc={g_c_analytic:.3f}")
plt.axhline(0.0, linestyle=':')
plt.xlabel("g", fontname="cmr10", fontsize=12)
plt.ylabel("Smallest Hessian Eigenvalue", fontname="cmr10", fontsize=12)
plt.title("Hessian Smallest Eigenvalue (N=2 vs N=6)", fontname="cmr10")
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,   
    edgecolor='black',
    framealpha=1.0 )
plt.grid(False)
plt.tight_layout()
p2 = os.path.join(outdir, "hessian_compare_N2_N6.png")
plt.savefig(p2, dpi=400)
plt.close()

# Plot 3: shape of unstable mode (if exists)
if unstable_mode is not None:
    plt.figure(figsize=(6,3))
    plt.stem(range(1,N6+1), unstable_mode, linefmt='blue',basefmt='black')
    plt.xlabel("Site Index",fontname="cmr10", fontsize=12)
    plt.ylabel("Mode Amplitude (Unnormalized)",fontname="cmr10", fontsize=12)
    plt.title(f"Unstable Hessian Mode at g={first_unstable_g:.3f} (N={N6})", fontname="cmr10")
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
    p3 = os.path.join(outdir, "unstable_mode.png")
    plt.savefig(p3, dpi=400)
    plt.close()

# Minimize total energy starting from a small displacement along unstable mode (if exists)
minimized_path = None
relaxed_u = None
relaxed_psi = None
if unstable_mode is not None:
    u0 = 1e-3 * unstable_mode  # small initial displacement
    res = minimize(lambda u: total_energy(u, t, first_unstable_g, K), u0, method='BFGS', options={'gtol':1e-8, 'maxiter':1000})
    relaxed_u = res.x
    relaxed_E = res.fun
    e_relaxed, psi_relaxed = electronic_groundstate(relaxed_u, t, first_unstable_g)
    relaxed_psi = psi_relaxed

    # plot site density
    plt.rc('text', usetex=True)
    plt.rc('font', family='Georgia')
    plt.rc('mathtext', fontset='cm')
    plt.figure(figsize=(6,3))
    plt.bar(np.arange(1,N6+1)-0.2, np.abs(psi_relaxed)**2, width=0.4, label=r'Relaxed $|\Psi|^2$', color='black')
    # also show symmetric u=0 density for comparison
    e0, psi0 = electronic_groundstate(np.zeros(N6), t, first_unstable_g)
    plt.bar(np.arange(1,N6+1)+0.2, np.abs(psi0)**2, width=0.4, label=r'Symmetric $|\Psi|^2$', color='blue')
    plt.xlabel("Site Index", fontname="cmr10", fontsize=12)
    plt.ylabel(r'Site Occupation ($|\Psi|^2$)', fontname="Georgia", fontsize=12)
    plt.title(f"Electronic Density before/after Relaxation at g={first_unstable_g:.3f}", fontname="cmr10")
    plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='cmr10')
    plt.legend(
        prop={'family': 'Georgia', 'size': 10},
        frameon=True,   
        edgecolor='black',
        framealpha=1.0 )
    plt.tight_layout()
    p4 = os.path.join(outdir, "relaxed_density.png")
    plt.savefig(p4, dpi=400)
    plt.close()

# Plot 4: IPR vs g
plt.figure(figsize=(4.5,4))
plt.plot(g_values, ipr_vals, marker='+', markersize=12, markeredgecolor='black',markerfacecolor='black')
plt.xlabel("g", fontname="cmr10", fontsize=12)
plt.ylabel("IPR (Inverse Participation Ratio)", fontname="cmr10", fontsize=12)
plt.title(f"IPR of Electronic Ground State at u=0 (N={N6})")
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='cmr10')
plt.grid(False)
plt.tight_layout()
p5 = os.path.join(outdir, "ipr_vs_g.png")
plt.savefig(p5, dpi=400)
plt.close()

# Save numeric arrays for possible CeTZ plotting
np.savetxt(os.path.join(outdir, "g_values_N6.csv"), g_values, delimiter=",")
np.savetxt(os.path.join(outdir, "min_eigvals_N6.csv"), min_eigvals_N6, delimiter=",")
np.savetxt(os.path.join(outdir, "g_vals_fine_N2.csv"), g_vals_fine, delimiter=",")
np.savetxt(os.path.join(outdir, "min_eigvals_N2.csv"), min_eigvals_N2, delimiter=",")
if unstable_mode is not None:
    np.savetxt(os.path.join(outdir, "unstable_mode.csv"), unstable_mode, delimiter=",")
if relaxed_psi is not None:
    np.savetxt(os.path.join(outdir, "relaxed_psi.csv"), np.abs(relaxed_psi)**2, delimiter=",")
    np.savetxt(os.path.join(outdir, "symmetric_psi.csv"), np.abs(psi0)**2, delimiter=",")

# Report produced files
produced = sorted([f for f in os.listdir(outdir)])
produced, outdir
