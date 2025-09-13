# Author: Jaafar Mehrez, jaafarmehrez@sjtu.edu.cn
# Periodic boundary conditions (PBC) adiabatic model: compute smallest Hessian eigenvalue vs g for N up to 20 (PBC).
# Quantum-phonon exact diagonalization for N=2 (dimer) Holstein model with one electron and truncated phonon basis:
#    H = -t sum_{<ij>} c_i^† c_j + ω sum_i b_i^† b_i + g sum_i (b_i + b_i^†) n_i
#    We'll diagonalize in basis |site> inner_prodcut |n1,n2> with phonon cutoff pmax.

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from itertools import product
from math import sqrt

outdir = "./data-PBC/polaron_demo_extended"
os.makedirs(outdir, exist_ok=True)

# --- PBC adiabatic Hessian ---

def tb_hamiltonian_pbc(u, t):
    N = len(u)
    H = np.zeros((N,N))
    for i in range(N):
        H[i,i] = u[i]
    for i in range(N):
        j = (i+1) % N
        H[i,j] = -t
        H[j,i] = -t
    return H

def electronic_groundstate_pbc(u, t, g):
    H = tb_hamiltonian_pbc(g*u, t)
    vals, vecs = eigh(H)
    return vals[0], vecs[:,0]

def total_energy_pbc(u, t, g, K):
    e, _ = electronic_groundstate_pbc(u, t, g)
    return e + 0.5*K*np.sum(u**2)

def numeric_hessian_pbc(u0, t, g, K, eps=1e-6):
    N = len(u0)
    Hmat = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            ei = np.zeros(N); ej = np.zeros(N)
            ei[i] = eps; ej[j] = eps
            E_pp = total_energy_pbc(u0 + ei + ej, t, g, K)
            E_pm = total_energy_pbc(u0 + ei - ej, t, g, K)
            E_mp = total_energy_pbc(u0 - ei + ej, t, g, K)
            E_mm = total_energy_pbc(u0 - ei - ej, t, g, K)
            Hmat[i,j] = (E_pp - E_pm - E_mp + E_mm) / (4*eps*eps)
    return Hmat

t = 1.0; 
K = 1.0
Ns = [4,8,12,16,20]
g_values = np.linspace(0.0,4.0,41)
min_eig_by_N = {}

for N in Ns:
    min_eigs = []
    for g in g_values:
        Hh = numeric_hessian_pbc(np.zeros(N), t, g, K, eps=1e-6)
        eigs = np.linalg.eigvalsh(Hh)
        min_eigs.append(eigs.min())
    min_eig_by_N[N] = np.array(min_eigs)
    np.savetxt(os.path.join(outdir, f"min_eig_N{N}.csv"), min_eig_by_N[N], delimiter=",")

# Plot PBC curves
plt.figure(figsize=(6,4.5))
for i, N in enumerate(Ns):
    colors = ['black', 'red', 'blue', 'green','purple', 'brown', 'pink', 'gray']
    plt.plot(g_values, min_eig_by_N[N], label=f"N={N}", color=colors[i % len(colors)] , linestyle='-.' )
plt.axhline(0, linestyle='--', color='k')
plt.xlabel("g", fontname="cmr10", fontsize=12)
plt.ylabel("smallest Hessian eigenvalue", fontname="cmr10", fontsize=12) 
plt.title("PBC: smallest Hessian eigenvalue vs g",fontname="cmr10")
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0)
plt.grid(False)
plt.tight_layout()
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
p_pbc = os.path.join(outdir, "hessian_vs_g_PBC.png")
plt.savefig(p_pbc, dpi=400); plt.close()

# --- Quantum phonons exact diag for N=2 (dimer) ---

# Build basis: electron position (0 or 1) and phonon occupations n0, n1 each in [0,pmax-1]

def build_holstein_dimer_hamiltonian(t, g, omega, pmax):
    # basis size = 2 * pmax * pmax
    Nsite = 2
    basis = []
    for site in range(Nsite):
        for n0 in range(pmax):
            for n1 in range(pmax):
                basis.append((site, n0, n1))
    dim = len(basis)
    H = np.zeros((dim, dim))
    # precompute sqrt factors for b and b^\dagger
    for i, (site_i, n0_i, n1_i) in enumerate(basis):
        # phonon energies
        H[i,i] += omega * (n0_i + n1_i)
        # electron-phonon coupling g*(b+b^\dagger)*n_i
        # if electron on site 0: coupling acts on n0
        if site_i == 0:
            # b on n0 -> sqrt(n0) reduce n0 by 1: matrix element g*sqrt(n0)
            if n0_i > 0:
                j = basis.index((site_i, n0_i-1, n1_i))
                H[i,j] += g * sqrt(n0_i)
            # b^\dagger on n0 -> increase n0 by 1
            if n0_i+1 < pmax:
                j = basis.index((site_i, n0_i+1, n1_i))
                H[i,j] += g * sqrt(n0_i+1)
        else:
            # electron on site 1 couples to n1
            if n1_i > 0:
                j = basis.index((site_i, n0_i, n1_i-1))
                H[i,j] += g * sqrt(n1_i)
            if n1_i+1 < pmax:
                j = basis.index((site_i, n0_i, n1_i+1))
                H[i,j] += g * sqrt(n1_i+1)
        # hopping: electron moves to other site keeping phonons same
        other = 1 - site_i
        j = basis.index((other, n0_i, n1_i))
        H[i,j] += -t
    return H, basis

def groundstate_dimer_quantum(t, g, omega, pmax):
    H, basis = build_holstein_dimer_hamiltonian(t, g, omega, pmax)
    vals, vecs = eigh(H)
    gs = vecs[:,0]
    # compute electron site occupations <n_i>
    occ0 = 0.0; occ1 = 0.0
    for idx, (site, n0, n1) in enumerate(basis):
        prob = np.abs(gs[idx])**2
        if site == 0: occ0 += prob
        else: occ1 += prob
    # IPR for electronic part: need marginal electron prob per site as above
    ipr_elec = occ0**2 + occ1**2
    E_gs = vals[0]
    return E_gs, occ0, occ1, ipr_elec

# scan g for different phonon frequencies omega
pmax = 6
g_vals = np.linspace(0.0,4.0,81)
omegas = [0.2, 1.0, 5.0]  # low (adiabatic-like), medium, high (anti-adiabatic)
results_quantum = {omega: {"occ0":[], "occ1":[], "ipr":[] , "E":[] } for omega in omegas}

for omega in omegas:
    for g in g_vals:
        E, occ0, occ1, ipr_elec = groundstate_dimer_quantum(t, g, omega, pmax)
        results_quantum[omega]["occ0"].append(occ0)
        results_quantum[omega]["occ1"].append(occ1)
        results_quantum[omega]["ipr"].append(ipr_elec)
        results_quantum[omega]["E"].append(E)
    # save
    np.savetxt(os.path.join(outdir, f"quant_occ_omega_{omega}.csv"), 
               np.vstack([g_vals, results_quantum[omega]["occ0"], results_quantum[omega]["occ1"], results_quantum[omega]["ipr"]]).T,
               delimiter=",", header="g,occ0,occ1,ipr")
    
# Plot quantum IPR vs g for different omegas
plt.figure(figsize=(6,4.5))
for i,omega in enumerate(omegas):
    colors = ['black', 'red', 'blue', 'green','purple', 'brown', 'pink', 'gray']
    plt.plot(g_vals, results_quantum[omega]["ipr"], label=f"ω={omega}", color=colors[i % len(colors)])
plt.xlabel("g", fontname="cmr10", fontsize=12) 
plt.ylabel("electronic IPR (dimer quantum)",  fontname="cmr10", fontsize=12)
plt.title("Dimer quantum Holstein: IPR vs g (various ω)", fontname="Georgia")
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0)
plt.grid(False)
plt.tight_layout()
p_q = os.path.join(outdir, "dimer_quant_ipr_vs_g.png")
plt.savefig(p_q, dpi=400)
plt.close()

# Also plot site occupation difference occ0-occ1 vs g to see localization asymmetry
plt.figure(figsize=(6,4.5))
for i,omega in enumerate(omegas):
    occ0 = np.array(results_quantum[omega]["occ0"])
    occ1 = np.array(results_quantum[omega]["occ1"])
    colors = ['black', 'red', 'blue', 'green','purple', 'brown', 'pink', 'gray']
    plt.plot(g_vals, occ0-occ1, label=f"ω={omega}", color=colors[i % len(colors)])
#plt.axhline(0, linestyle='--', color='k')
plt.xlabel("g", fontname="cmr10", fontsize=12)
plt.ylabel("Occ0 - Occ1", fontname="cmr10", fontsize=12)
plt.title("Dimer quantum: site occupation difference vs g",  fontname="cmr10")
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0
)
plt.grid(False)
plt.tight_layout()
p_q2 = os.path.join(outdir, "dimer_quant_occ_diff.png")
plt.savefig(p_q2, dpi=400)
plt.close()

# Save files list
files = sorted(os.listdir(outdir))
files, outdir
