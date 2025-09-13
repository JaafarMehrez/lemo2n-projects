# Author: Jaafar Mehrez, jaafarmehrez@sjtu.edu.cn
# Compare PBC vs OBC for larger N = [8,12,16,20], g sweep
# Compute approximate critical g_c (first g where smallest Hessian eigenvalue < 0) per case
# Quantum phonon ED for dimer with pmax=6 and omegas [0.1,0.5,1.0,5.0], finer g grid

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from math import sqrt

outdir = "./data-PBC-OBC/polaron_demo_extended2"
os.makedirs(outdir, exist_ok=True)

# --- TB H (OBC and PBC), electronic groundstate, total energy, numeric Hessian ---
def tb_hamiltonian(u, t, pbc=False):
    N = len(u)
    H = np.zeros((N,N))
    for i in range(N):
        H[i,i] = u[i]
    for i in range(N-1):
        H[i,i+1] = -t
        H[i+1,i] = -t
    if pbc and N>1:
        H[0,-1] = -t
        H[-1,0] = -t
    return H

def electronic_groundstate(u, t, g, pbc=False):
    H = tb_hamiltonian(g*u, t, pbc=pbc)
    vals, vecs = np.linalg.eigh(H)
    return vals[0], vecs[:,0]

def total_energy(u, t, g, K, pbc=False):
    e, _ = electronic_groundstate(u, t, g, pbc=pbc)
    return e + 0.5*K*np.sum(u**2)

def numeric_hessian(u0, t, g, K, pbc=False, eps=1e-5):
    N = len(u0)
    Hmat = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            ei = np.zeros(N); ej = np.zeros(N)
            ei[i] = eps; ej[j] = eps
            E_pp = total_energy(u0 + ei + ej, t, g, K, pbc=pbc)
            E_pm = total_energy(u0 + ei - ej, t, g, K, pbc=pbc)
            E_mp = total_energy(u0 - ei + ej, t, g, K, pbc=pbc)
            E_mm = total_energy(u0 - ei - ej, t, g, K, pbc=pbc)
            Hmat[i,j] = (E_pp - E_pm - E_mp + E_mm) / (4*eps*eps)
    return Hmat

t = 1.0; 
K = 1.0
Ns = [8,12,16,20,30]
g_values = np.linspace(0.0,3.5,50)  # finer but not too large
results = {"pbc":{}, "obc":{}}

for pbc_flag in [True, False]:
    for N in Ns:
        min_eigs = []
        modes = []  # store eigenvector at first unstable g if any
        first_unstable_g = None
        for g in g_values:
            Hh = numeric_hessian(np.zeros(N), t, g, K, pbc=pbc_flag, eps=1e-6)
            eigs, vecs = np.linalg.eigh(Hh)
            min_eigs.append(eigs[0])
            if first_unstable_g is None and eigs[0] < 0:
                first_unstable_g = g
                modes = vecs[:,0]
        results["pbc" if pbc_flag else "obc"][N] = {"min_eigs":np.array(min_eigs), "g_c":first_unstable_g, "mode":modes}
        
# Plot comparison curves (PBC vs OBC) for each N on same axes
plt.figure(figsize=(6,5))
for i, N in enumerate(Ns):
    colors = ['black', 'red', 'blue', 'green','purple', 'brown', 'pink', 'gray']
    plt.plot(g_values, results["pbc"][N]["min_eigs"], label=f"PBC N={N}", linestyle='-', color=colors[i % len(colors)])
    plt.plot(g_values, results["obc"][N]["min_eigs"], label=f"OBC N={N}", linestyle='-.', color=colors[i % len(colors)])
plt.axhline(0, color='k', linestyle=':')
plt.xlabel("g",fontname="cmr10", fontsize=12) 
plt.ylabel("smallest Hessian eigenvalue",fontname="cmr10", fontsize=12)
plt.title("PBC vs OBC: smallest Hessian eigenvalue vs g",fontname="cmr10")
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.legend(
    ncol=2, 
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0
    )
plt.grid(False)
plt.tight_layout()
p_cmp = os.path.join(outdir, "hessian_pbc_vs_obc.png")
plt.savefig(p_cmp, dpi=400)
plt.close()

# Plot g_c vs N for PBC and OBC (use inf for no instability -> plot as NaN)
gcs_pbc = [results["pbc"][N]["g_c"] if results["pbc"][N]["g_c"] is not None else np.nan for N in Ns]
gcs_obc = [results["obc"][N]["g_c"] if results["obc"][N]["g_c"] is not None else np.nan for N in Ns]

plt.figure(figsize=(6,4))
plt.plot(Ns, gcs_pbc, marker='s', label="PBC", color='black')
plt.plot(Ns, gcs_obc, marker='o', label="OBC", color='red'  )
plt.xlabel("N", fontname="cmr10", fontsize=12)
plt.ylabel("approx. g_c (first unstable g)", fontname="Georgia", fontsize=12)
plt.title("Estimated g_c vs N", fontname="Georgia")
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.grid(False)
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0
)
plt.tight_layout()
p_gc = os.path.join(outdir, "g_c_vs_N.png")
plt.savefig(p_gc, dpi=400)
plt.close()

# Save modes for visualization: save first unstable mode for each N if exists
for ptag in ["pbc","obc"]:
    for N in Ns:
        mode = results[ptag][N]["mode"]
        if mode is not None and len(mode)>0:
            np.savetxt(os.path.join(outdir, f"{ptag}_mode_N{N}.csv"), mode, delimiter=",")
            
# --- Quantum phonons: ED for dimer with larger pmax and more omegas ---
def build_holstein_dimer_hamiltonian(t, g, omega, pmax):
    basis = []
    for site in range(2):
        for n0 in range(pmax):
            for n1 in range(pmax):
                basis.append((site, n0, n1))
    dim = len(basis)
    H = np.zeros((dim, dim))
    for i, (site_i, n0_i, n1_i) in enumerate(basis):
        H[i,i] += omega * (n0_i + n1_i)
        if site_i == 0:
            if n0_i > 0:
                j = basis.index((site_i, n0_i-1, n1_i)); H[i,j] += g * sqrt(n0_i)
            if n0_i+1 < pmax:
                j = basis.index((site_i, n0_i+1, n1_i)); H[i,j] += g * sqrt(n0_i+1)
        else:
            if n1_i > 0:
                j = basis.index((site_i, n0_i, n1_i-1)); H[i,j] += g * sqrt(n1_i)
            if n1_i+1 < pmax:
                j = basis.index((site_i, n0_i, n1_i+1)); H[i,j] += g * sqrt(n1_i+1)
        other = 1 - site_i
        j = basis.index((other, n0_i, n1_i)); H[i,j] += -t
    return H, basis

def groundstate_dimer_quantum(t, g, omega, pmax):
    H, basis = build_holstein_dimer_hamiltonian(t, g, omega, pmax)
    vals, vecs = np.linalg.eigh(H)
    gs = vecs[:,0]
    occ0 = 0.0; occ1 = 0.0
    for idx, (site, n0, n1) in enumerate(basis):
        prob = np.abs(gs[idx])**2
        if site == 0: occ0 += prob
        else: occ1 += prob
    ipr_elec = occ0**2 + occ1**2
    return vals[0], occ0, occ1, ipr_elec
pmax = 6
g_vals = np.linspace(0.0,4.0,50)
omegas = [0.1, 0.5, 1.0, 5.0]
quant_results = {omega: {"g":[], "occ0":[], "occ1":[], "ipr":[], "E":[]} for omega in omegas}

for omega in omegas:
    for g in g_vals:
        E, occ0, occ1, ipr_elec = groundstate_dimer_quantum(t, g, omega, pmax)
        quant_results[omega]["g"].append(g)
        quant_results[omega]["occ0"].append(occ0)
        quant_results[omega]["occ1"].append(occ1)
        quant_results[omega]["ipr"].append(ipr_elec)
        quant_results[omega]["E"].append(E)
    np.savetxt(os.path.join(outdir, f"quant_pmax{pmax}_omega{omega}.csv"),
               np.vstack([quant_results[omega]["g"], quant_results[omega]["occ0"], quant_results[omega]["occ1"], quant_results[omega]["ipr"]]).T,
               delimiter=",", header="g,occ0,occ1,ipr")

# Plot quantum IPR and occupation diff heatmaps / curves
plt.figure(figsize=(6,4.5))
for i, omega in enumerate(omegas):
    colors = ['black', 'red', 'blue', 'green','purple', 'brown', 'pink', 'gray']
    plt.plot(quant_results[omega]["g"], quant_results[omega]["ipr"], label=f"ω={omega}", color=colors[i % len(colors)])
plt.xlabel("g", fontname="cmr10", fontsize=12)
plt.ylabel("electronic IPR", fontname="cmr10", fontsize=12)
plt.title(f"Dimer quantum IPR (pmax={pmax})", fontname="cmr10")
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0
)
plt.grid(False)
plt.tight_layout()
p_qipr = os.path.join(outdir, "dimer_quant_ipr_pmax6.png")
plt.savefig(p_qipr, dpi=400)
plt.close()

plt.figure(figsize=(6,4.5))
for i,omega in enumerate(omegas):
    occ0 = np.array(quant_results[omega]["occ0"]); occ1 = np.array(quant_results[omega]["occ1"])
    colors = ['black', 'red', 'blue', 'green','purple', 'brown', 'pink', 'gray']
    plt.plot(quant_results[omega]["g"], occ0-occ1, label=f"ω={omega}", color=colors[i % len(colors)])
#plt.axhline(0, linestyle='--', color='k')
plt.xlabel("g", fontname="cmr10", fontsize=12)
plt.ylabel("Occ0-Occ1", fontname="cmr10", fontsize=12)
plt.title("Dimer quantum occupation difference (pmax=6)", fontname="cmr10")
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0
)
plt.grid(False)
plt.tight_layout()
p_qocc = os.path.join(outdir, "dimer_quant_occdiff_pmax6.png")
plt.savefig(p_qocc, dpi=400)
plt.close()

# Save summary metadata (g_c per case)
with open(os.path.join(outdir, "summary_gcs.txt"), "w") as f:
    f.write("N, pbc_g_c, obc_g_c\n")
    for N in Ns:
        f.write(f"{N}, {results['pbc'][N]['g_c']}, {results['obc'][N]['g_c']}\n")
files = sorted(os.listdir(outdir))
files, outdir