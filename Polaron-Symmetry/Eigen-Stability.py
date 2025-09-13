# Author: Jaafar Mehrez, jaafarmehrez@sjtu.edu.cn
# Compute unstable eigenvector and relax along it for N=12, both OBC and PBC.

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

outdir = "./data-Eigen/polaron_demo_results"
os.makedirs(outdir, exist_ok=True)

def tb_hamiltonian(u, t, pbc=False):
    N = len(u)
    H = np.zeros((N,N))
    for i in range(N):
        H[i,i] = u[i]
    for i in range(N-1):
        H[i,i+1] = -t
        H[i+1,i] = -t
    if pbc and N>1:
        H[0,-1] = -t; H[-1,0] = -t
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

# Parameters
t = 1.0; 
K = 1.0; 
N = 50

g_values = np.linspace(0.0,3.5,71)  # step ~0.05
eps_hess = 1e-5
results = {}

for pbc_flag in [False, True]:
    tag = "PBC" if pbc_flag else "OBC"
    print(f"Scanning for first unstable g for {tag} ...")
    first_unstable_g = None
    first_mode = None
    min_eigs = []
    for g in g_values:
        Hh = numeric_hessian(np.zeros(N), t, g, K, pbc=pbc_flag, eps=eps_hess)
        eigs, vecs = np.linalg.eigh(Hh)
        min_eigs.append(eigs[0])
        if first_unstable_g is None and eigs[0] < 0:
            first_unstable_g = g
            first_mode = vecs[:,0]
            # don't break: continue scanning to store min_eigs for plotting

    min_eigs = np.array(min_eigs)
    results[tag] = {"g_values":g_values, "min_eigs":min_eigs, "g_c":first_unstable_g, "mode":first_mode}
    print(f"{tag}: first unstable g = {first_unstable_g}")
    
    # If unstable mode found, perform perturbation and relaxation
    if first_unstable_g is not None:
        v = first_mode
        # normalize mode such that max|v| = 1 (for scaling convenience)
        v_norm = v / np.max(np.abs(v))
        delta = 0.08  # initial displacement amplitude
        u0 = delta * v_norm
        # minimize total energy starting from u0
        res = minimize(lambda u: total_energy(u, t, first_unstable_g, K, pbc=pbc_flag), u0, method='BFGS', options={'gtol':1e-9, 'maxiter':1000})
        u_star = res.x
        E0 = total_energy(np.zeros(N), t, first_unstable_g, K, pbc=pbc_flag)
        E_star = total_energy(u_star, t, first_unstable_g, K, pbc=pbc_flag)
        
        # electronic densities
        e_sym, psi_sym = electronic_groundstate(np.zeros(N), t, first_unstable_g, pbc=pbc_flag)
        e_rel, psi_rel = electronic_groundstate(u_star, t, first_unstable_g, pbc=pbc_flag)
        dens_sym = np.abs(psi_sym)**2
        dens_rel = np.abs(psi_rel)**2

        # energy profile along alpha * v_norm from 0 to 1.5
        alphas = np.linspace(0.0, 1.5, 61)
        E_alphas = [total_energy(alpha * v_norm * delta, t, first_unstable_g, K, pbc=pbc_flag) for alpha in alphas]
        
        # save in results
        results[tag].update({"v_norm":v_norm, "u0":u0, "u_star":u_star, "E0":E0, "E_star":E_star, "dens_sym":dens_sym, "dens_rel":dens_rel, "alphas":alphas, "E_alphas":E_alphas, "relax_res":res})
        
        # save CSVs and plots
        np.savetxt(os.path.join(outdir, f"{tag}_g_values.csv"), g_values, delimiter=",")
        np.savetxt(os.path.join(outdir, f"{tag}_min_eigs.csv"), min_eigs, delimiter=",")
        np.savetxt(os.path.join(outdir, f"{tag}_mode_vnorm.csv"), v_norm, delimiter=",")
        np.savetxt(os.path.join(outdir, f"{tag}_u_star.csv"), u_star, delimiter=",")
        np.savetxt(os.path.join(outdir, f"{tag}_dens_sym.csv"), dens_sym, delimiter=",")
        np.savetxt(os.path.join(outdir, f"{tag}_dens_rel.csv"), dens_rel, delimiter=",")
        np.savetxt(os.path.join(outdir, f"{tag}_E_alphas.csv"), np.vstack([alphas, E_alphas]).T, delimiter=",", header="alpha,E(alpha)")
        
        # Plotting
        # 1) plot min_eigs vs g with g_c marker
        plt.figure(figsize=(6.5,4))
        plt.plot(g_values, min_eigs, label=f"{tag} N={N}", color='black')
        if first_unstable_g is not None:
            plt.axvline(first_unstable_g, linestyle='--', color='r', label=f"g_c≈{first_unstable_g:.3f}")
        plt.axhline(0, color='k', linestyle=':')
        plt.xlabel("g", fontname="cmr10", fontsize=12)
        plt.ylabel("smallest Hessian eigenvalue", fontname="cmr10", fontsize=12)
        plt.title(f"{tag}: min Hessian eigenvalue vs g (N={N})", fontname="Georgia")
        plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
        plt.legend(
            prop={'family': 'Georgia', 'size': 10},
            frameon=True,
            edgecolor='black',
            framealpha=1.0
        )
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{tag}_min_eig_vs_g.png"), dpi=400)
        plt.close()

        # 2) unstable eigenvector (v_norm)
        plt.figure(figsize=(6,3))
        plt.stem(range(1,N+1), v_norm, linefmt='blue', basefmt='black')
        plt.xlabel("site",fontname="cmr10", fontsize=12)
        plt.ylabel("mode amplitude",fontname="cmr10", fontsize=12)
        plt.title(f"{tag}: unstable eigenvector (normalized) at g≈{first_unstable_g:.3f}",fontname="Georgia")
        plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{tag}_unstable_mode.png"), dpi=400)
        plt.close()

        # 3) initial u0 and relaxed u_star
        plt.figure(figsize=(6,3))
        plt.plot(range(1,N+1), u0, marker='o', label="initial perturbation $u_0$", color='black', markersize=4)
        plt.plot(range(1,N+1), u_star, marker='s', label="relaxed $u^*$", color='blue', markersize=4)
        plt.xlabel("site",fontname="cmr10", fontsize=12)
        plt.ylabel("displacement $u$",fontname="cmr10", fontsize=12)
        plt.title(f"{tag}: initial vs relaxed displacements (g≈{first_unstable_g:.3f})", fontname="Georgia")
        plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
        plt.legend(
            prop={'family': 'Georgia', 'size': 10},
            frameon=True,   
            edgecolor='black',
            framealpha=1.0 
        )
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{tag}_u0_u_star.png"), dpi=400)
        plt.close()

        # 4) electronic density before/after
        plt.figure(figsize=(6,3))
        plt.bar(np.arange(1,N+1)-0.2, dens_sym, width=0.4, label="symmetric |ψ|^2", color='black')
        plt.bar(np.arange(1,N+1)+0.2, dens_rel, width=0.4, label="relaxed |ψ|^2", color='blue')
        plt.xlabel("site", fontname="cmr10", fontsize=12)
        plt.ylabel("site occupation", fontname="cmr10", fontsize=12)
        plt.title(f"{tag}: electronic density before/after relaxation (g≈{first_unstable_g:.3f})", fontname='Georgia')
        plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
        plt.legend(
            prop={'family': 'Georgia', 'size': 10},
            frameon=True,   
            edgecolor='black',
            framealpha=1.0 
        )
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{tag}_density_before_after.png"), dpi=400)
        plt.close()

        # 5) energy profile along alpha
        plt.figure(figsize=(6.5,4))
        plt.plot(alphas, E_alphas, marker='o', color='black', markersize=4)
        plt.axvline(1.0, linestyle='--', color='r', label='alpha=1 (initial perturbation)')
        plt.axhline(E0, color='k', linestyle=':', label='E(u=0)')
        plt.scatter([1.0], [total_energy(delta * v_norm, t, first_unstable_g, K, pbc=pbc_flag)], color='r')
        plt.scatter([0.0], [E0], color='k')
        plt.xlabel("alpha", fontname="cmr10", fontsize=12)
        plt.ylabel("E(alpha * u0)", fontname="cmr10", fontsize=12)
        plt.title(f"{tag}: energy profile along unstable direction (g≈{first_unstable_g:.3f})", fontname='Georgia')
        plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
        plt.legend(
            prop={'family': 'Georgia', 'size': 10},
            frameon=True,   
            edgecolor='black',
            framealpha=1.0 
        )
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{tag}_E_along_mode.png"), dpi=400)
        plt.close()

        # record energy lowering
        results[tag]['energy_lowering'] = E0 - E_star
        results[tag]['E0'] = E0; results[tag]['E_star'] = E_star
    else:
        # save min_eigs anyway
        np.savetxt(os.path.join(outdir, f"{tag}_min_eig_vs_g.csv"), min_eigs, delimiter=",")

# List files produced and some key numeric outputs
files = sorted(os.listdir(outdir))
summary = {tag: {"g_c": results[tag]["g_c"], "energy_lowering": results[tag].get("energy_lowering", None)} for tag in results}
files, summary