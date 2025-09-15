
"""
Author: Jaafar Mehrez, jaafarmehrez@sjtu.edu.cn
Polaron model with Holstein + Peierls (bond) coupling + Hubbard U (mean-field),
with analytic HF-like forces (Hellmann-Feynman) for hopping/onsite and
small finite-difference correction for the HF double-counting term.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# User-set parameters
outdir = "data-Polaron-Peierls-Hubbard"
os.makedirs(outdir, exist_ok=True)

N = 12                # sites
t0 = 1.0              # base hopping (eV)
alpha = 2.0           # Peierls coupling (eV/Å) ; t_{i,i+1} = t0 - alpha*(u_{i+1}-u_i)
g = 3.0               # Holstein coupling (eV/Å)
U = 2.0               # Hubbard U (eV)
K = 6.54              # spring constant eV/Å^2 (example value from mapping)
Ne_total = N          # total electrons (choose N for half-filling)
spin_balanced = True  # if True, initial Ne_up = Ne_dn = Ne_total/2

pbc = False           # run for OBC or PBC in relaxation; later we do both
mixing_param = 0.6    # density mixing for SCF
scf_tol = 1e-6
scf_maxiter = 200
fd_eps_double_count = 1e-6   # FD for derivative of DC term
relax_tol = 1e-7

# Helper functions
def build_hopping(u, t0=t0, alpha=alpha, pbc=False):
    Nloc = len(u)
    tmat = np.zeros((Nloc, Nloc))
    for i in range(Nloc - 1):
        tij = t0 - alpha*(u[i+1] - u[i])
        tmat[i, i+1] = tij
        tmat[i+1, i] = tij
    if pbc and Nloc > 1:
        tij = t0 - alpha*(u[0] - u[-1])
        tmat[0, -1] = tij
        tmat[-1, 0] = tij
    return tmat

def build_mf_hamiltonians(u, n_up, n_dn, t0=t0, alpha=alpha, g=g, U=U, pbc=False):
    """
    Build one-particle mean-field Hamiltonians for up and down spins.
    Onsite potential for spin sigma: g*u_i + U * n_{i,opposite}
    """
    Nloc = len(u)
    tmat = build_hopping(u, t0, alpha, pbc=pbc)
    # onsite potentials:
    onsite_up = g * u + U * n_dn
    onsite_dn = g * u + U * n_up
    Hup = np.array(tmat, copy=True)
    Hdn = np.array(tmat, copy=True)
    np.fill_diagonal(Hup, onsite_up)
    np.fill_diagonal(Hdn, onsite_dn)
    return Hup, Hdn

def scf_solve(u, Ne_total, t0=t0, alpha=alpha, g=g, U=U, pbc=False,
              mixing=0.6, tol=1e-6, maxiter=200):
    """
    For fixed displacement u, solve unrestricted HF (mean-field) self-consistently.
    Returns: densities n_up, n_dn arrays, eigenvalues and eigenvectors for each spin,
             total electronic energy (mean-field expression), and a 'converged' flag.
    """
    Nloc = len(u)
    # initial densities (uniform)
    if Ne_total % 2 == 0:
        Ne_up = Ne_dn = Ne_total // 2
    else:
        # put the extra electron in up spin
        Ne_up = Ne_total // 2 + 1
        Ne_dn = Ne_total // 2

    n_up = np.full(Nloc, Ne_up / Nloc)
    n_dn = np.full(Nloc, Ne_dn / Nloc)

    for it in range(maxiter):
        Hup, Hdn = build_mf_hamiltonians(u, n_up, n_dn, t0=t0, alpha=alpha, g=g, U=U, pbc=pbc)
        eps_up, vecs_up = eigh(Hup)
        eps_dn, vecs_dn = eigh(Hdn)
        # compute new densities by occupying lowest Ne_up/Ne_dn states
        occ_up = np.zeros_like(eps_up, dtype=bool)
        occ_dn = np.zeros_like(eps_dn, dtype=bool)
        occ_up[:Ne_up] = True
        occ_dn[:Ne_dn] = True
        new_n_up = np.sum(np.abs(vecs_up[:, occ_up])**2, axis=1)
        new_n_dn = np.sum(np.abs(vecs_dn[:, occ_dn])**2, axis=1)
        # mixing
        n_up = mixing * new_n_up + (1-mixing) * n_up
        n_dn = mixing * new_n_dn + (1-mixing) * n_dn
        # check convergence
        dn = max(np.max(np.abs(new_n_up - n_up)), np.max(np.abs(new_n_dn - n_dn)))
        if dn < tol:
            # final eigenpairs and energies recomputed below
            break
    else:
        # not converged
        pass

    # final eigenpairs
    Hup, Hdn = build_mf_hamiltonians(u, n_up, n_dn, t0=t0, alpha=alpha, g=g, U=U, pbc=pbc)
    eps_up, vecs_up = eigh(Hup)
    eps_dn, vecs_dn = eigh(Hdn)
    # electronic energy: sum occupied eigenvalues
    e_elec = np.sum(eps_up[:Ne_up]) + np.sum(eps_dn[:Ne_dn])
    # double-counting correction
    double_count = -0.5 * U * np.sum(n_up * n_dn)
    E_mf = e_elec + double_count
    # return densities, eigenpairs, E_mf
    return {
        "n_up": n_up,
        "n_dn": n_dn,
        "eps_up": eps_up, "vecs_up": vecs_up,
        "eps_dn": eps_dn, "vecs_dn": vecs_dn,
        "E_elec": e_elec,
        "double_count": double_count,
        "E_mf": E_mf,
        "Ne_up": Ne_up, "Ne_dn": Ne_dn
    }

def electronic_expectation_hop_terms(vecs_up, vecs_dn, occ_up, occ_dn):
    """
    Compute expectation values <c_i^dag c_j> for occupied states for both spins.
    Returns matrix C_ij = sum_{occ,σ} <c_i^dag c_j>_σ
    """
    C = np.zeros((vecs_up.shape[0], vecs_up.shape[0]), dtype=float)
    # up spin
    if vecs_up is not None:
        C_up = vecs_up[:, :].conj() @ vecs_up[:, :].T  # not directly; we'll sum occupied
    # compute properly by summing occupied
    # We'll rely on occ masks passed by caller — but we'll instead compute sums where needed outside
    return None  # not used; expectation computed explicitly in force routine

# Force (gradient) computation
def compute_force(u, scf_result, t0=t0, alpha=alpha, g=g, U=U, pbc=False, fd_eps=1e-6):
    """
    Compute analytic Hellmann-Feynman force contributions plus finite-difference correction
    for the double-counting derivative.

    Returns force array F_i = dE_total/du_i (E_total = E_mf + 0.5 K sum u^2)
    """
    Nloc = len(u)
    # Unpack scf data
    n_up = scf_result["n_up"]
    n_dn = scf_result["n_dn"]
    vecs_up = scf_result["vecs_up"]
    vecs_dn = scf_result["vecs_dn"]
    Ne_up = scf_result["Ne_up"]; Ne_dn = scf_result["Ne_dn"]

    # 1) Holstein onsite HF contribution: g * sum_sigma |psi_{i,σ}|^2
    dens_total = n_up + n_dn  # this is exactly sum over occupied orbitals
    onsite_force = g * dens_total  # derivative of electronic energy wrt u_i from onsite term

    # 2) Peierls hopping HF contribution:
    # expectation of -∂t * (c_i^† c_j + h.c.) -> see derivation in analysis
    # For linear t_{i,i+1} = t0 - alpha*(u_{i+1}-u_i), we have:
    # ∂H/∂u_i (hopping part) = alpha * sum_sigma (c_{i-1}† c_i + h.c.) - alpha * sum_sigma (c_i† c_{i+1} + h.c.)
    # Expectation: compute real part of occupied density matrix elements
    # Build occupied density matrices:
    # rho_ij = sum_occ psi_i^* psi_j  (sum over spins)
    Nn = Nloc
    rho = np.zeros((Nn, Nn), dtype=float)
    # up spin occupied
    if Ne_up > 0:
        rho += (vecs_up[:, :Ne_up] * vecs_up[:, :Ne_up].conj()).sum(axis=1)[:,None] * 0  # placeholder
    # Rather than fancy broadcasting, compute directly:
    # easier: accumulate contributions rho_ij = sum_{occ} psi_i* psi_j
    rho.fill(0.0)
    for svecs, Nocc in ((vecs_up, Ne_up), (vecs_dn, Ne_dn)):
        if Nocc > 0:
            psi_occ = svecs[:, :Nocc]  # shape N x Nocc
            # rho += psi_occ @ psi_occ.conj().T
            rho += np.real(psi_occ @ psi_occ.conj().T)

    # Now compute Peierls contribution for each site i:
    peierls_force = np.zeros(Nn, dtype=float)
    for i in range(Nn):
        # term from bond (i-1,i)
        j = (i-1) % Nn
        # if OBC then skip bond if i==0 and not pbc
        term_left = 0.0
        if not (not pbc and i==0):
            term_left = rho[j, i] + rho[i, j]  # real part sum
        # term from bond (i,i+1)
        k = (i+1) % Nn
        term_right = 0.0
        if not (not pbc and i==Nn-1):
            term_right = rho[i, k] + rho[k, i]
        peierls_force[i] = alpha * (term_left - term_right)

    # 3) Ionic restoring force: K * u_i
    ionic_force = K * u

    # 4) double-counting energy derivative: d(-0.5 U sum n_up n_dn)/du
    # We approximate derivative numerically (finite difference) because n_up/dn depends on u implicitly.
    # Evaluate double_count at u +/- fd_eps and finite difference.
    def compute_double_count(u_local):
        res = scf_solve(u_local, Ne_total, t0=t0, alpha=alpha, g=g, U=U, pbc=pbc,
                        mixing=mixing_param, tol=scf_tol, maxiter=scf_maxiter)
        return -0.5 * U * np.sum(res["n_up"] * res["n_dn"])
    # finite diff per site:
    dc_dud = np.zeros(Nn, dtype=float)
    # use central differences
    for i in range(Nn):
        du = np.zeros(Nn)
        du[i] = fd_eps
        dc_plus = compute_double_count(u + du)
        dc_minus = compute_double_count(u - du)
        dc_dud[i] = (dc_plus - dc_minus) / (2.0 * fd_eps)

    # Total gradient (dE/du) = HF_expectation_deriv + ionic_force + derivative of double-counting
    # Note: HF expectation included onsite_force + peierls_force (these are dE_elec/du within MF Hamiltonian)
    grad = onsite_force + peierls_force + ionic_force + dc_dud

    return grad

# Single relaxation step (minimization) for given initial u
def relax_u_with_hf(u0, Ne_total, t0=t0, alpha=alpha, g=g, U=U, K=K,
                    pbc=False, verbose=True):
    """
    Minimize the total mean-field energy E(u) using L-BFGS-B and analytic+FD forces.
    """

    # wrapper returning energy and gradient for minimizer
    def E_and_grad(u_vec):
        scf = scf_solve(u_vec, Ne_total, t0=t0, alpha=alpha, g=g, U=U, pbc=pbc,
                        mixing=mixing_param, tol=scf_tol, maxiter=scf_maxiter)
        E_elec = scf["E_mf"]  # MF electronic energy (includes DC)
        E_total = E_elec + 0.5 * K * np.sum(u_vec**2)
        grad = compute_force(u_vec, scf, t0=t0, alpha=alpha, g=g, U=U, pbc=pbc, fd_eps=fd_eps_double_count)
        if verbose:
            print("E_total (call) = ", E_total)
        return E_total, grad

    res = minimize(lambda u: E_and_grad(u)[0],
                   u0, method='L-BFGS-B',
                   jac=lambda u: E_and_grad(u)[1],
                   options={'ftol':1e-10, 'gtol':relax_tol, 'maxiter':1000})
    # final SCF at u*
    u_star = res.x
    scf_final = scf_solve(u_star, Ne_total, t0=t0, alpha=alpha, g=g, U=U, pbc=pbc,
                          mixing=mixing_param, tol=scf_tol, maxiter=scf_maxiter)
    E_elec = scf_final["E_mf"]
    E_total = E_elec + 0.5 * K * np.sum(u_star**2)
    return {"res": res, "u_star": u_star, "E_total": E_total, "scf": scf_final}

# High-level scan of g (relaxation per g) for both BC
def scan_g_and_relax(g_list, N, Ne_total, t0=t0, alpha=alpha, g_h=g, U=U, K=K, pbc=False):
    u_prev = np.zeros(N)   # initial guess
    results = {"g": [], "u_star": [], "E_total": [], "ipr": [], "dens_up": [], "dens_dn": []}
    for gg in g_list:
        # update global g used by SCF functions by passing as argument; to keep code simple, temporarily override g global:
        global g
        g_old = g
        g = gg
        print(f"\n--- relaxing at g = {gg:.4f} eV/Å, pbc={pbc} ---")
        out = relax_u_with_hf(u_prev, Ne_total, t0=t0, alpha=alpha, g=g, U=U, K=K, pbc=pbc, verbose=False)
        u_star = out["u_star"]
        scf = out["scf"]
        # compute IPR from occupied MF states: build occupied single particle wavefunctions (all spins)
        psi_all = np.hstack((scf["vecs_up"][:, :scf["Ne_up"]], scf["vecs_dn"][:, :scf["Ne_dn"]]))
        # normalize not needed; compute site occupations:
        occ_sites = np.sum(np.abs(psi_all)**2, axis=1)
        ipr = np.sum(occ_sites**2)
        results["g"].append(gg)
        results["u_star"].append(u_star.copy())
        results["E_total"].append(out["E_total"])
        results["ipr"].append(ipr)
        results["dens_up"].append(scf["n_up"].copy())
        results["dens_dn"].append(scf["n_dn"].copy())
        u_prev = u_star  # continuation
        g = g_old
    # convert to arrays
    for k in ["g","E_total","ipr"]:
        results[k] = np.array(results[k])
    results["u_star"] = np.array(results["u_star"])
    results["dens_up"] = np.array(results["dens_up"])
    results["dens_dn"] = np.array(results["dens_dn"])
    return results

# Run example scans: do both OBC and PBC
# g grid (eV/Å)
g_list = np.linspace(0.0, 6.0, 7)  # coarse grid to be quick; increase if you want more resolution

print("Running scan (this may take some time because of SCF inside forces)...")
res_obc = scan_g_and_relax(g_list, N, Ne_total, t0=t0, alpha=alpha, g_h=g, U=U, K=K, pbc=False)
res_pbc = scan_g_and_relax(g_list, N, Ne_total, t0=t0, alpha=alpha, g_h=g, U=U, K=K, pbc=True)
print("Scans finished.")

# Save CSVs and plots
np.savetxt(os.path.join(outdir, "g_list.csv"), res_obc["g"], delimiter=",", header="g")
np.savetxt(os.path.join(outdir, "ipr_obc.csv"), np.vstack([res_obc["g"], res_obc["ipr"]]).T, delimiter=",", header="g,IPR")
np.savetxt(os.path.join(outdir, "ipr_pbc.csv"), np.vstack([res_pbc["g"], res_pbc["ipr"]]).T, delimiter=",", header="g,IPR")
np.savetxt(os.path.join(outdir, "E_obc.csv"), np.vstack([res_obc["g"], res_obc["E_total"]]).T, delimiter=",", header="g,E_total")
np.savetxt(os.path.join(outdir, "E_pbc.csv"), np.vstack([res_pbc["g"], res_pbc["E_total"]]).T, delimiter=",", header="g,E_total")
np.save(os.path.join(outdir, "u_star_obc.npy"), res_obc["u_star"])
np.save(os.path.join(outdir, "u_star_pbc.npy"), res_pbc["u_star"])
np.save(os.path.join(outdir, "dens_obc.npy"), np.vstack([res_obc["dens_up"], res_obc["dens_dn"]]))
np.save(os.path.join(outdir, "dens_pbc.npy"), np.vstack([res_pbc["dens_up"], res_pbc["dens_dn"]]))

# plots: IPR vs g
plt.figure(figsize=(7,4))
plt.plot(res_obc["g"], res_obc["ipr"], '-o', label="OBC", color='black')
plt.plot(res_pbc["g"], res_pbc["ipr"], '-s', label="PBC", color='blue')
plt.xlabel("g (eV/Å)", fontname="Georgia", fontsize=12)
plt.ylabel("IPR (occupied electrons at u*)", fontname="Georgia", fontsize=12)
plt.title("IPR vs g (with Peierls & Hubbard U, MF)",fontname="Georgia")
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0
)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "ipr_vs_g_peierls_hubbard.png"), dpi=400)
plt.close()

# sample u* at three g (first, mid, last)
inds = [0, len(g_list)//2, -1]
plt.figure(figsize=(8,6))
for i, idx in enumerate(inds):
    plt.subplot(3,1,i+1)
    plt.plot(np.arange(1, N+1), res_obc["u_star"][idx], '-o', label=f"OBC u* g={res_obc['g'][idx]:.3f}", color='black')
    plt.plot(np.arange(1, N+1), res_pbc["u_star"][idx], '-s', label=f"PBC u* g={res_pbc['g'][idx]:.3f}", color='blue')
    plt.ylabel("u* (Å)",fontname="Georgia", fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
    plt.legend()
    plt.grid(False)
plt.xlabel("site",fontname="Georgia", fontsize=12)
plt.suptitle("Sample relaxed u*(g)", fontname="Georgia")
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(outdir, "sample_u_stars_peierls_hubbard.png"), dpi=400)
plt.close()

print("Results saved in:", outdir)

