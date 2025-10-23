"""
Compare Velocity-Verlet (VV) vs Yoshida-4 composition and benchmark:
 - accuracy (RMS and max relative energy error)
 - computational cost (force evaluations and runtime)
Author: Jaafar Mehrez, jaafarmehrez@sjtu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import os

outdir = "./Data-OneBody/"
os.makedirs(outdir, exist_ok=True)

FORCE_EVALS = 0
def incr_force_count(n=1):
    global FORCE_EVALS
    FORCE_EVALS += n


def force(q):
    incr_force_count(1)
    return -q / np.linalg.norm(q)**3 
def potential(q):
    return -1 / np.linalg.norm(q)
def kinetic(p):
    return np.dot(p, p) / 2.0

def vv_step(q, p, dt):
    
    F = force(q)                    
    p_half = p + 0.5 * dt * F
    q_new = q + dt * p_half
    F_new = force(q_new)            
    p_new = p_half + 0.5 * dt * F_new
    return q_new, p_new


a1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
a0 = - (2.0**(1.0/3.0)) / (2.0 - 2.0**(1.0/3.0))

def yoshida_step(q, p, dt):
    dt1 = a1 * dt
    q1, p1 = vv_step(q, p, dt1)
    dt0 = a0 * dt
    q2, p2 = vv_step(q1, p1, dt0)
    q3, p3 = vv_step(q2, p2, dt1)
    return q3, p3


def run_integrator(step_func, q0, p0, dt, n_steps, store_all=True):
    
    global FORCE_EVALS
    FORCE_EVALS = 0  
    q = q0.copy()
    p = p0.copy()
    qs = []
    ps = []
    Ts = []
    Vs = []
    Es = []

    t0 = time.perf_counter()
    for i in range(n_steps):
        q, p = step_func(q, p, dt)
        if store_all:
            qs.append(q.copy())
            ps.append(p.copy())
            T = kinetic(p)
            V = potential(q)
            Ts.append(T)
            Vs.append(V)
            Es.append(T+V)
    t1 = time.perf_counter()

    runtime = t1 - t0
    force_evals = FORCE_EVALS
    if store_all:
        return (np.array(qs), np.array(ps), np.array(Ts), np.array(Vs), np.array(Es),
                force_evals, runtime)
    else:
        # not used, kept for completeness
        return None, None, None, None, None, force_evals, runtime


def relative_energy_metrics(E):
    E0 = E[0]
    rel = (E - E0) / abs(E0)
    rms = np.sqrt(np.mean(rel**2))
    maxabs = np.max(np.abs(rel))
    return rms, maxabs


if __name__ == "__main__":
    
    q0 = np.array([0.2, 0.0])
    p0 = np.array([0.0, 2.0])

    
    T_total = 10   # simulate 10 time units
    # list of base timesteps to test (log-spaced-ish)
    dt_list = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3]

    # storage for results
    vv_results = {'dt':[], 'force_evals':[], 'runtime':[], 'rms':[], 'max':[]}
    y_results  = {'dt':[], 'force_evals':[], 'runtime':[], 'rms':[], 'max':[]}

    for dt in dt_list:
        n_steps = int(np.round(T_total / dt))
        if n_steps < 2:
            continue
        print(f"Testing dt={dt:.1e} (n_steps={n_steps}) ...", flush=True)

        vv_q, vv_p, vv_T, vv_V, vv_E, vv_force, vv_time = run_integrator(vv_step, q0, p0, dt, n_steps)
        vv_rms, vv_max = relative_energy_metrics(vv_E)
        vv_results['dt'].append(dt); vv_results['force_evals'].append(vv_force)
        vv_results['runtime'].append(vv_time); vv_results['rms'].append(vv_rms); vv_results['max'].append(vv_max)
        print(f" VV: force_evals={vv_force}, time={vv_time:.3f}s, rms_relE={vv_rms:.3e}, max_relE={vv_max:.3e}")

        y_q, y_p, y_T, y_V, y_E, y_force, y_time = run_integrator(yoshida_step, q0, p0, dt, n_steps)
        y_rms, y_max = relative_energy_metrics(y_E)
        y_results['dt'].append(dt); y_results['force_evals'].append(y_force)
        y_results['runtime'].append(y_time); y_results['rms'].append(y_rms); y_results['max'].append(y_max)
        print(f" Y4: force_evals={y_force}, time={y_time:.3f}s, rms_relE={y_rms:.3e}, max_relE={y_max:.3e}")


    for D in (vv_results, y_results):
        for k,v in D.items():
            if k != 'dt':
                D[k] = np.array(v)

    
    plt.figure(figsize=(4.5, 4))
    plt.loglog(vv_results['force_evals'], vv_results['rms'], marker='o', label='VV', color='black')
    plt.loglog(y_results['force_evals'], y_results['rms'], marker='s', label='Yoshida-4', color='blue')
    plt.xlabel('Force evaluations (total over run)', fontname="Georgia", fontsize=12)
    plt.ylabel('RMS relative energy error', fontname="Georgia", fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
    plt.title(f'Accuracy vs Force-evals (T_total={T_total})', fontname='Georgia')
    plt.legend(
        prop={'family': 'Georgia', 'size': 10},
        frameon=True,
        edgecolor='black',
        framealpha=1.0,
        loc='upper right'
    )
    plt.tight_layout()
    plt.grid(False)
    p1 = os.path.join(outdir, "Accuracy_Force-evals .png")
    plt.savefig(p1, dpi=400)
    plt.close()

    
    plt.figure(figsize=(4.5, 4))
    plt.loglog(vv_results['runtime'], vv_results['rms'], marker='o', label='VV', color='black')
    plt.loglog(y_results['runtime'], y_results['rms'], marker='s', label='Yoshida-4', color='blue')
    plt.xlabel('Runtime (seconds)', fontname="Georgia", fontsize=12)
    plt.ylabel('RMS relative energy error', fontname="Georgia", fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
    plt.title(f'Accuracy vs Runtime (T_total={T_total})', fontname='Georgia')
    plt.legend(
        prop={'family': 'Georgia', 'size': 10},
        frameon=True,
        edgecolor='black',
        framealpha=1.0,
        loc='upper right'
    )
    plt.tight_layout()
    plt.grid(False)
    p1 = os.path.join(outdir, "Accuracy_Runtime.png")
    plt.savefig(p1, dpi=400)
    plt.close()

    
    print("\nSummary table (per integrator):")
    print("Integrator | dt | force-evals | time(s) | RMS relE | max relE")
    for i,dt in enumerate(vv_results['dt']):
        print(f"VV    {dt:.1e}  {vv_results['force_evals'][i]:6d}  {vv_results['runtime'][i]:6.3f}  {vv_results['rms'][i]:.3e}  {vv_results['max'][i]:.3e}")
    for i,dt in enumerate(y_results['dt']):
        print(f"Y4    {dt:.1e}  {y_results['force_evals'][i]:6d}  {y_results['runtime'][i]:6.3f}  {y_results['rms'][i]:.3e}  {y_results['max'][i]:.3e}")

    
    np.savez("bench_vv_yoshida_results.npz", vv=vv_results, yoshida=y_results)
    print("\nSaved results to bench_vv_yoshida_results.npz")
