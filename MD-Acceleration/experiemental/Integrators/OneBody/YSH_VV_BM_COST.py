"""
Compare Velocity-Verlet (VV), Yoshida-4 (Y4) and Blanes-Moan 4th-order (BM4)
 - BM4 coefficients are the standard optimized Blanes-Moan 4th-order (6-stage symmetric).
 - Force evaluations are counted exactly by incrementing a global counter inside force().
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


def force(q, soft=0.0):
    incr_force_count(1)
    return -q / np.linalg.norm(q)**3
def potential(q, soft=0.0):
    return -1 / np.linalg.norm(q)
def kinetic(p):
    return np.dot(p, p) / 2.0

def vv_step(q, p, dt):
    F = force(q)                    # 1
    p_half = p + 0.5 * dt * F
    q_new = q + dt * p_half
    F_new = force(q_new)            # 1
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


b1 = 0.0792036964311957
b2 = 0.353172906049774
b3 = -0.0420650803577195
b4 = 1.0 - 2.0*(b1 + b2 + b3)   # symmetric: b = [b1,b2,b3,b4,b3,b2,b1]

# a1 is zero here (no initial drift), a2,a3,a4 given; a4 computed to satisfy symmetry.
a2 = 0.209515106613362
a3 = -0.143851773179818
a4 = 0.5 - (a2 + a3)  # so that a2+a3+a4 = 1/2 (centered symmetric drift)
# Sequence: B(b1) A(a2) B(b2) A(a3) B(b3) A(a4) B(b4) A(a4) B(b3) A(a3) B(b2) A(a2) B(b1)

def blanes_moan_step(q, p, dt):
    p = p + b1 * dt * force(q)
    q = q + a2 * dt * p
    p = p + b2 * dt * force(q)
    q = q + a3 * dt * p
    p = p + b3 * dt * force(q)
    q = q + a4 * dt * p
    p = p + b4 * dt * force(q)
    # symmetry back
    q = q + a4 * dt * p
    p = p + b3 * dt * force(q)
    q = q + a3 * dt * p
    p = p + b2 * dt * force(q)
    q = q + a2 * dt * p
    p = p + b1 * dt * force(q)
    return q, p

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
        return None, None, None, None, None, force_evals, runtime

def relative_energy_metrics(E):
    E0 = E[0]
    rel = (E - E0) / abs(E0)
    rms = np.sqrt(np.mean(rel**2))
    maxabs = np.max(np.abs(rel))
    return rms, maxabs


if __name__ == "__main__":

    q0 = np.array([0.5, 0.0])
    p0 = np.array([0.0, 1.0])


    T_total = 10.0 
    dt_list = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3]

    results = {
        'VV':   {'dt':[], 'force_evals':[], 'runtime':[], 'rms':[], 'max':[]},
        'Y4':   {'dt':[], 'force_evals':[], 'runtime':[], 'rms':[], 'max':[]},
        'BM4':  {'dt':[], 'force_evals':[], 'runtime':[], 'rms':[], 'max':[]},
    }

    for dt in dt_list:
        n_steps = int(np.round(T_total / dt))
        if n_steps < 2:
            continue
        print(f"\nTesting dt={dt:.1e} (n_steps={n_steps}) ...", flush=True)

        vv_q, vv_p, vv_T, vv_V, vv_E, vv_force, vv_time = run_integrator(vv_step, q0, p0, dt, n_steps)
        vv_rms, vv_max = relative_energy_metrics(vv_E)
        r = results['VV']
        r['dt'].append(dt); r['force_evals'].append(vv_force)
        r['runtime'].append(vv_time); r['rms'].append(vv_rms); r['max'].append(vv_max)
        print(f" VV: force_evals={vv_force}, time={vv_time:.3f}s, rms_relE={vv_rms:.3e}, max_relE={vv_max:.3e}")

        y_q, y_p, y_T, y_V, y_E, y_force, y_time = run_integrator(yoshida_step, q0, p0, dt, n_steps)
        y_rms, y_max = relative_energy_metrics(y_E)
        r = results['Y4']
        r['dt'].append(dt); r['force_evals'].append(y_force)
        r['runtime'].append(y_time); r['rms'].append(y_rms); r['max'].append(y_max)
        print(f" Y4: force_evals={y_force}, time={y_time:.3f}s, rms_relE={y_rms:.3e}, max_relE={y_max:.3e}")

        bm_q, bm_p, bm_T, bm_V, bm_E, bm_force, bm_time = run_integrator(blanes_moan_step, q0, p0, dt, n_steps)
        bm_rms, bm_max = relative_energy_metrics(bm_E)
        r = results['BM4']
        r['dt'].append(dt); r['force_evals'].append(bm_force)
        r['runtime'].append(bm_time); r['rms'].append(bm_rms); r['max'].append(bm_max)
        print(f" BM4: force_evals={bm_force}, time={bm_time:.3f}s, rms_relE={bm_rms:.3e}, max_relE={bm_max:.3e}")

    for integr in results:
        for k,v in results[integr].items():
            results[integr][k] = np.array(v)

    plt.figure(figsize=(4.5, 4))
    plt.loglog(results['VV']['force_evals'], results['VV']['rms'], marker='o', label='VV', color='black')
    plt.loglog(results['Y4']['force_evals'], results['Y4']['rms'], marker='s', label='Yoshida-4', color='blue')
    plt.loglog(results['BM4']['force_evals'], results['BM4']['rms'], marker='^', label='Blanes-Moan-4', color='red')
    plt.xlabel('Force evaluations (total over run)',fontname="Georgia", fontsize=12)
    plt.ylabel('RMS relative energy error',fontname="Georgia", fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
    plt.title(f'Accuracy vs Force-evals (T_total={T_total})', fontname="Georgia")
    plt.legend(
        prop={'family': 'Georgia', 'size': 10},
        frameon=True,
        edgecolor='black',
        framealpha=1.0,
        loc='upper right'
    )
    plt.grid(False)
    plt.tight_layout()
    p1 = os.path.join(outdir, "AccuracyForceVVYMB.png")
    plt.savefig(p1, dpi=400)
    plt.close()

    plt.figure(figsize=(4.5, 4))
    plt.loglog(results['VV']['runtime'], results['VV']['rms'], marker='o', label='VV', color='black')
    plt.loglog(results['Y4']['runtime'], results['Y4']['rms'], marker='s', label='Yoshida-4', color='blue')
    plt.loglog(results['BM4']['runtime'], results['BM4']['rms'], marker='^', label='Blanes-Moan-4', color='red')
    plt.xlabel('Runtime (seconds)',fontname="Georgia", fontsize=12)
    plt.ylabel('RMS relative energy error',fontname="Georgia", fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
    plt.title(f'Accuracy vs Runtime (T_total={T_total})',fontname="Georgia")
    plt.legend(
        prop={'family': 'Georgia', 'size': 10},
        frameon=True,
        edgecolor='black',
        framealpha=1.0,
        loc='upper right'
    )
    plt.grid(False)
    plt.tight_layout()
    p1 = os.path.join(outdir, "AccuracyRunTimeVVYMB.png")
    plt.savefig(p1, dpi=400)
    plt.close()

    print("\nSummary table (per integrator):")
    print("Integrator | dt | force-evals | time(s) | RMS relE | max relE")
    for i,dt in enumerate(results['VV']['dt']):
        print(f"VV    {dt:.1e}  {int(results['VV']['force_evals'][i]):6d}  {results['VV']['runtime'][i]:6.3f}  {results['VV']['rms'][i]:.3e}  {results['VV']['max'][i]:.3e}")
    for i,dt in enumerate(results['Y4']['dt']):
        print(f"Y4    {dt:.1e}  {int(results['Y4']['force_evals'][i]):6d}  {results['Y4']['runtime'][i]:6.3f}  {results['Y4']['rms'][i]:.3e}  {results['Y4']['max'][i]:.3e}")
    for i,dt in enumerate(results['BM4']['dt']):
        print(f"BM4   {dt:.1e}  {int(results['BM4']['force_evals'][i]):6d}  {results['BM4']['runtime'][i]:6.3f}  {results['BM4']['rms'][i]:.3e}  {results['BM4']['max'][i]:.3e}")

    np.savez("bench_vv_yoshida_blanesmoan_results.npz", vv=results['VV'], y4=results['Y4'], bm4=results['BM4'])
    print("\nSaved results to bench_vv_yoshida_blanesmoan_results.npz")
