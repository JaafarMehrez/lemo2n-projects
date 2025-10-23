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

def incr_force_count(n=1):
    global FORCE_EVALS
    FORCE_EVALS += n


def force(q, soft=0.0):
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

def run_integrator(step_func, q0, p0, dt, n_steps, store_every=1):
    q = q0.copy()
    p = p0.copy()
    qs = []
    ps = []
    Ts = []
    Vs = []
    Es = []

    for i in range(n_steps):
        q, p = step_func(q, p, dt)
        if (i % store_every) == 0:
            qs.append(q.copy())
            ps.append(p.copy())
            T = kinetic(p)
            V = potential(q)
            Ts.append(T)
            Vs.append(V)
            Es.append(T+V)
    return np.array(qs), np.array(ps), np.array(Ts), np.array(Vs), np.array(Es)

dt = 0.001               
n_steps = 100000        

q0 = np.array([0.2, 0.0])
p0 = np.array([0.0, 2.0])

print("Running velocity-Verlet...")
vv_q, vv_p, vv_T, vv_V, vv_E = run_integrator(vv_step, q0, p0, dt, n_steps)

print("Running Yoshida 4th-order composition (3 substeps per base step)...")
y_q, y_p, y_T, y_V, y_E = run_integrator(yoshida_step, q0, p0, dt, n_steps)

print("Running Blanes-Moan composition (7 substeps per base step)...")
bm_q, bm_p, bm_T, bm_V, bm_E = run_integrator(blanes_moan_step, q0, p0, dt, n_steps)

plt.figure(figsize=(4.5, 4))
plt.plot(vv_q[:,0], vv_q[:,1], lw=0.6, label='VV', color='black')
plt.plot(y_q[:,0], y_q[:,1], lw=0.6, label='Yoshida-4', color='blue')
plt.plot(bm_q[:,0], bm_q[:,1], lw=0.6, label='BM-7', color='red')
plt.scatter([q0[0]],[q0[1]], c='g', s=20, label='start')
plt.plot(0, 0, 'yo', label='Central Body', color='red')
plt.gca().set_aspect('equal')
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0,
    loc='upper right'
)
plt.title('Orbits: Velocity-Verlet vs Yoshida-4 vs Blanes-Moan-7',fontname="Georgia")
plt.xlabel('x', fontname="Georgia", fontsize=12)
plt.ylabel('y', fontname="Georgia", fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.tight_layout()
plt.grid(False)
p1 = os.path.join(outdir, "OrbitsVelocity-VerletvsYoshida-Blanes-Moan.png")
plt.savefig(p1, dpi=400)
plt.close()

t = np.arange(len(vv_E)) * dt
E0_vv = vv_E[0]
E0_y = y_E[0]
E0_bm=bm_E[0]
rel_vv = (vv_E - E0_vv) / abs(E0_vv)
rel_y = (y_E - E0_y) / abs(E0_y)
rel_bm = (bm_E - E0_bm) / abs(E0_bm)

plt.figure(figsize=(4.5, 4))
plt.plot(t, rel_vv, label='VV rel E error', linewidth=0.7, color='black')
plt.plot(t, rel_y, label='Yoshida-4 rel E error', linewidth=0.7, color='blue')
plt.plot(t, rel_bm, label='Blanes-Moan-7 rel E error', linewidth=0.7, color='red')
plt.yscale('symlog', linthresh=1e-12)  
plt.xlabel('time', fontname="Georgia", fontsize=12)
plt.ylabel('relative energy error', fontname="Georgia", fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0,
    loc='upper right'
) 
plt.title('Relative energy error (E(t)-E(0))/|E(0)|', fontname="Georgia")
plt.tight_layout()
plt.grid(False)
p1 = os.path.join(outdir, "RelativeEnergyError.png")
plt.savefig(p1, dpi=400)
plt.close()

zoom = int(2000)  
plt.figure(figsize=(4.5, 4))
plt.plot(t[:zoom], rel_vv[:zoom], label='VV', linewidth=0.8, color='black')
plt.plot(t[:zoom], rel_y[:zoom], label='Yoshida-4', linewidth=0.8, color='blue')
plt.plot(t[:zoom], rel_bm[:zoom], label='Blanes-Moan-7', linewidth=0.8, color='red')
plt.xlabel('time',fontname="Georgia", fontsize=12)
plt.ylabel('relative energy error', fontname="Georgia", fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0,
    loc='upper right'
)
plt.title('Zoom: early-time relative energy error', fontname="Georgia")
plt.tight_layout()
plt.grid(False)
p1 = os.path.join(outdir, "Early-timeRelativeEnergyError.png")
plt.savefig(p1, dpi=400)
plt.close()

np.save("blanes-moan_q.npy", bm_q)
np.save("blanes-moan_p.npy", bm_p)
np.save("yoshida_q.npy", y_q)
np.save("yoshida_p.npy", y_p)
np.save("vv_q.npy", vv_q)
np.save("vv_p.npy", vv_p)

print("Done. Saved blanes-moan_q.npy, blanes-moan_p.npy, yoshida_q.npy, yoshida_p.npy, vv_q.npy, vv_p.npy")