"""
Compare velocity-Verlet (VV) vs a 4th-order Yoshida composition of VV
for the central -1/r problem (Kepler-like).
Author: Jaafar Mehrez, jaafarmehrez@sjtu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
import os

outdir = "./Data-OneBody/"
os.makedirs(outdir, exist_ok=True)

def force(q):
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

def run_integrator(step_func, q0, p0, dt, n_steps, store_every=1):
    qs = []
    ps = []
    Es = []
    Ts = []
    Vs = []
    q = q0.copy()
    p = p0.copy()
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


dt = 0.01               
n_steps = 10000        

q0 = np.array([0.2, 0.0])
p0 = np.array([0.0, 2.0])

print("Running velocity-Verlet...")
vv_q, vv_p, vv_T, vv_V, vv_E = run_integrator(vv_step, q0, p0, dt, n_steps)

print("Running Yoshida 4th-order composition (3 substeps per base step)...")
y_q, y_p, y_T, y_V, y_E = run_integrator(yoshida_step, q0, p0, dt, n_steps)

plt.figure(figsize=(4.5, 4))
plt.plot(vv_q[:,0], vv_q[:,1], lw=0.6, label='VV', color='black')
plt.plot(y_q[:,0], y_q[:,1], lw=0.6, label='Yoshida-4', color='blue')
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
plt.title('Orbits: Velocity-Verlet vs Yoshida-4',fontname="Georgia")
plt.xlabel('x', fontname="Georgia", fontsize=12)
plt.ylabel('y', fontname="Georgia", fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.tight_layout()
plt.grid(False)
p1 = os.path.join(outdir, "OrbitsVelocity-VerletvsYoshida-4.png")
plt.savefig(p1, dpi=400)
plt.close()

t = np.arange(len(vv_E)) * dt
E0_vv = vv_E[0]
E0_y = y_E[0]
rel_vv = (vv_E - E0_vv) / abs(E0_vv)
rel_y = (y_E - E0_y) / abs(E0_y)

plt.figure(figsize=(4.5, 4))
plt.plot(t, rel_vv, label='VV rel E error', linewidth=0.7, color='black')
plt.plot(t, rel_y, label='Yoshida-4 rel E error', linewidth=0.7, color='blue')
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

np.save("yoshida_q.npy", y_q)
np.save("yoshida_p.npy", y_p)
np.save("vv_q.npy", vv_q)
np.save("vv_p.npy", vv_p)

print("Done. Saved yoshida_q.npy, yoshida_p.npy, vv_q.npy, vv_p.npy")
