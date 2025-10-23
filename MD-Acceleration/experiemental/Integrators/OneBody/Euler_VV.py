# Author: Jaafar Mehrez, jaafarmehrez@sjtu.edu.cn

import numpy as np
import matplotlib.pyplot as plt
import os


outdir = "./Data-OneBody/"
os.makedirs(outdir, exist_ok=True)

dt = 0.001
n_steps = 100000

q = np.array([0.2, 0.0])       
p = np.array([0.0, 2.0])

euler_traj = [q.copy()]
euler_momenta = [p.copy()]
euler_kinetic = [np.dot(p, p) / 2.0]  
euler_potential = [-1 / np.linalg.norm(q)]
#eps = 1e-4

for _ in range(n_steps):
    f = -q / np.linalg.norm(q)**3
    #f = -q / (np.linalg.norm(q)**2 + eps**2)**1.5
    q = q + dt * p
    p = p + dt * f
    euler_traj.append(q.copy())
    euler_kinetic.append(np.dot(p, p) / 2.0)
    euler_momenta.append(p.copy())
    euler_potential.append(-1 / np.linalg.norm(q))

L_euler = np.cross(euler_traj, euler_momenta)
print(L_euler)

euler_traj = np.array(euler_traj)
euler_kinetic = np.array(euler_kinetic)
euler_potential = np.array(euler_potential)
euler_total = euler_kinetic + euler_potential

# Reset initial conditions
q = np.array([0.5, 0.0])
p = np.array([0.0, 1.0])
verlet_traj = [q.copy()]
verlet_momenta = [p.copy()]  
verlet_kinetic = [np.dot(p, p) / 2.0]  
verlet_potential = [-1 / np.linalg.norm(q)]

#F = -q / (np.linalg.norm(q)**2 + eps**2)**1.5 # Initial potential energy
F = -q / np.linalg.norm(q)**3

for _ in range(n_steps):
    p = p + 0.5 * dt * F
    q = q + dt * p
   #F = -q / (np.linalg.norm(q)**2 + eps**2)**1.5
    F = -q / np.linalg.norm(q)**3
    p = p + 0.5 * dt * F
    verlet_traj.append(q.copy())
    verlet_momenta.append(p.copy())
    verlet_kinetic.append(np.dot(p, p) / 2.0)
    verlet_potential.append(-1 / np.linalg.norm(q))

L_verlet = np.cross(verlet_traj, verlet_momenta)
print(L_verlet)

verlet_traj = np.array(verlet_traj)
verlet_kinetic = np.array(verlet_kinetic)
verlet_potential = np.array(verlet_potential)
verlet_total = verlet_kinetic + verlet_potential



plt.figure(figsize=(4.5, 4))
plt.plot(verlet_traj[:, 0], verlet_traj[:, 1], label='Verlet',color='black')
plt.plot(euler_traj[:, 0], euler_traj[:, 1], label='Euler', linestyle='--',color='blue')
plt.plot(0, 0, 'yo', label='Central Body', color='red')
plt.scatter([q[0]],[q[1]], c='g', s=20, label='start')
plt.axis('equal')
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0,
    loc='upper right'
)
plt.title("Orbit Simulation: Euler vs Verlet", fontname="Georgia")
plt.xlabel("x", fontname="Georgia", fontsize=12)
plt.ylabel("y", fontname="Georgia", fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.xlim(verlet_traj[:, 0].min() - 0.1, verlet_traj[:, 0].max() + 0.1)
plt.ylim(verlet_traj[:, 1].min() - 0.1, verlet_traj[:, 1].max() + 0.1)
plt.grid(False)
plt.tight_layout()
p1 = os.path.join(outdir, "OrbitalSimulation.png")
plt.savefig(p1, dpi=400)
plt.close()


plt.figure()
plt.plot(euler_kinetic-euler_kinetic.mean(), label="Kinetic", color='black')
plt.plot(euler_potential-euler_potential.mean(), label="Potential", color='blue')
plt.plot(euler_total-euler_total.mean(), label="Total", color='red')
plt.xlabel("Time Steps", fontname="Georgia", fontsize=12)
plt.ylabel("Energy", fontname="Georgia", fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0,
    loc='upper right'
)
plt.grid(False)
plt.tight_layout()
p1 = os.path.join(outdir, "EulerEnergy.png")
plt.savefig(p1, dpi=400)
plt.close()



plt.figure()
plt.plot(verlet_kinetic-verlet_kinetic.mean(), label="Kinetic", color='black')
plt.plot(verlet_potential-verlet_potential.mean(), label="Potential",color='blue')
plt.plot(verlet_total-verlet_total.mean(), label="Total", color='red')
plt.xlabel("Time Steps",fontname="Georgia", fontsize=12)
plt.ylabel("Energy", fontname="Georgia", fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
plt.legend(
    prop={'family': 'Georgia', 'size': 10},
    frameon=True,
    edgecolor='black',
    framealpha=1.0,
    loc='upper right'
)
plt.grid(False)
plt.tight_layout()
p1 = os.path.join(outdir, "VerletEnergy.png")
plt.savefig(p1, dpi=400)
plt.close()



np.save("q.npy", verlet_traj)
np.save("p.npy", verlet_momenta)
