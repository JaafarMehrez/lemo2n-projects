import numpy as np
import matplotlib.pyplot as plt
import torch
torch.set_default_dtype(torch.float64)

class DirectNet(torch.nn.Module):
    def __init__(self):
        super(DirectNet, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(12, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 12),
        )

    def forward(self, x):
        return 0.1 * self.mlp(x)
    
direct_net = DirectNet()
state_dict = torch.load("direct.ckpt")
direct_net.load_state_dict(state_dict)

n_steps = 4000

q = np.array([[1.0, 0.0], [-0.5, np.sqrt(3.0)/2.0], [-0.5, -np.sqrt(3.0)/2.0]])  # Initial position
p = np.array([[0.0, 1.0], [-np.sqrt(3.0)/2.0, -0.5], [np.sqrt(3.0)/2.0, -0.5]])*0.5      # Initial momentum

direct_traj = [q.copy()]
direct_kinetic = [np.sum(p**2) / 2.0]  # Initial kinetic energy
direct_potential = [-1 / np.linalg.norm(q[1] - q[0]) - 1 / np.linalg.norm(q[2] - q[0]) - 1 / np.linalg.norm(q[2] - q[1])]

for _ in range(n_steps):
    output = direct_net(torch.tensor(np.concatenate((p, q)).reshape(1, -1))).detach().numpy()
    output = output.reshape(1, 2, 3, 2) 
    p = p + output[0, 0]  # Update momentum
    q = q + output[0, 1]  # Update position
    direct_traj.append(q.copy())
    direct_kinetic.append(np.sum(p**2) / 2.0)
    direct_potential.append(-1 / np.linalg.norm(q[1] - q[0]) - 1 / np.linalg.norm(q[2] - q[0]) - 1 / np.linalg.norm(q[2] - q[1]))

direct_traj = np.array(direct_traj)
direct_kinetic = np.array(direct_kinetic)
direct_potential = np.array(direct_potential)
direct_total = direct_kinetic + direct_potential

class SimpleGNN(torch.nn.Module):
    def __init__(self, node_feat_dim=4, hidden=128, n_message_layers=3, n_mp_steps=3):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden = hidden
        self.n_mp_steps = n_mp_steps
        self.msg_net = torch.nn.Sequential(
            torch.nn.Linear(node_feat_dim * 3, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, node_feat_dim)
        )
        self.update_net = torch.nn.Sequential(
            torch.nn.Linear(node_feat_dim * 2, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, node_feat_dim)
        )
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(node_feat_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1)
        )
    def forward(self, x):
        B = x.shape[0]
        x_ch = x.view(B, 2, 3, 2)
        node_feats = x_ch.permute(0, 2, 1, 3).reshape(B, 3, -1)
        h = node_feats
        for _ in range(self.n_mp_steps):
            hi = h.unsqueeze(2)
            hj = h.unsqueeze(1)
            hi_exp = hi.expand(-1, -1, 3, -1)
            hj_exp = hj.expand(-1, 3, -1, -1)
            diff = hi_exp - hj_exp
            edge_input = torch.cat([hi_exp, hj_exp, diff], dim=-1)
            m_ij = self.msg_net(edge_input)
            m_i = m_ij.sum(dim=2)
            h = self.update_net(torch.cat([h, m_i], dim=-1))
        graph_feat = h.sum(dim=1)
        out = self.readout(graph_feat)
        return out

class SymplecticNet(torch.nn.Module):
    def __init__(self):
        super(SymplecticNet, self).__init__()
        self.gnn = SimpleGNN(node_feat_dim=4, hidden=128, n_mp_steps=3)

    def forward(self, x):
        plus = self.gnn(x)
        xminus = torch.concatenate([-x[..., :6], x[..., 6:]], dim=-1)
        minus = self.gnn(xminus)
        return (plus + minus) / 2.0

symplectic_net = SymplecticNet()
state_dict = torch.load("symplectic_gnn.ckpt")
symplectic_net.load_state_dict(state_dict)

n_steps = 4000
accuracy_threshold = 1e-8
dt = 0.0001 * 256

# Initial conditions (position and momenta)
q = np.array([[1.0, 0.0], [-0.5, np.sqrt(3.0)/2.0], [-0.5, -np.sqrt(3.0)/2.0]])  # Initial position
p = np.array([[0.0, 1.0], [-np.sqrt(3.0)/2.0, -0.5], [np.sqrt(3.0)/2.0, -0.5]])*0.5      # Initial momentum

symplectic_traj = [q.copy()]
symplectic_kinetic = [np.sum(p**2) / 2.0]  # Initial kinetic energy
symplectic_potential = [-1 / np.linalg.norm(q[1] - q[0]) - 1 / np.linalg.norm(q[2] - q[0]) - 1 / np.linalg.norm(q[2] - q[1])]

for _ in range(n_steps):
    # Verlet-like step to initialize
    pp = p.copy()
    qq = q.copy()

    # F = -qq / np.linalg.norm(qq)**3
    # pp = pp + 0.5 * dt * F
    # qq = qq + dt * pp
    # F = -qq / np.linalg.norm(qq)**3
    # pp = pp + 0.5 * dt * F

    F = np.zeros_like(qq)
    F[0] = -1 * (qq[0] - qq[1]) / np.linalg.norm(qq[0] - qq[1])**3 - (qq[0] - qq[2]) / np.linalg.norm(qq[0] - qq[2])**3
    F[1] = -1 * (qq[1] - qq[0]) / np.linalg.norm(qq[1] - qq[0])**3 - (qq[1] - qq[2]) / np.linalg.norm(qq[1] - qq[2])**3
    F[2] = -1 * (qq[2] - qq[0]) / np.linalg.norm(qq[2] - qq[0])**3 - (qq[2] - qq[1]) / np.linalg.norm(qq[2] - qq[1])**3
    pp = pp + dt * F
    qq = qq + dt * pp

    n_iter = 0
    accuracy = np.inf
    pbar = (pp + p) / 2.0
    qbar = (qq + q) / 2.0
    pbar2 = (pp + p) / 2.0
    qbar2 = (qq + q) / 2.0
    while accuracy > accuracy_threshold:
        pbar = pbar * 0.7 + pbar2 * 0.3
        qbar = qbar * 0.7 + qbar2 * 0.3
        input = torch.tensor(np.concatenate((pbar, qbar))).requires_grad_(True)
        output = symplectic_net(input.reshape(1, -1))
        output.backward()
        derivatives = input.grad.numpy().reshape(1, -1)
        pp = p - derivatives[0, 6:].reshape(3, 2)  # Update momentum
        qq = q + derivatives[0, :6].reshape(3, 2)  # Update position
        pbar2 = (pp + p) / 2.0
        qbar2 = (qq + q) / 2.0
        accuracy = np.linalg.norm(pbar2 - pbar) + np.linalg.norm(qbar2 - qbar)
        n_iter += 1
    print(n_iter)

    p = pp
    q = qq
      
    symplectic_traj.append(q.copy())
    symplectic_kinetic.append(np.sum(p**2) / 2.0)
    symplectic_potential.append(-1 / np.linalg.norm(q[1] - q[0]) - 1 / np.linalg.norm(q[2] - q[0]) - 1 / np.linalg.norm(q[2] - q[1]))

symplectic_traj = np.array(symplectic_traj)
symplectic_kinetic = np.array(symplectic_kinetic)
symplectic_potential = np.array(symplectic_potential)
symplectic_total = symplectic_kinetic + symplectic_potential

n_steps = 4000
dt = 0.0001 * 256

# Reset initial conditions
q = np.array([[1.0, 0.0], [-0.5, np.sqrt(3.0)/2.0], [-0.5, -np.sqrt(3.0)/2.0]])  # Initial position
p = np.array([[0.0, 1.0], [-np.sqrt(3.0)/2.0, -0.5], [np.sqrt(3.0)/2.0, -0.5]])*0.5       # Initial momentum

verlet_traj = [q.copy()]
verlet_momenta = [p.copy()]  # Store momenta for Verlet method
verlet_kinetic = [np.sum(p**2) / 2.0]  # Initial kinetic energy
verlet_potential = [-1 / np.linalg.norm(q[1] - q[0]) - 1 / np.linalg.norm(q[2] - q[0]) - 1 / np.linalg.norm(q[2] - q[1])]

F = np.zeros_like(q)
# Calculate initial forces
F[0] = -1 * (q[0] - q[1]) / np.linalg.norm(q[0] - q[1])**3 - (q[0] - q[2]) / np.linalg.norm(q[0] - q[2])**3
F[1] = -1 * (q[1] - q[0]) / np.linalg.norm(q[1] - q[0])**3 - (q[1] - q[2]) / np.linalg.norm(q[1] - q[2])**3
F[2] = -1 * (q[2] - q[0]) / np.linalg.norm(q[2] - q[0])**3 - (q[2] - q[1]) / np.linalg.norm(q[2] - q[1])**3

for _ in range(n_steps):
    p = p + 0.5 * dt * F
    q = q + dt * p
    F = np.zeros_like(q)
    F[0] = -1 * (q[0] - q[1]) / np.linalg.norm(q[0] - q[1])**3 - (q[0] - q[2]) / np.linalg.norm(q[0] - q[2])**3
    F[1] = -1 * (q[1] - q[0]) / np.linalg.norm(q[1] - q[0])**3 - (q[1] - q[2]) / np.linalg.norm(q[1] - q[2])**3
    F[2] = -1 * (q[2] - q[0]) / np.linalg.norm(q[2] - q[0])**3 - (q[2] - q[1]) / np.linalg.norm(q[2] - q[1])**3
    p = p + 0.5 * dt * F
    verlet_traj.append(q.copy())
    verlet_momenta.append(p.copy())
    verlet_kinetic.append(np.sum(p**2) / 2.0)
    verlet_potential.append(-1 / np.linalg.norm(q[1] - q[0]) - 1 / np.linalg.norm(q[2] - q[0]) - 1 / np.linalg.norm(q[2] - q[1]))

verlet_traj = np.array(verlet_traj)
verlet_kinetic = np.array(verlet_kinetic)
verlet_potential = np.array(verlet_potential)
verlet_total = verlet_kinetic + verlet_potential

# font size of 14 everywhere
plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(np.arange(len(verlet_total)), verlet_total, "k:", label="Verlet", linewidth=1.0)
ax.plot(np.arange(len(direct_total)), direct_total, "k--", label="Direct", linewidth=1.0)
ax.plot(np.arange(len(symplectic_total)), symplectic_total, "k-", label="Symplectic", linewidth=1.0)
ax.set_xlabel("Time step")
ax.set_ylabel("Energy (arb. units)")
ax.set_ylim(-8.0, -0.0)

# Inset axes: positioned inside the main plot
# Change the numbers to move/resize inset
inset_ax = fig.add_axes([0.53, 0.2, 0.5, 0.5])
inset_ax.plot(verlet_traj[:, 0, 0], verlet_traj[:, 0, 1], "r:", label='Verlet', linewidth=0.5)
inset_ax.plot(verlet_traj[:, 1, 0], verlet_traj[:, 1, 1], "b:", label='Verlet', linewidth=0.5)
inset_ax.plot(verlet_traj[:, 2, 0], verlet_traj[:, 2, 1], "g:", label='Verlet', linewidth=0.5)
inset_ax.plot(symplectic_traj[:, 0, 0], symplectic_traj[:, 0, 1], "r-", label='Symplectic NN', linewidth=0.5)
inset_ax.plot(symplectic_traj[:, 1, 0], symplectic_traj[:, 1, 1], "b-", label='Symplectic NN', linewidth=0.5)
inset_ax.plot(symplectic_traj[:, 2, 0], symplectic_traj[:, 2, 1], "g-", label='Symplectic NN', linewidth=0.5)
inset_ax.plot(direct_traj[:, 0, 0], direct_traj[:, 0, 1], "r--", label='Direct NN', linewidth=0.5)
inset_ax.plot(direct_traj[:, 1, 0], direct_traj[:, 1, 1], "b--", label='Direct NN', linewidth=0.5)
inset_ax.plot(direct_traj[:, 2, 0], direct_traj[:, 2, 1], "g--", label='Direct NN', linewidth=0.5)
inset_ax.set_aspect('equal')
inset_ax.set_xlim(-1.1, 1.1)
inset_ax.set_ylim(-1.1, 1.1)

inset_ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

ax.legend(loc="lower left")

plt.tight_layout()
plt.savefig("3body.jpg", dpi=400)
