import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Utility MLP ----
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128,128), out_dim=1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ---- Symplectic GNN that is rotation-invariant by construction ----
class SymplecticInvariantGNN(nn.Module):
    """
    Input:
      q: (N,3) positions (float tensor, requires_grad=True for autodiff)
      p: (N,3) momenta (float tensor, requires_grad=True)
    Output:
      H: scalar Hamiltonian (batching not included — it's for a single system here)
      dynamics: time derivatives dq_dt, dp_dt computed from H
    """
    def __init__(self, node_feat_dim=16, edge_feat_dim=32, mlp_hidden=(128,128)):
        super().__init__()
        # Edge MLP : maps invariant edge features to scalar message
        # edge features used: [dist, inv_dist, p_i·p_j, p_i·r_ij, p_j·r_ij, node_scalar_i, node_scalar_j]
        self.edge_mlp = MLP(in_dim=6, hidden=mlp_hidden, out_dim=edge_feat_dim)
        # Node MLP : maps aggregated edge messages to node scalar
        self.node_mlp = MLP(in_dim=edge_feat_dim, hidden=mlp_hidden, out_dim=node_feat_dim)
        # Final readout: map node features to scalar contributions then sum -> H
        self.readout = MLP(in_dim=node_feat_dim, hidden=mlp_hidden, out_dim=1)

    def forward(self, q, p):
        """
        q, p: (N,3) tensors with requires_grad = True if you want to compute dynamics
        returns H (scalar), dq_dt (N,3), dp_dt (N,3)
        """
        N = q.shape[0]
        # build all-pairs edges (i->j), skip i==j
        idx_i = torch.arange(N, device=q.device).unsqueeze(1).repeat(1, N).view(-1)  # (N*N,)
        idx_j = torch.arange(N, device=q.device).unsqueeze(0).repeat(N, 1).view(-1)  # (N*N,)
        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]
        qi = q[idx_i]          # (E,3)
        qj = q[idx_j]
        pi = p[idx_i]
        pj = p[idx_j]
        rij = qj - qi          # vector from i to j
        dist = torch.norm(rij, dim=1, keepdim=True) + 1e-9  # (E,1)
        inv_dist = 1.0 / dist
        rij_unit = rij * (inv_dist)  # (E,3)
        # build invariant edge features:
        # 1) distance scalar, 2) inv distance, 3) p_i·p_j, 4) p_i·r_ij, 5) p_j·r_ij, 6) (optional) dot of positions (qi·qj) could be added
        pdot = (pi * pj).sum(dim=1, keepdim=True)
        pi_dot_r = (pi * rij).sum(dim=1, keepdim=True)
        pj_dot_r = (pj * rij).sum(dim=1, keepdim=True)
        edge_feat = torch.cat([dist, inv_dist, pdot, pi_dot_r, pj_dot_r, dist*inv_dist], dim=1)  # (E,6)

        # edge -> message scalar vector
        e_msg = self.edge_mlp(edge_feat)  # (E, edge_feat_dim)

        # aggregate messages per receiver node (i)
        node_msg = torch.zeros((N, e_msg.shape[1]), device=q.device)
        node_msg = node_msg.index_add(0, idx_i, e_msg)  # sum messages incoming to i

        # node MLP -> node features -> node scalar
        node_hidden = self.node_mlp(node_msg)  # (N, node_feat_dim)
        node_scalar = self.readout(node_hidden).squeeze(-1)  # (N,)

        # Global Hamiltonian is sum of node scalars (could also add pairwise symmetric term / external potential)
        H = node_scalar.sum()  # scalar

        # Compute symplectic dynamics via autodiff:
        # dq_dt = dH/dp, dp_dt = - dH/dq
        # make sure q,p require grad
        dq_dt = torch.autograd.grad(H, p, create_graph=True)[0]
        dp_dt = -torch.autograd.grad(H, q, create_graph=True)[0]

        return H, dq_dt, dp_dt

# ---- Small demonstration ----
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = 'cpu'
    N = 6
    q = torch.randn(N,3, device=device, requires_grad=True)
    p = torch.randn(N,3, device=device, requires_grad=True)

    model = SymplecticInvariantGNN()
    H, dq_dt, dp_dt = model(q, p)
    print("H:", H.item())
    print("dq_dt shape:", dq_dt.shape, "dp_dt shape:", dp_dt.shape)

    # Example: take an Euler step (just demonstration; for symplectic integration, use symplectic integrators)
    dt = 1e-3
    q_new = q + dt * dq_dt
    p_new = p + dt * dp_dt


