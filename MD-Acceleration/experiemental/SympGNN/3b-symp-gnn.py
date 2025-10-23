import torch
torch.set_default_dtype(torch.float64)
import numpy as np
import copy

np.random.seed(0)
torch.manual_seed(0)

n_predicted_time_steps = 256

# -------------------------
# Simple, pure-PyTorch GNN
# -------------------------
class SimpleGNN(torch.nn.Module):
    """
    Interprets the 12-d input as:
      - 2 channels (p and q)
      - 3 nodes
      - 2 coords per node  -> shape (2,3,2) -> flattened to 12

    Node feature: concat(p_i (2), q_i (2)) -> 4-d per node

    Message passing: fully connected graph (including self-edge),
    message network processes (h_i, h_j, h_i - h_j) and aggregates by sum.
    Final pooling -> scalar output.
    """
    def __init__(self, node_feat_dim=4, hidden=128, n_message_layers=3, n_mp_steps=3):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden = hidden
        self.n_mp_steps = n_mp_steps

        # Message MLP (input: concat(h_i, h_j, h_i - h_j) -> produce message dim = node_feat_dim)
        self.msg_net = torch.nn.Sequential(
            torch.nn.Linear(node_feat_dim * 3, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, node_feat_dim)
        )

        # Update MLP (input: concat(h_i, aggregated_message) -> new h_i)
        self.update_net = torch.nn.Sequential(
            torch.nn.Linear(node_feat_dim * 2, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, node_feat_dim)
        )

        # Final readout MLP: pool node features -> scalar
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(node_feat_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1)
        )

    def forward(self, x):
        # x: (B, 12)
        B = x.shape[0]

        # Recover (B, 2, 3, 2): 2 channels (p,q), 3 nodes, 2 coords
        x_ch = x.view(B, 2, 3, 2)
        # Permute to (B, 3, 2, 2) then flatten last two dims -> node features (B, 3, 4)
        node_feats = x_ch.permute(0, 2, 1, 3).reshape(B, 3, -1)  # (B, n_nodes=3, node_feat_dim=4)

        h = node_feats  # initial node states, (B, 3, 4)

        # Message passing steps
        for _ in range(self.n_mp_steps):
            # Prepare pairwise combinations: hi (B,3,1,F), hj (B,1,3,F) -> broadcast to (B,3,3,F)
            hi = h.unsqueeze(2)  # (B,3,1,F)
            hj = h.unsqueeze(1)  # (B,1,3,F)
            hi_exp = hi.expand(-1, -1, 3, -1)
            hj_exp = hj.expand(-1, 3, -1, -1)
            diff = hi_exp - hj_exp  # (B,3,3,F)

            # Edge features: concat(hi, hj, hi-hj) -> (B,3,3, F*3)
            edge_input = torch.cat([hi_exp, hj_exp, diff], dim=-1)

            # Apply message net to each edge: result shape (B,3,3,node_feat_dim)
            m_ij = self.msg_net(edge_input)

            # Aggregate messages for each target node i: sum_j m_ij -> (B,3,node_feat_dim)
            m_i = m_ij.sum(dim=2)

            # Update node features
            h = self.update_net(torch.cat([h, m_i], dim=-1))

        # Readout: pool across nodes (sum) then final MLP to scalar
        graph_feat = h.sum(dim=1)  # (B, node_feat_dim)
        out = self.readout(graph_feat)  # (B, 1)
        return out


# -------------------------
# Replace previous Net with one that uses SimpleGNN
# -------------------------
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gnn = SimpleGNN(node_feat_dim=4, hidden=128, n_mp_steps=3)

    def forward(self, x):
        # plus = gnn(x)
        # xminus = [-x[..., :6], x[..., 6:]] concatenated as before
        plus = self.gnn(x)
        xminus = torch.concatenate([-x[..., :6], x[..., 6:]], dim=-1)
        minus = self.gnn(xminus)
        return (plus + minus) / 2.0

net = Net()

p = torch.tensor(np.load("p.npy"))
q = torch.tensor(np.load("q.npy"))

dataset = torch.utils.data.TensorDataset(torch.stack([p[:-n_predicted_time_steps],
                                                     q[:-n_predicted_time_steps],
                                                     p[n_predicted_time_steps:],
                                                     q[n_predicted_time_steps:]], dim=1))

def get_random_rotation():
    theta = np.random.uniform(0, 2 * np.pi)
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def apply_random_rotation(sample):
    # sample shape: (4, 3, 2)  -> matrix-multiply last dim by 2x2 rotation
    rotation = get_random_rotation()
    return sample @ torch.tensor(rotation, dtype=torch.float64)

train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
)
validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=8,
    shuffle=False,
)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.7)

best_state_dict = None
best_validation_rmse = float('inf')

for epoch in range(20):
    for batch in train_dataloader:
        optimizer.zero_grad()
        sample_list = torch.split(batch[0], 1, dim=0)
        rotated_samples = [apply_random_rotation(sample.squeeze(0)) for sample in sample_list]
        pq = torch.stack([r[:2] for r in rotated_samples], dim=0)
        pq_prime = torch.stack([r[2:] for r in rotated_samples], dim=0)
        input = ((pq + pq_prime) / 2.0).reshape(-1, 12).requires_grad_(True)
        target = pq_prime - pq
        target = target.reshape(-1, 12)
        output = net(input)
        derivatives = torch.autograd.grad(
            outputs=output,
            inputs=input,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True
        )[0]
        predictions = torch.concatenate([-derivatives[:, 6:], derivatives[:, :6]], dim=1)
        loss = torch.nn.functional.mse_loss(predictions, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

    validation_sum_of_squared_errors = 0.0
    validation_num_samples = 0

    
    for batch in validation_dataloader:
        sample_list = torch.split(batch[0], 1, dim=0)
        #rotated_samples = [apply_random_rotation(sample.squeeze(0)) for sample in sample_list]
        pq = torch.stack([r[:2] for r in rotated_samples], dim=0)
        pq_prime = torch.stack([r[2:] for r in rotated_samples], dim=0)
        input = ((pq + pq_prime) / 2.0).reshape(-1, 12).requires_grad_(True)
        target = pq_prime - pq
        target = target.reshape(-1, 12)
        output = net(input)
        derivatives = torch.autograd.grad(
            outputs=output,
            inputs=input,
            grad_outputs=torch.ones_like(output),
            )[0]
        predictions = torch.concatenate([-derivatives[:, 6:], derivatives[:, :6]], dim=1)
        validation_sum_of_squared_errors += torch.sum((predictions - target) ** 2).item()
        validation_num_samples += predictions.numel()

    validation_rmse = np.sqrt(validation_sum_of_squared_errors / validation_num_samples)
    if validation_rmse < best_validation_rmse:
        best_validation_rmse = validation_rmse
        best_state_dict = copy.deepcopy(net.state_dict())
    print(f"Epoch {epoch + 1}, Validation RMSE: {validation_rmse:.7f}, learning rate: {scheduler.get_last_lr()[0]}")

net.load_state_dict(best_state_dict)
torch.save(net.state_dict(), "symplectic_gnn.ckpt")


