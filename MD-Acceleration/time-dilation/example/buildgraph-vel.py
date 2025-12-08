#!/usr/bin/env python3

from trajcast.data.dataset import AtomicGraphDataset
import numpy as np
import torch
import sys

# Optional plotting / networkx imports guarded
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False


def tensor_to_np(t):
    """Safely convert a torch tensor/np array to numpy on CPU. Return None for None input."""
    if t is None:
        return None
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    if isinstance(t, np.ndarray):
        return t
    try:
        return np.array(t)
    except Exception:
        return None


def get_first_attr_np(g, names):
    """
    Return the first attribute present in `names` from object g, converted to numpy (or None).
    This avoids using `or` on tensors (which is ambiguous).
    """
    for name in names:
        if hasattr(g, name):
            val = getattr(g, name)
            if val is not None:
                npv = tensor_to_np(val)
                if npv is not None:
                    return npv, name
                # if tensor_to_np returned None, still return raw val name info
                return None, name
    return None, None


def describe_graph(g, verbose=True):
    """Print defensive description of a single torch_geometric Data object."""
    # get keys safely
    try:
        keys = list(g.keys())
    except Exception:
        keys = [k for k in g.__dict__.keys() if not k.startswith('_')]

    if verbose:
        print("----- DATA KEYS -----")
        print(keys)
        print("\n----- BASIC SHAPES & TYPES -----")

    # num_nodes (try attribute then infer)
    num_nodes = getattr(g, "num_nodes", None)
    if num_nodes is None:
        pos_np, _ = get_first_attr_np(g, ("pos", "positions"))
        if pos_np is not None:
            try:
                num_nodes = int(pos_np.shape[0])
            except Exception:
                num_nodes = None

    print("num_nodes:", num_nodes)
    print("num_edges (directed count):", getattr(g, "num_edges", None))

    # edge_index
    ei_np, ei_name = get_first_attr_np(g, ("edge_index",))
    try:
        ei_shape = tuple(ei_np.shape) if ei_np is not None else None
    except Exception:
        ei_shape = str(type(getattr(g, "edge_index", None)))
    print("edge_index shape:", ei_shape)

    # generic node feature x (may be None)
    x_np, x_name = get_first_attr_np(g, ("x",))
    if x_np is None:
        print("x: None")
    else:
        print("x shape:", tuple(x_np.shape), "dtype:", x_np.dtype)

    # positions: check common names
    pos_np, pos_name = get_first_attr_np(g, ("positions", "pos", "r", "coords"))
    if pos_np is not None:
        print(f"{pos_name} shape:", tuple(pos_np.shape), "dtype:", pos_np.dtype)
        try:
            mins = np.min(pos_np, axis=0)
            maxs = np.max(pos_np, axis=0)
            print("positions min per axis:", mins, "max per axis:", maxs)
        except Exception:
            pass
    else:
        print("positions: None or unavailable")

    # atomic numbers / types (safe)
    an_np, an_name = get_first_attr_np(g, ("atomic_numbers", "z"))
    if an_np is not None:
        try:
            print(f"{an_name} shape:", tuple(an_np.shape), "unique:", np.unique(an_np))
        except Exception:
            print(f"{an_name} present but could not compute unique values")
    else:
        print("atomic_numbers / z: None")

    at_np, at_name = get_first_attr_np(g, ("atom_types",))
    if at_np is not None:
        try:
            print(f"{at_name} shape:", tuple(at_np.shape), "unique:", np.unique(at_np))
        except Exception:
            print(f"{at_name} present but unreadable")

    # velocities and velocity cutoff (vel_max)
    vel_np, vel_name = get_first_attr_np(g, ("velocities", "vel"))
    if vel_np is not None:
        print(f"{vel_name} shape:", tuple(vel_np.shape), "dtype:", vel_np.dtype)
        try:
            vel_magnitudes = np.linalg.norm(vel_np, axis=1)
            print("velocity magnitudes: min, max, mean, std:", 
                  float(vel_magnitudes.min()), float(vel_magnitudes.max()), 
                  float(vel_magnitudes.mean()), float(vel_magnitudes.std()))
        except Exception:
            pass
    
    # Check for vel_max (velocity cutoff parameter)
    vel_max_np, vel_max_name = get_first_attr_np(g, ("vel_max", "velocity_cutoff", "max_velocity"))
    if vel_max_np is not None:
        print(f"{vel_max_name} (velocity cutoff):", vel_max_np)
        if vel_np is not None:
            import numpy as np
            vel_magnitudes = np.linalg.norm(vel_np, axis=1)
            print(f"  velocities within cutoff? {np.all(vel_magnitudes <= vel_max_np + 1e-6)}")
            print(f"  max velocity / vel_max ratio: {float(vel_magnitudes.max() / vel_max_np):.4f}")
    else:
        print("vel_max (velocity cutoff): Not found in graph")

    # partial charges (safe)
    pc_np, pc_name = get_first_attr_np(g, ("partial_charges", "partial_charge"))
    if pc_np is None:
        print("partial_charges: None or not present")
    else:
        print("partial_charges shape:", tuple(pc_np.shape))
        try:
            print("partial_charges stats: min, max, mean, std:",
                  float(np.min(pc_np)), float(np.max(pc_np)), float(np.mean(pc_np)), float(np.std(pc_np)))
        except Exception:
            pass
        print("any NaNs in partial_charges?", bool(np.isnan(pc_np).any()))

    # edge_attr if present
    ea_np, ea_name = get_first_attr_np(g, ("edge_attr",))
    if ea_np is not None:
        try:
            print("edge_attr shape:", tuple(ea_np.shape))
        except Exception:
            print("edge_attr present, type:", type(ea_np))

    # print device/dtypes for tensors in keys (best-effort)
    for k in keys:
        try:
            v = getattr(g, k)
            if torch.is_tensor(v):
                print(f"{k}: device={v.device} dtype={v.dtype} shape={tuple(v.shape)}")
        except Exception:
            pass

    # edge distances if positions present
    if pos_np is not None and ei_np is not None:
        try:
            vecs = pos_np[ei_np[0].astype(int)] - pos_np[ei_np[1].astype(int)]
            import numpy as np
            dists = np.linalg.norm(vecs, axis=1)
            print("edge distances (directed): min, max, mean, std:",
                  float(dists.min()), float(dists.max()), float(dists.mean()), float(dists.std()))
            print("edge distance median:", float(np.median(dists)))
        except Exception as e:
            print("failed to compute edge distances:", e)

    # degree and connected components (safe)
    if ei_np is not None:
        try:
            num_nodes_infer = int(num_nodes) if num_nodes is not None else (pos_np.shape[0] if pos_np is not None else None)
            if num_nodes_infer is not None:
                idxs = np.concatenate([ei_np[0].astype(int), ei_np[1].astype(int)])
                deg = np.bincount(idxs, minlength=num_nodes_infer)
                print("degree: min, max, mean:", int(deg.min()), int(deg.max()), float(deg.mean()))
            else:
                print("degree: cannot infer num_nodes for degree calculation")

            if HAS_NX:
                G = nx.Graph()
                edges = list(zip(ei_np[0].astype(int).tolist(), ei_np[1].astype(int).tolist()))
                G.add_edges_from(edges)
                cc = list(nx.connected_components(G))
                print("connected components (count):", len(cc))
                sizes = sorted([len(c) for c in cc], reverse=True)[:5]
                print("largest component sizes:", sizes)
            else:
                print("networkx not installed — skipping connected components")
        except Exception as e:
            print("degree / components check failed:", e)

    if verbose:
        print("----- END DESCRIPTION -----\n")


def plot_graph(g, show_edges=True, max_edges_to_plot=2000):
    """Quick scatter plot of positions. Falls back cleanly if matplotlib not present."""
    if not HAS_MPL:
        print("matplotlib not available; skipping plot.")
        return

    pos_np, pos_name = get_first_attr_np(g, ("pos", "positions"))
    if pos_np is None:
        print("No positions to plot.")
        return

    an_np, _ = get_first_attr_np(g, ("atomic_numbers", "z"))

    dim = pos_np.shape[1] if pos_np.ndim == 2 else 1
    fig = plt.figure(figsize=(6, 6))
    if dim == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2], c=an_np if an_np is not None else None, s=40)
        if show_edges:
            ei_np, _ = get_first_attr_np(g, ("edge_index",))
            if ei_np is not None:
                if ei_np.shape[1] > max_edges_to_plot:
                    print(f"Too many edges ({ei_np.shape[1]}) to draw — skipping edge lines.")
                else:
                    for i, j in zip(ei_np[0].astype(int), ei_np[1].astype(int)):
                        xs = [pos_np[i, 0], pos_np[j, 0]]
                        ys = [pos_np[i, 1], pos_np[j, 1]]
                        zs = [pos_np[i, 2], pos_np[j, 2]]
                        ax.plot(xs, ys, zs, linewidth=0.4)
    else:
        ax = fig.add_subplot(111)
        ax.scatter(pos_np[:, 0], pos_np[:, 1], c=an_np if an_np is not None else None, s=40)
        if show_edges:
            ei_np, _ = get_first_attr_np(g, ("edge_index",))
            if ei_np is not None:
                if ei_np.shape[1] > max_edges_to_plot:
                    print(f"Too many edges ({ei_np.shape[1]}) to draw — skipping edge lines.")
                else:
                    for i, j in zip(ei_np[0].astype(int), ei_np[1].astype(int)):
                        ax.plot([pos_np[i, 0], pos_np[j, 0]], [pos_np[i, 1], pos_np[j, 1]], linewidth=0.4)
    ax.set_title("AtomicGraph positions (color = atomic number)")
    plt.tight_layout()
    plt.show()


def look_for_time_reversal_pairs(dataset, idx=0, search_N=200):
    """Brute-force check: find a graph with identical positions and negated velocities."""
    g0 = dataset[idx]
    pos0_np, _ = get_first_attr_np(g0, ("pos", "positions"))
    vel0_np, _ = get_first_attr_np(g0, ("velocities", "vel"))
    if pos0_np is None or vel0_np is None:
        print("No pos/vel fields to check time-reversal invariance.")
        return

    N = min(len(dataset), search_N)
    found = False
    for i in range(N):
        if i == idx:
            continue
        gi = dataset[i]
        posi_np, _ = get_first_attr_np(gi, ("pos", "positions"))
        veli_np, _ = get_first_attr_np(gi, ("velocities", "vel"))
        if posi_np is None or veli_np is None:
            continue
        if np.allclose(pos0_np, posi_np, atol=1e-6):
            if np.allclose(vel0_np, -veli_np, atol=1e-6):
                print(f"Found time-reversed pair: {idx} <--> {i}")
                found = True
                break
    if not found:
        print(f"No time-reversal pair found within first {N} graphs.")


def dataset_summary(dataset, sample_N=100):
    """Compute quick dataset-level histograms from up-to sample_N random graphs."""
    N = len(dataset)
    sample_N = min(sample_N, N)
    idxs = np.random.choice(N, sample_N, replace=False)
    node_counts = []
    edge_counts = []
    species_counts = {}
    pc_min = []
    pc_max = []
    vel_max_values = []
    
    for i in idxs:
        g = dataset[i]
        num_nodes = getattr(g, "num_nodes", None)
        if num_nodes is None:
            p_np, _ = get_first_attr_np(g, ("pos", "positions"))
            if p_np is not None:
                num_nodes = p_np.shape[0]
        node_counts.append(int(num_nodes) if num_nodes is not None else 0)
        
        ei_np, _ = get_first_attr_np(g, ("edge_index",))
        edge_counts.append(int(ei_np.shape[1]) if ei_np is not None else 0)
        
        an_np, _ = get_first_attr_np(g, ("atomic_numbers", "z"))
        if an_np is not None:
            for a in np.unique(an_np):
                species_counts[int(a)] = species_counts.get(int(a), 0) + int(np.sum(an_np == a))
        
        pc_np, _ = get_first_attr_np(g, ("partial_charges",))
        if pc_np is not None:
            pc_min.append(float(np.min(pc_np)))
            pc_max.append(float(np.max(pc_np)))
        
        # Check for vel_max in each graph
        vel_max_np, _ = get_first_attr_np(g, ("vel_max", "velocity_cutoff", "max_velocity"))
        if vel_max_np is not None:
            vel_max_values.append(float(vel_max_np))

    print("----- DATASET SUMMARY (sampled) -----")
    print("num_graphs (total):", N)
    print("sampled graphs:", sample_N)
    
    if node_counts:
        print("nodes per graph: min, max, mean, median :", 
              int(min(node_counts)), int(max(node_counts)),
              float(np.mean(node_counts)), float(np.median(node_counts)))
    
    if edge_counts:
        print("edges per graph (directed): min, max, mean :", 
              int(min(edge_counts)), int(max(edge_counts)), float(np.mean(edge_counts)))
    
    if species_counts:
        print("species counts (sample sum):", species_counts)
    
    if pc_min:
        print("partial_charges (sample) overall min/max:", float(min(pc_min)), float(max(pc_max)))
    
    # Report vel_max statistics if found
    if vel_max_values:
        vel_max_arr = np.array(vel_max_values)
        print("velocity cutoff (vel_max) statistics:")
        print(f"  found in {len(vel_max_values)}/{sample_N} sampled graphs")
        print(f"  min: {vel_max_arr.min():.6f}")
        print(f"  max: {vel_max_arr.max():.6f}")
        print(f"  mean: {vel_max_arr.mean():.6f}")
        print(f"  std: {vel_max_arr.std():.6f}")
        if np.allclose(vel_max_arr, vel_max_arr[0]):
            print(f"  constant value: {vel_max_arr[0]:.6f}")
    else:
        print("vel_max (velocity cutoff): Not found in sampled graphs")
        # Also check dataset object itself
        dataset_vel_max = getattr(dataset, 'vel_max', None)
        if dataset_vel_max is not None:
            print(f"Dataset-level vel_max: {dataset_vel_max}")
    
    print("----- END SUMMARY -----\n")


def main():
    filename = "proc-traj.extxyz"
    dataset = AtomicGraphDataset(
        root='./data/ref-12/',
        name='graph_extxyz',
        cutoff_radius=4.0,
        files=filename,
        atom_type_mapper={14:1,8:2,1:3},
        time_reversibility=True,
        rename=True
    )

    print(f"Number of graphs: {len(dataset)}")
    
    # Check for dataset-level vel_max
    dataset_vel_max = getattr(dataset, 'vel_max', None)
    if dataset_vel_max is not None:
        print(f"Dataset-level vel_max (velocity cutoff): {dataset_vel_max}")
    
    g = dataset[0]
    print(f"First graph num_nodes: {getattr(g, 'num_nodes', None)}\n")

    # Detailed single-graph description
    describe_graph(g)

    # Visual quick-check (if matplotlib available)
    plot_graph(g, show_edges=True)

    # Time reversal check (if dataset created with time_reversibility)
    look_for_time_reversal_pairs(dataset, idx=0, search_N=200)

    # Dataset-wide sampled summary
    dataset_summary(dataset, sample_N=100)

    # Additional velocity analysis
    vel_np, vel_name = get_first_attr_np(g, ("velocities", "vel"))
    if vel_np is not None:
        import numpy as np
        print("\n----- ADDITIONAL VELOCITY ANALYSIS -----")
        vel_magnitudes = np.linalg.norm(vel_np, axis=1)
        print(f"Velocity magnitude stats:")
        print(f"  min: {vel_magnitudes.min():.6f}")
        print(f"  max: {vel_magnitudes.max():.6f}")
        print(f"  mean: {vel_magnitudes.mean():.6f}")
        print(f"  std: {vel_magnitudes.std():.6f}")
        
        # Check if velocities follow Maxwell-Boltzmann distribution
        from scipy import stats
        try:
            # Fit Maxwell-Boltzmann distribution to velocity magnitudes
            # Parameter a = sqrt(kT/m) where k is Boltzmann constant, T is temperature, m is mass
            params = stats.maxwell.fit(vel_magnitudes)
            print(f"Maxwell-Boltzmann fit parameters (scale, loc, shape): {params}")
            
            # KS test for goodness of fit
            ks_stat, ks_p = stats.kstest(vel_magnitudes, 'maxwell', args=params)
            print(f"KS test for Maxwell-Boltzmann: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
            if ks_p > 0.05:
                print("  Velocities appear to follow Maxwell-Boltzmann distribution (p > 0.05)")
            else:
                print("  Velocities may NOT follow Maxwell-Boltzmann distribution")
        except ImportError:
            print("scipy not available for Maxwell-Boltzmann distribution analysis")
        except Exception as e:
            print(f"Could not fit Maxwell-Boltzmann distribution: {e}")

    import numpy as np
    # assume g is your Data object (dataset[0])
    pos = g.pos.detach().cpu().numpy()            # (N,3)
    ei = g.edge_index.detach().cpu().numpy()      # (2, E)
    shifts = g.shifts.detach().cpu().numpy()      # (E, 3) integer shifts
    cell = g.cell.detach().cpu().numpy()          # (3,3) lattice vectors (rows or columns - library dependent)
    
    # convert integer shifts to Cartesian translations
    # use dot(shifts, cell): shifts @ cell -> (E,3)
    shift_cart = shifts.dot(cell)
    
    # compute correct displacement vectors and distances (directed edges)
    vecs = pos[ei[0].astype(int)] - (pos[ei[1].astype(int)] + shift_cart)
    dists = np.linalg.norm(vecs, axis=1)
    
    print("correct edge distances: min, max, mean, std:", dists.min(), dists.max(), dists.mean(), dists.std())
    print("median (directed):", np.median(dists))
    
    # Many neighbor lists are stored directed (i->j and j->i). Keep only one direction:
    mask = ei[0] < ei[1]   # simple canonical filter; works when both directions present
    unique_ei = ei[:, mask]
    unique_dists = dists[mask]
    print("unique (undirected) edges:", unique_ei.shape[1])
    print("unique edge distances: min, max, mean, median:", unique_dists.min(), unique_dists.max(), unique_dists.mean(), np.median(unique_dists))

    # degree per node (undirected)
    num_nodes = pos.shape[0]
    idxs = np.concatenate([ei[0].astype(int), ei[1].astype(int)])
    deg = np.bincount(idxs, minlength=num_nodes)
    print("degree stats (min, max, mean):", deg.min(), deg.max(), deg.mean())

    # per-atom-type average degree
    atomic_numbers = g.atomic_numbers.detach().cpu().numpy().squeeze()  # shape (N,)
    for atom in np.unique(atomic_numbers):
        mask_nodes = (atomic_numbers == atom)
        print(f"atom {int(atom)} count:", mask_nodes.sum(), "avg degree:", deg[mask_nodes].mean())

if __name__ == "__main__":
    main()
