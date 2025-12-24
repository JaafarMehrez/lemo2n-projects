import torch
import ase.io
from trajcast.model.models import EfficientTrajCastModel
from torch_nl.neighbor_list import compute_neighborlist
import sys

# ---------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

torch.set_default_dtype(torch.float32)

print(f"--- DEBUG START ---")
print(f"Device: {device}")

# Load Frame & Wrap
try:
    start_frame = ase.io.read("../data/test.extxyz", index="-1")
    start_frame.wrap() # Force atoms into box
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 2. NEIGHBOR LIST
# ---------------------------------------------------------
print("Computing Neighbor List...", end=" ")

# Prepare tensors
pos = torch.tensor(start_frame.positions, dtype=torch.float32, device=device)
cell = torch.tensor(start_frame.cell.array, dtype=torch.float32, device=device).unsqueeze(0)
pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device).unsqueeze(0)
batch = torch.zeros(len(start_frame), dtype=torch.long, device=device)
cutoff = 4.5

# Compute
try:
    mapping, mapping_batch, shifts_idx = compute_neighborlist(
        cutoff, pos, cell, pbc, batch, self_interaction=False
    )
    print(f"Done. Edges: {mapping.shape[1]}")
except Exception as e:
    print(f"\nNL Failed: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 3. LOAD MODEL
# ---------------------------------------------------------
print("Loading Model...", end=" ")
try:
    model = EfficientTrajCastModel.build_from_yaml("../config.yaml")
    model.load_state_dict(torch.load("../model_params.pt", map_location=device))
    model.to(device)
    model.eval()
    print("Done.")
except Exception as e:
    print(f"\nModel Load Failed: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 4. FIX ATOM TYPES
# ---------------------------------------------------------
print("Mapping Atom Types...", end=" ")

# Try to find the trained species list
species_list = None
if hasattr(model, "hparams") and "species" in model.hparams:
    species_list = model.hparams["species"]
elif hasattr(model, "species"):
    species_list = model.species

# Create Mapping
if species_list is not None:
    z_mapping = {int(z): i for i, z in enumerate(species_list)}
else:
    unique_z = sorted(list(set(start_frame.numbers)))
    z_mapping = {int(z): i for i, z in enumerate(unique_z)}

# Apply Mapping (Map Z to Index 0, 1, 2...)
type_indices = [z_mapping.get(int(z), 0) for z in start_frame.numbers]
atom_types = torch.tensor(type_indices, dtype=torch.long, device=device).unsqueeze(1)
print("Done.")

# ---------------------------------------------------------
# 5. CONSTRUCT GRAPH & INJECT ATTRIBUTES
# ---------------------------------------------------------
from trajcast.data.atomic_graph import AtomicGraph

# Velocities
raw_vel = start_frame.get_velocities()
vel = torch.tensor(raw_vel, dtype=torch.float32, device=device) if raw_vel is not None else torch.zeros_like(pos)

# Masses (Per Atom)
# Must be (N, 1) and stored as 'atomic_masses'
masses = torch.tensor(start_frame.get_masses(), dtype=torch.float32, device=device).unsqueeze(1)

# [FIX] Total Mass (Per Graph)
# Since we have 1 graph, we sum all masses. Shape should be (1, 1) to match broadcasting
total_mass = masses.sum().reshape(1, 1)

# Initialize
test_graph = AtomicGraph(
    positions=pos,
    cell=cell[0],
    atomic_numbers=torch.tensor(start_frame.numbers, dtype=torch.int32, device=device),
    cutoff=cutoff,
    device=device
)

# INJECT EVERYTHING
test_graph.edge_index = mapping
test_graph.batch = batch
test_graph.pos = pos
test_graph.velocities = vel
test_graph.atom_types = atom_types
test_graph.atomic_masses = masses

# [FIX] Inject 'total_mass'
test_graph.total_mass = total_mass

# Shifts
shifts_idx = shifts_idx.to(device)
shifts = torch.matmul(shifts_idx.to(torch.float32), cell[0])
test_graph.shifts_idx = shifts_idx
test_graph.shifts = shifts


# ---------------------------------------------------------
# 6. RUN & DIAGNOSE (FINAL)
# ---------------------------------------------------------
print("\n--- RUNNING PREDICTION ---")

try:
    with torch.no_grad():
        # The output is the modified graph object itself
        result_graph = model(test_graph)

    print("\n--- RAW OUTPUT INSPECTION ---")

    # Check 'target' (Likely [Displacements, VelUpdates])
    if hasattr(result_graph, "target"):
        pred = result_graph.target
        print(f"Prediction Tensor ('target') Shape: {pred.shape}")

        # Assume format is [dx, dy, dz, dvx, dvy, dvz]
        displacements = pred[:, 0:3]
        vel_updates = pred[:, 3:6]

        # 1. Check Displacements
        max_disp = displacements.abs().max().item()
        mean_disp = displacements.abs().mean().item()
        print(f"\nPredicted Displacements:")
        print(f"  Max:  {max_disp:.5f} Å")
        print(f"  Mean: {mean_disp:.5f} Å")

        # 2. Check Velocity Updates
        max_vel = vel_updates.abs().max().item()
        print(f"\nPredicted Velocity Updates:")
        print(f"  Max:  {max_vel:.5f}")

        # --- DIAGNOSIS ---
        print("\n--- DIAGNOSIS ---")
        if torch.isnan(pred).any():
            print(">> CRITICAL FAILURE: Model output contains NaNs.")
            print("   This is why the simulation crashes.")
        elif max_disp > 2.0:
            print(">> CAUSE FOUND: EXPLODING DISPLACEMENTS")
            print(f"   Atoms are moving {max_disp:.2f} Å in a single step!")
            print("   This puts them outside the neighbor list range immediately.")
            print("   FIX: The model is untrained or the learning rate was too high.")
        elif max_disp == 0.0:
            print(">> WARNING: Model predicts EXACTLY ZERO change.")
            print("   Weights might be dead, or inputs aren't reaching the output.")
        else:
            print(">> PREDICTIONS LOOK REASONABLE")
            print("   Max displacement is small. The original crash is definitely due to")
            print("   atoms drifting out of the box boundaries over time.")
            print("   FIX: You MUST enable wrapping in your forecast loop.")

    else:
        print(">> ERROR: 'target' attribute not found in output graph.")
        print(f"   Available attributes: {result_graph.keys}")

except Exception as e:
    print(f"!!! CRASH DURING FORWARD PASS: {e}")
    import traceback
    traceback.print_exc()
