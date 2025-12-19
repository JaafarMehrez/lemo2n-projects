import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from trajcast.model.models import EfficientTrajCastModel
from trajcast.data.dataset import AtomicGraphDataset
import os
import time
from scipy import stats

# ================ Device Configuration ================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Set precision - ensure this matches training precision
torch.set_default_dtype(torch.float64)

# ================ Model Loading ================
print("\n" + "="*50)
print("Loading model...")
start_time = time.time()

# Load model configuration
model = EfficientTrajCastModel.build_from_yaml("../config.yaml")

# Load trained weights
checkpoint_path = "../model_params.pt"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

print(f"Model loaded in {time.time()-start_time:.2f} seconds")
print(f"Model on device: {next(model.parameters()).device}")

# ================ Dataset Preparation ================
print("\n" + "="*50)
print("Preparing dataset...")

test_set_dict = {
    "root": ".",
    "name": "interface_test",
    "cutoff_radius": 4.5,
    "files": ["../data/test.extxyz"],
    "rename": True,
    "atom_type_mapper": {
        14: 0,  # Si -> 0
        8: 1,   # O -> 1
        1: 2,   # H -> 2
    },
}

test_set = AtomicGraphDataset(**test_set_dict)

# Get normalization scales from the model
vel_scale = model._encoding.Normalization.stds["update_velocities"].item()
disp_scale = model._encoding.Normalization.stds["displacements"].item()

print(f"Velocity scale: {vel_scale:.4f}")
print(f"Displacement scale: {disp_scale:.4f}")
print(f"Test set size: {len(test_set)}")

# ================ Data Loading and Ground Truth ================
# Move ground truth to numpy for comparison
true_vel = test_set.update_velocities.detach().cpu().numpy()
true_disp = test_set.displacements.detach().cpu().numpy()

# Create DataLoader with appropriate batch size
batch_size = 5  # Adjust based on GPU memory
data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# ================ Model Inference ================
print("\n" + "="*50)
print("Running inference...")

pred_vel = []
pred_disp = []
batch_times = []

with torch.no_grad():
    for i, batch in enumerate(data_loader):
        batch_start = time.time()

        # Move batch to device
        batch = batch.to(device)

        # Forward pass
        batch = model(batch)

        # Extract predictions and apply inverse scaling
        batch_disp = batch.target[:, 0:3].cpu().numpy() * disp_scale
        batch_vel = batch.target[:, 3:].cpu().numpy() * vel_scale

        pred_disp.append(batch_disp)
        pred_vel.append(batch_vel)

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if (i + 1) % 5 == 0:  # Print progress every 5 batches
            print(f"Processed {i+1}/{len(data_loader)} batches")

# Concatenate predictions
pred_disp = np.concatenate(pred_disp)
pred_vel = np.concatenate(pred_vel)

print(f"Inference completed. Average batch time: {np.mean(batch_times):.3f}s")

# ================ Analysis Metrics ================
print("\n" + "="*50)
print("Computing analysis metrics...")

def compute_metrics(true, pred, name="Metric"):
    """Compute comprehensive evaluation metrics"""
    # Flatten for scalar metrics
    true_flat = true.flatten()
    pred_flat = pred.flatten()

    # Basic error metrics
    mae = np.mean(np.abs(pred_flat - true_flat))
    rmse = np.sqrt(np.mean((pred_flat - true_flat)**2))

    # Relative errors
    rel_error = np.mean(np.abs(pred_flat - true_flat) / (np.abs(true_flat) + 1e-10))

    # Correlation metrics
    r2 = 1 - np.sum((true_flat - pred_flat)**2) / np.sum((true_flat - np.mean(true_flat))**2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(true_flat, pred_flat)

    print(f"\n{name}:")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Relative Error: {rel_error:.4%}")
    print(f"  R² Score: {r2:.6f}")
    print(f"  Pearson R: {r_value:.6f}")
    print(f"  Regression slope: {slope:.6f} ± {std_err:.6f}")

    return {
        'mae': mae,
        'rmse': rmse,
        'rel_error': rel_error,
        'r2': r2,
        'r_value': r_value,
        'slope': slope
    }

# Compute metrics for velocities and displacements
vel_metrics = compute_metrics(true_vel, pred_vel, "Velocity Analysis")
disp_metrics = compute_metrics(true_disp, pred_disp, "Displacement Analysis")

# ================ Visualization ================
print("\n" + "="*50)
print("Generating plots...")

# Create output directory for plots
output_dir = "rollout_analysis"
os.makedirs(output_dir, exist_ok=True)

# Function to create and save scatter plots with enhanced styling
def create_scatter_plot(true, pred, xlabel, ylabel, title, filename,
                        lim_range=0.15, aspect='equal'):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Scatter plot with transparency
    scatter = ax.scatter(true.flatten(), pred.flatten(),
                        s=10, alpha=0.5, c='blue', edgecolors='none')

    # Perfect prediction line
    ax.plot([-lim_range, lim_range], [-lim_range, lim_range],
            color='red', ls='--', lw=2, label='Perfect prediction')

    # Set limits and labels
    ax.set_xlim(-lim_range, lim_range)
    ax.set_ylim(-lim_range, lim_range)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add legend
    ax.legend()

    # Add aspect ratio
    if aspect == 'equal':
        ax.set_aspect('equal')

    # Add text box with key metrics
    r2 = 1 - np.sum((true - pred)**2) / np.sum((true - np.mean(true))**2)
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true)**2))
    metrics_text = f"R² = {r2:.4f}\nMAE = {mae:.6f}\nRMSE = {rmse:.6f}"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

# Create velocity plot
create_scatter_plot(true_vel, pred_vel,
                   xlabel=r"Reference Updated Velocities $[\mathrm{\AA}/fs]$",
                   ylabel=r"Predicted Updated Velocities $[\mathrm{\AA}/fs]$",
                   title="Velocity Predictions",
                   filename="velocity_predictions.png",
                   lim_range=0.15,
                   aspect='equal')

# Create displacement plot
create_scatter_plot(true_disp, pred_disp,
                   xlabel=r"Reference Displacements $[\mathrm{\AA}]$",
                   ylabel=r"Predicted Displacements $[\mathrm{\AA}]$",
                   title="Displacement Predictions",
                   filename="displacement_predictions.png",
                   lim_range=0.35,
                   aspect='equal')

# ================ Additional Analysis: Error Distribution ================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Velocity error distribution
vel_error = pred_vel - true_vel
axes[0,0].hist(vel_error.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0,0].set_xlabel('Velocity Error [Å/fs]')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Velocity Error Distribution')
axes[0,0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0,0].grid(True, alpha=0.3)

# Displacement error distribution
disp_error = pred_disp - true_disp
axes[0,1].hist(disp_error.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
axes[0,1].set_xlabel('Displacement Error [Å]')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('Displacement Error Distribution')
axes[0,1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0,1].grid(True, alpha=0.3)

# Error vs magnitude
axes[1,0].scatter(np.abs(true_vel.flatten()), np.abs(vel_error.flatten()),
                  alpha=0.5, s=10, c='blue')
axes[1,0].set_xlabel('|True Velocity| [Å/fs]')
axes[1,0].set_ylabel('|Error| [Å/fs]')
axes[1,0].set_title('Error vs True Velocity Magnitude')
axes[1,0].grid(True, alpha=0.3)

# Component-wise analysis
for i, comp in enumerate(['X', 'Y', 'Z']):
    axes[1,1].scatter(true_vel[:, i], pred_vel[:, i],
                      alpha=0.5, s=10, label=f'{comp}-component')
axes[1,1].plot([-0.15, 0.15], [-0.15, 0.15], 'r--', label='Perfect')
axes[1,1].set_xlabel('True Velocity [Å/fs]')
axes[1,1].set_ylabel('Predicted Velocity [Å/fs]')
axes[1,1].set_title('Component-wise Velocity Predictions')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "error_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: error_analysis.png")

# ================ Save Metrics to File ================
metrics_summary = {
    'velocities': vel_metrics,
    'displacements': disp_metrics,
    'inference_stats': {
        'avg_batch_time': np.mean(batch_times),
        'total_samples': len(test_set),
        'device_used': str(device),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
}

# Save metrics to JSON
import json
with open(os.path.join(output_dir, "metrics_summary.json"), 'w') as f:
    json.dump(metrics_summary, f, indent=2, default=str)

# Also save as text summary
with open(os.path.join(output_dir, "summary_report.txt"), 'w') as f:
    f.write("="*60 + "\n")
    f.write("MODEL ROLLOUT ANALYSIS REPORT\n")
    f.write("="*60 + "\n\n")

    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Model: EfficientTrajCastModel\n")
    f.write(f"Test samples: {len(test_set)}\n\n")

    f.write("VELOCITY METRICS:\n")
    f.write("-"*40 + "\n")
    for key, value in vel_metrics.items():
        f.write(f"{key:>15}: {value:.6f}\n")

    f.write("\nDISPLACEMENT METRICS:\n")
    f.write("-"*40 + "\n")
    for key, value in disp_metrics.items():
        f.write(f"{key:>15}: {value:.6f}\n")

    f.write(f"\nInference average batch time: {np.mean(batch_times):.3f}s\n")

print(f"\n" + "="*50)
print(f"Analysis complete! All outputs saved to '{output_dir}/'")
print(f"  - velocity_predictions.png")
print(f"  - displacement_predictions.png")
print(f"  - error_analysis.png")
print(f"  - metrics_summary.json")
print(f"  - summary_report.txt")
print("="*50)
