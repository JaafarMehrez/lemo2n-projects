import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

# Import the modified functions
from trajcast_comp import compute_particle_position_error, compare_instantaneous_temperature

def run_trajectory_comparison(pred_traj_path, ref_traj_path):
    """
    Example of how to use the trajectory comparison functions.
    """
    print("Loading trajectories...")
    
    # Compute position errors
    print("Computing position errors...")
    position_errors, dt = compute_particle_position_error(
        pred_traj_path,
        ref_traj_path,
        mode="rmse"
    )
    
    # Compute temperature differences
    print("Computing temperature differences...")
    try:
        temp_differences, _ = compare_instantaneous_temperature(
            pred_traj_path,
            ref_traj_path
        )
        has_temperatures = True
    except Exception as e:
        print(f"Could not compute temperature differences: {e}")
        has_temperatures = False
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot position errors
    frames = np.arange(len(position_errors))
    ax1.plot(frames, position_errors, 'r-', linewidth=2)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('RMSE (Å)')
    ax1.set_title('Position Errors (Predicted vs Reference)')
    ax1.grid(True, alpha=0.3)
    
    # Plot temperature differences if available
    if has_temperatures:
        ax2.plot(frames[:len(temp_differences)], temp_differences, 'b-', linewidth=2)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Temperature Difference (K)')
        ax2.set_title('Temperature Differences')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"Average position error: {position_errors.mean():.3f} ± {position_errors.std():.3f} Å")
    print(f"Maximum position error: {position_errors.max():.3f} Å")
    
    if has_temperatures:
        print(f"Average temperature difference: {temp_differences.mean():.1f} ± {temp_differences.std():.1f} K")
    
    return {
        'position_errors': position_errors,
        'temperature_differences': temp_differences if has_temperatures else None
    }

# Run the comparison
if __name__ == "__main__":
    pred_trajectory = "predicted_trajectory.xyz"
    ref_trajectory = "reference_trajectory.xyz"
    
    results = run_trajectory_comparison(pred_trajectory, ref_trajectory)
