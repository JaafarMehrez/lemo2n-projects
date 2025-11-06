import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

from physical_behaviour import track_min_max_distances_in_molecule
from physical_behaviour import track_average_distance_from_molecule_center_of_mass
from physical_behaviour import track_instantaneous_temperature

def run_validation_analysis(trajectory_file):
    """
    Run all validation tests on a trajectory.
    
    Args:
        trajectory_file: Path to trajectory file (XYZ, etc.)
    """
    
    # Load trajectory with ASE
    try:
        trajectory = read(trajectory_file, index=':')
        print(f"Loaded: {len(trajectory[0])} atoms, {len(trajectory)} frames")
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return None
    
    print("Running validation tests...")
    
    # 1. Min/Max distances
    min_dists, max_dists = track_min_max_distances_in_molecule(trajectory)
    print("Min distances:", min_dists)
    
    # 2. Average distance from center of mass
    mean_com_dists = track_average_distance_from_molecule_center_of_mass(trajectory)
    
    # 3. Instantaneous temperature (if velocities are available)
    try:
        temps = track_instantaneous_temperature(trajectory)
        has_temperatures = True
    except AttributeError as e:
        print(f"No velocities found in trajectory - skipping temperature calculation: {e}")
        has_temperatures = False
    except Exception as e:
        print(f"Error calculating temperature: {e}")
        has_temperatures = False
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    frames = np.arange(len(trajectory))
    
    # Plot 1: Distance evolution
    axes[0,0].plot(frames, min_dists, 'b-', label='Min distance', linewidth=2)
    axes[0,0].plot(frames, max_dists, 'r-', label='Max distance', linewidth=2)
    axes[0,0].set_ylabel('Distance (Å)')
    axes[0,0].set_xlabel('Frame')
    axes[0,0].set_title('Minimum and Maximum Distances')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Distance from COM
    axes[0,1].plot(frames, mean_com_dists, 'g-', linewidth=2)
    axes[0,1].set_ylabel('Distance (Å)')
    axes[0,1].set_xlabel('Frame')
    axes[0,1].set_title('Average Distance from Center of Mass')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Temperature (if available)
    if has_temperatures:
        axes[1,0].plot(frames, temps, 'orange', linewidth=2)
        axes[1,0].axhline(y=temps.mean(), color='red', linestyle='--', 
                         label=f'Average: {temps.mean():.1f} K')
        axes[1,0].set_ylabel('Temperature (K)')
        axes[1,0].set_xlabel('Frame')
        axes[1,0].set_title('Instantaneous Temperature')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    else:
        # Hide the temperature subplot if no data
        axes[1,0].set_visible(False)
    
    # Plot 4: Distance distributions
    axes[1,1].hist(min_dists, alpha=0.7, label='Min distances', bins=20)
    axes[1,1].hist(max_dists, alpha=0.7, label='Max distances', bins=20)
    axes[1,1].set_xlabel('Distance (Å)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Distance Distributions')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'min_distances': min_dists,
        'max_distances': max_dists, 
        'mean_com_distances': mean_com_dists,
        'temperatures': temps if has_temperatures else None
    }

# Example with specific file paths
if __name__ == "__main__":
    # Replace with your actual trajectory file path
    trajectory_file = "first-traj.extxyz"  # ASE can read XYZ, etc.
    
    try:
        results = run_validation_analysis(trajectory_file)
        
        if results is not None:
            # Print statistical summary
            print("\n" + "="*50)
            print("VALIDATION SUMMARY")
            print("="*50)
            print(f"Minimum distance stats: {results['min_distances'].mean():.3f} ± {results['min_distances'].std():.3f} Å")
            print(f"Maximum distance stats: {results['max_distances'].mean():.3f} ± {results['max_distances'].std():.3f} Å")
            print(f"COM distance stats: {results['mean_com_distances'].mean():.3f} ± {results['mean_com_distances'].std():.3f} Å")
            
            if results['temperatures'] is not None:
                print(f"Temperature stats: {results['temperatures'].mean():.1f} ± {results['temperatures'].std():.1f} K")
                
    except FileNotFoundError:
        print(f"Could not find trajectory file. Please check the file path.")
        print(f"Looking for: {trajectory_file}")
    except Exception as e:
        print(f"An error occurred: {e}")