import numpy as np

# --- Configuration ---
profile_file = 'energy_stats.profile'
restart_frequency = 1000  # You used 'restart 1000' in LAMMPS
threshold = 0.001         # The relative error threshold from the paper

def find_valid_candidates():
    print(f"--- Processing {profile_file} ---")
    
    # 1. Load the data (skipping the header comments)
    # Assuming columns are: [TimeStep, v_etot]
    try:
        data = np.loadtxt(profile_file, comments='#')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    timesteps = data[:, 0]
    energies = data[:, 1]

    # 2. Calculate the Mean Energy (mu)
    # We use the whole file assuming the system was reasonably settled 
    # (or you can slice it: energies[100:] to skip the first few frames)
    mean_energy = np.mean(energies)
    print(f"Mean Total Energy (mu): {mean_energy:.6f} eV")

    # 3. Filter for valid restart files
    valid_candidates = []

    print(f"\nSearching for restart steps with relative error < {threshold}...")
    print(f"{'TimeStep':<15} | {'Energy':<15} | {'Rel Error':<15} | {'Status'}")
    print("-" * 65)

    for t, e in zip(timesteps, energies):
        # We only care about steps where a restart file exists
        if t > 0 and t % restart_frequency == 0:
            
            # Calculate Relative Error
            rel_error = abs(e - mean_energy) / abs(mean_energy)
            
            is_valid = rel_error < threshold
            status = "ACCEPTED" if is_valid else "Rejected"
            
            if is_valid:
                valid_candidates.append(int(t))
                print(f"{int(t):<15} | {e:<15.6f} | {rel_error:<15.6f} | {status}")

    # 4. Summary and Instructions
    print("-" * 65)
    print(f"Found {len(valid_candidates)} valid restart files out of {int(timesteps[-1]/restart_frequency)}.")
    
    if len(valid_candidates) >= 5:
        # The paper requires selecting 5 configurations uniformly sampled or random
        # to ensure they are at least 1 ps apart.
        print("\nRecommended Selection (Spaced out for independence):")
        
        # Simple strategy: Pick 5 indices evenly spaced across the valid list
        indices = np.linspace(0, len(valid_candidates) - 1, 5, dtype=int)
        selected_steps = [valid_candidates[i] for i in indices]
        
        for i, step in enumerate(selected_steps):
            role = "Training" if i < 3 else ("Validation" if i == 3 else "Test")
            print(f"  {role}: restart.nvt.{step}")
    else:
        print("\nWARNING: Not enough valid candidates found. Your NVT simulation might need to run longer")
        print("or your thermostat relaxation time might be too aggressive/loose.")

if __name__ == "__main__":
    find_valid_candidates()
