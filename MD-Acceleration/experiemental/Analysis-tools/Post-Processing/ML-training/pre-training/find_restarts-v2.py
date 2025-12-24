import numpy as np

# --- Configuration ---
profile_file = 'energy_stats.profile'
restart_frequency = 1000   # Must match 'restart' command in LAMMPS
min_separation = 1000      # Steps between selected restarts (Paper says 1ps = 1000 steps)
num_candidates_needed = 5

def select_candidates():
    print(f"--- Analyzing {profile_file} ---")
    
    # 1. Load Data
    try:
        data = np.loadtxt(profile_file, comments='#')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Assuming format: [TimeStep, v_etot]
    timesteps = data[:, 0]
    energies = data[:, 1]

    # 2. Calculate Statistics over the whole trajectory
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    
    print(f"Mean Energy:   {mean_energy:.6f} eV")
    print(f"Std Deviation: {std_energy:.6f} eV")
    print(f"Rel. Fluctuation: {(std_energy/abs(mean_energy))*100:.4f}%")
    print("-" * 60)

    # 3. Identify Restart Frames
    # We create a list of (timestep, deviation_from_mean)
    candidates = []
    
    for t, e in zip(timesteps, energies):
        if t > 0 and t % restart_frequency == 0:
            deviation = abs(e - mean_energy)
            candidates.append((int(t), deviation, e))

    # 4. Sort by Deviation (closest to mean first)
    candidates.sort(key=lambda x: x[1])

    # 5. Select 5 Spaced-Out Candidates
    # We iterate through the best candidates and only pick them if they 
    # are far enough apart from those already picked.
    selected = []
    
    print(f"{'TimeStep':<10} | {'Energy':<12} | {'Deviation':<10} | {'Sigma Units'}")
    print("-" * 60)
    
    for cand in candidates:
        t, dev, e = cand
        
        # Check separation from currently selected candidates
        is_far_enough = True
        for sel in selected:
            if abs(t - sel[0]) < min_separation:
                is_far_enough = False
                break
        
        if is_far_enough:
            selected.append(cand)
            sigma_dist = dev / std_energy
            print(f"{t:<10} | {e:<12.4f} | {dev:<10.4f} | {sigma_dist:.2f} σ")
            
        if len(selected) >= num_candidates_needed:
            break

    # 6. Output Instructions
    print("-" * 60)
    if len(selected) < num_candidates_needed:
        print("WARNING: Could not find 5 separated candidates. Simulation might be too short.")
    else:
        # Sort by time for convenience
        selected.sort(key=lambda x: x[0])
        print("\nSUCCESS! Use these restart files for your 5 NVE runs:")
        roles = ["Train", "Train", "Train", "Validation", "Test"]
        for i, (t, _, _) in enumerate(selected):
            print(f"  {roles[i]}: restart.nvt.{t}")

if __name__ == "__main__":
    select_candidates()
