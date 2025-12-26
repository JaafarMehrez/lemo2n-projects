import numpy as np

profile_file = 'energy_stats_para.profile'
restart_freq = 2000  # Must match 'restart' command in equil.in
threshold = 0.001    # Paper requirement for Paracetamol

print("Analyzing Paracetamol Equilibration...")
try:
    data = np.loadtxt(profile_file, comments='#')
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

steps = data[:, 0]
energies = data[:, 1]
mean_E = np.mean(energies)

print(f"Mean Energy: {mean_E:.4f} kcal/mol")
print(f"Threshold (0.1%): {abs(mean_E * threshold):.4f} kcal/mol")

candidates = []
for t, e in zip(steps, energies):
    if t > 0 and t % restart_freq == 0:
        rel_err = abs(e - mean_E) / abs(mean_E)
        if rel_err < threshold:
            candidates.append(int(t))

print(f"\nFound {len(candidates)} valid restart files.")
if len(candidates) >= 5:
    # Pick 5 spread out
    indices = np.linspace(0, len(candidates)-1, 5, dtype=int)
    selected = [candidates[i] for i in indices]
    print("Use these files for Production:")
    for s in selected:
        print(f"  restart.nvt.{s}")
else:
    print("Warning: Not enough candidates found. Run equilibration longer.")
