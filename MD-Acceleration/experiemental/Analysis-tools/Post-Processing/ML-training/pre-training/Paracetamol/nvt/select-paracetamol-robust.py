import numpy as np

# Configuration
profile_file = 'energy_stats_para.profile'
restart_freq = 2000
strict_threshold = 0.001
min_separation_steps = 2000 # Keep selected frames distinct

print("--- Analyzing Paracetamol Trajectory ---")

try:
    data = np.loadtxt(profile_file, comments='#')
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

steps = data[:, 0]
energies = data[:, 1]

# 1. Calculate Mean (Ignoring first 10% as equilibration)
start_idx = int(len(energies) * 0.1)
mean_E = np.mean(energies[start_idx:])
print(f"Mean Energy (last 90%): {mean_E:.4f} kcal/mol")

# 2. Collect all restart points and calculate errors
candidates = []
for t, e in zip(steps, energies):
    if t > 0 and t % restart_freq == 0:
        rel_err = abs(e - mean_E) / abs(mean_E)
        candidates.append({
            "step": int(t),
            "energy": e,
            "error": rel_err
        })

# 3. Sort by lowest error (closest to mean)
candidates.sort(key=lambda x: x["error"])

# 4. Select 5 distinct candidates
selected = []
print(f"\nTop Candidates found:")
print(f"{'Step':<10} | {'Energy':<12} | {'Rel Error':<12} | {'Status'}")
print("-" * 50)

for cand in candidates:
    # Check strict threshold
    status = "OK" if cand["error"] < strict_threshold else "Loose"
    
    # Simple check to ensure we don't pick identical or too-close frames
    # (Though with 50k steps and 2k freq, they are likely distinct enough)
    is_distinct = True
    for s in selected:
        if abs(cand["step"] - s["step"]) < min_separation_steps:
            is_distinct = False
            break
            
    if is_distinct:
        selected.append(cand)
        print(f"{cand['step']:<10} | {cand['energy']:<12.4f} | {cand['error']:<12.6f} | {status}")
    
    if len(selected) >= 5:
        break

# 5. Output
print("-" * 50)
if len(selected) < 5:
    print("Error: Simulation too short to find 5 restart files.")
else:
    # Sort by step number for logical ordering
    selected.sort(key=lambda x: x["step"])
    
    print("\nSUCCESS! Use these restart steps for production:")
    step_list = []
    for s in selected:
        print(f"  Step {s['step']} (Error: {s['error']:.5f})")
        step_list.append(str(s['step']))
    
    print("\nUpdate your submission script with:")
    print(f'STEPS="{" ".join(step_list)}"')
