#!/usr/bin/env python3
import random

# Read data
with open("nvt_temperature.dat") as f:
    data = [line.split() for line in f if line.strip() and not line.startswith('#')]

timesteps = [int(d[0]) for d in data if len(d) >= 2]
temps = [float(d[1]) for d in data if len(d) >= 2]
times = [t * 0.0005 for t in timesteps]

# Find candidates
candidates = [(t, time, temp) for t, time, temp in zip(timesteps, times, temps) 
              if 290 <= temp <= 310]

# Select 5 with 0.1 ps separation
random.shuffle(candidates)
selected = []
for cand in candidates:
    if len(selected) >= 5:
        break
    t, time, temp = cand
    if all(abs(time - sel[1]) >= 0.1 for sel in selected):
        selected.append(cand)

# Output
print("Extract these timesteps from nvt_equil.lammpstrj:")
for i, (t, time, temp) in enumerate(sorted(selected, key=lambda x: x[0]), 1):
    print(f"{i}. Timestep: {t:8d}  Time: {time:7.3f} ps  Temp: {temp:6.1f} K")
