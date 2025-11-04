from ase.io import read, write
import numpy as np

frames = read('traj_with_displacement.extxyz', index=':')

N = len(frames)
n_train = int(0.70 * N)
n_val   = int(0.15 * N)

train = frames[:n_train]
val   = frames[n_train:n_train+n_val]
test  = frames[n_train+n_val:]

# Write all frames from each dataset to a single file
write('train.extxyz', train)
write('val.extxyz', val)
write('test.extxyz', test)

print(f"Total frames: {N}")
print(f"Training frames: {len(train)}")
print(f"Validation frames: {len(val)}")
print(f"Test frames: {len(test)}")
