import h5py

with h5py.File('processed.h5', 'r') as f:
    print('datasets:', list(f.keys()))
    # metadata (reader used, etc.)
    print('attrs:', dict(f.attrs))

    #pos_unwrapped = f['positions_unwrapped'][:]          # shape (n_frames, n_atoms, 3)
    #pos_wrapped   = f['positions_wrapped'][:]
    #velocities    = f.get('velocities')[:] if 'velocities' in f else None
    charges       = f.get('charges')[:] if 'charges' in f else None

    #print('frames, atoms, coords:', pos_unwrapped.shape)
    #if velocities is not None:
       # print('velocities shape:', velocities.shape)
    if charges is not None:
        print('charges shape:', charges.shape)

