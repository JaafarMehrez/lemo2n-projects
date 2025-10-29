import os
import numpy as np
import h5py
from trajectory_processor import ASETrajectory, HDF5Trajectory
from _keys import (
    POSITIONS_KEY, VELOCITIES_KEY, PARTIAL_CHARGES_KEY, PBC_KEY, CELL_KEY, DISPLACEMENTS_KEY, 
    UPDATE_PARTIAL_CHARGES_KEY, UPDATE_VELOCITIES_KEY
)

ROOT = "../data/"
FILENAME = "dump-100.lammpstrj"
OUT_PREFIX = "traj_hdf5"
HDF5_PREFIX = "traj_hdf5"
FRAME_DT_FS = None

time_between_frames = 50  # time interval between frames in fs
timestep_forward = 100 # forward timestep

traj = ASETrajectory.read_from_file(
    root=ROOT,
    filename=FILENAME,
    frame_interval = time_between_frames,
    wrapper="lammps",
    wrapper_kwargs={"lammps_units":"real", "desired_units":"real", "type_mapping":{1:14,2:8}},
    index=":",
    apply_unwrapping=False,
)

print(f"Original trajectory frames: {traj.n_frames}")
print(f"Available fields: {traj.available_fields}")

traj.compute_additional_fields(
    add_fields={DISPLACEMENTS_KEY, PARTIAL_CHARGES_KEY, UPDATE_PARTIAL_CHARGES_KEY, UPDATE_VELOCITIES_KEY},
    time_step=int(timestep_forward / time_between_frames),
    truncate=True
)

print(f"Trajectory frames after processing: {len(traj.data)}")

traj.write_to_file(
    root=ROOT,
    filename_prefix=OUT_PREFIX,
    chosen_fields={POSITIONS_KEY, PARTIAL_CHARGES_KEY, PBC_KEY, CELL_KEY, 
                   VELOCITIES_KEY, DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY, UPDATE_PARTIAL_CHARGES_KEY},
    format="extxyz"
)

def ase_trajectory_to_dict_simple(ase_traj, chosen_fields):
    traj_dict = {}
    
    field_mapping = {
        'positions': 'positions',
        'partial_charges': 'partial_charges', 
        'update_partial_charges': 'update_partial_charges',
        'displacements': 'displacements',
        'cell': 'cell',
        'pbc': 'pbc',
        'velocities': 'update_velocities'
    }
    
    for h5_key, ase_field in field_mapping.items():
        if ase_field in chosen_fields:
            data_list = []
            
            for frame in ase_traj.data:
                if ase_field in frame.arrays:
                    data_list.append(frame.arrays[ase_field])
                elif ase_field in frame.info:
                    data_list.append(frame.info[ase_field])
                elif hasattr(frame, ase_field):
                    data_list.append(getattr(frame, ase_field))
            
            if data_list:
                traj_dict[h5_key] = np.array(data_list)
                print(f"Converted field '{ase_field}' to HDF5 key '{h5_key}' with shape {traj_dict[h5_key].shape}")
    
    if len(ase_traj.data) > 0 and hasattr(ase_traj.data[0], 'get_atomic_numbers'):
        traj_dict['atomic_numbers'] = ase_traj.data[0].get_atomic_numbers()
        print(f"Added atomic_numbers with shape {traj_dict['atomic_numbers'].shape}")
    
    return traj_dict

chosen_fields = {
    POSITIONS_KEY, PARTIAL_CHARGES_KEY, PBC_KEY, CELL_KEY, 
    VELOCITIES_KEY, DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY, UPDATE_PARTIAL_CHARGES_KEY
}

print("\nConverting to dictionary for HDF5...")
traj_dict = ase_trajectory_to_dict_simple(traj, chosen_fields)
print(f"Dictionary keys: {list(traj_dict.keys())}")

print("\nWriting HDF5 file...")
hdf5_traj = HDF5Trajectory(
    hdf5_dictionary=traj_dict,
    key_mapping={}
)

hdf5_traj.write_to_file(
    root=ROOT,
    filename_prefix=HDF5_PREFIX,
    chosen_fields=set(traj_dict.keys())
)

hdf5_filepath = os.path.join(ROOT, HDF5_PREFIX + '.h5')
print(f"HDF5 file written: {hdf5_filepath}")

def inspect_hdf5_file(filepath):
    print("\n" + "="*60)
    print("HDF5 FILE INSPECTION")
    print("="*60)
    
    with h5py.File(filepath, 'r') as f:
        print(f"File: {filepath}")
        print(f"Number of datasets: {len(f.keys())}")
        print("\nDatasets:")
        
        for key in sorted(f.keys()):
            dataset = f[key]
            print(f"\n  {key}:")
            print(f"    Shape: {dataset.shape}")
            print(f"    Data type: {dataset.dtype}")
            print(f"    Size: {dataset.size} elements")
            
            if np.issubdtype(dataset.dtype, np.number):
                data = dataset[:]
                print(f"    Min: {np.min(data):.6f}")
                print(f"    Max: {np.max(data):.6f}")
                print(f"    Mean: {np.mean(data):.6f}")
                print(f"    Std: {np.std(data):.6f}")
                
                if dataset.size <= 10:
                    print(f"    Data: {data}")
                else:
                    print(f"    First 5 elements: {data.flat[:5]}")
            
            else:
                sample = dataset[0] if dataset.shape[0] > 0 else dataset[()]
                print(f"    Sample: {sample}")

def test_hdf5_read(filepath):
    print("\n" + "="*60)
    print("TESTING HDF5 READ FUNCTIONALITY")
    print("="*60)
    
    try:
        hdf5_traj_loaded = HDF5Trajectory.read_from_file(
            root=ROOT,
            filename=HDF5_PREFIX + '.h5',
            indices=":",
            key_mapping={}
        )
        
        print(f"✓ Successfully loaded HDF5 trajectory")
        print(f"  Frames: {hdf5_traj_loaded.n_frames}")
        print(f"  Available fields: {sorted(hdf5_traj_loaded.available_fields)}")
        
        hdf5_traj_partial = HDF5Trajectory.read_from_file(
            root=ROOT,
            filename=HDF5_PREFIX + '.h5',
            indices=":5",
            key_mapping={}
        )
        print(f"✓ Partial load successful (first 5 frames)")
        print(f"  Partial frames: {hdf5_traj_partial.n_frames}")
        
        print(f"\nData access test:")
        for field in sorted(hdf5_traj_loaded.available_fields):
            if field in hdf5_traj_loaded.data:
                data = hdf5_traj_loaded.data[field]
                print(f"  {field}: shape {data.shape}, dtype {data.dtype}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading HDF5 file: {e}")
        import traceback
        traceback.print_exc()
        return False

inspect_hdf5_file(hdf5_filepath)
read_success = test_hdf5_read(hdf5_filepath)
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
if read_success:
    print("✓ HDF5 functionality is working correctly!")
    print(f"  File: {hdf5_filepath}")
    print(f"  You can manually inspect the file using:")
    print(f"    h5ls -r {hdf5_filepath}")
    print(f"    h5dump -n {hdf5_filepath}")
else:
    print("✗ HDF5 functionality has issues - check the errors above")

print("\nTest completed!")
