'''
Generalized trajectory -> time-lagged dataset converter.

Writes an HDF5 file with one group per sample:
 /samples/000000
    attrs: traj_num, frame_idx, time_lag, system_id, timestep (optional)
    arrays: positions (N,3), momenta (N,3), masses (N,), numbers (N,),
            delta_q (N,3), delta_p (N,3),
            neighbors_flat (M,), neighbor_offsets (N+1,)  OR neighbors_padded (N, max_neigh), neighbor_count (N,)
Author: Jaafar Mehrez (jaafarmehrez@sjtu.edu.cn)
'''

import argparse
import ase.io
import numpy as np
from ase.neighborlist import neighbor_list
from ase.data import atomic_masses
import h5py
import copy
import os
import math

def build_neighbors_flat(atoms, cutoff):
    i, j, d = neighbor_list('ijd', atoms, cutoff)
    # Build flattened list and offsets (CSR-like)
    N = len(atoms)
    neighs = [[] for _ in range(N)]
    for a, b in zip(i, j):
        neighs[a].append(int(b))
    offsets = np.zeros(N + 1, dtype=np.int64)
    flat = []
    for idx, lst in enumerate(neighs):
        offsets[idx+1] = offsets[idx] + len(lst)
        flat.extend(lst)
    return np.array(flat, dtype=np.int32), offsets  # flat neighbors, offsets

def build_neighbors_padded(atoms, cutoff, pad_to=None):
    i, j, d = neighbor_list('ijd', atoms, cutoff)
    N = len(atoms)
    neighs = [[] for _ in range(N)]
    for a, b in zip(i, j):
        neighs[a].append(int(b))
    max_neigh = max((len(x) for x in neighs), default=0)
    if pad_to is not None:
        max_neigh = max(max_neigh, pad_to)
    if max_neigh == 0:
        return np.empty((N, 0), dtype=np.int32), np.zeros(N, dtype=np.int32)
    arr = -np.ones((N, max_neigh), dtype=np.int32)
    counts = np.zeros(N, dtype=np.int32)
    for idx, lst in enumerate(neighs):
        counts[idx] = len(lst)
        if len(lst):
            arr[idx, :len(lst)] = np.array(lst, dtype=np.int32)
    return arr, counts

def get_momenta_from_frame(frame):
    # frames may have 'momenta' array or 'velocities' or use get_momenta/get_velocities
    if "momenta" in frame.arrays:
        return frame.arrays["momenta"].astype(np.float64)
    # velocities likely available
    if "velocities" in frame.arrays:
        v = frame.arrays["velocities"].astype(np.float64)
        masses = frame.get_masses()[:, None]
        return (v * masses).astype(np.float64)
    # some dumps store velocities under 'momenta' name differently; try get_momenta()
    try:
        p = frame.get_momenta()
        if p is not None:
            return np.asarray(p, dtype=np.float64)
    except Exception:
        pass
    # fallback: zeros
    raise RuntimeError("No momenta or velocities found in frame; cannot compute momenta.")

def make_sample(frame_now, frame_ahead, time_lag, system_id,
                cutoff=5.0, max_speed_factor=10.0, timestep_fs=None,
                neighbor_mode="flat", pad_to=None, force_species=None,
                rescale_momenta=None):
    # deep copy to avoid altering trajectory in memory
    f0 = copy.deepcopy(frame_now)
    f1 = copy.deepcopy(frame_ahead)

    # Optionally force species to a single element (not default)
    if force_species is not None:
        Z = force_species
        f0.numbers[:] = Z
        f1.numbers[:] = Z

    numbers = np.array(f0.numbers, dtype=np.int32)  # per-atom Z (assumes f0 and f1 same length)
    N = len(numbers)
    masses = np.array([atomic_masses[int(Z)] for Z in numbers], dtype=np.float64)

    p0 = get_momenta_from_frame(f0)
    p1 = get_momenta_from_frame(f1)

    # optional rescale: callable or scalar factor applied to momenta arrays
    if rescale_momenta is not None:
        if callable(rescale_momenta):
            p0 = rescale_momenta(p0, masses)
            p1 = rescale_momenta(p1, masses)
        else:
            p0 = p0 * float(rescale_momenta)
            p1 = p1 * float(rescale_momenta)

    pos0 = np.array(f0.get_positions(), dtype=np.float64)
    pos1 = np.array(f1.get_positions(), dtype=np.float64)

    # compute displacement and check speed threshold
    delta_q = pos1 - pos0  # angstroms
    # default: treat max speed threshold as (max_speed_factor * 0.25 * time_lag) Angstroms per time-lag,
    # If you have timestep in fs, you can convert to A/fs.
    threshold = max_speed_factor * 0.25 * time_lag
    if np.any(np.abs(delta_q) > threshold):
        return None  # skip sample

    pos_avg = 0.5 * (pos0 + pos1)
    p_avg = 0.5 * (p0 + p1)
    delta_p = p1 - p0

    # Build ASE Atoms for midpoint to feed neighbor_list
    atoms_avg = copy.deepcopy(f0)
    atoms_avg.set_positions(pos_avg)
    atoms_avg.numbers = numbers  # ensure numbers correct

    if neighbor_mode == "flat":
        neighbors_flat, neighbor_offsets = build_neighbors_flat(atoms_avg, cutoff)
        neighbors_padded = None
        neighbor_count = None
    else:
        neighbors_padded, neighbor_count = build_neighbors_padded(atoms_avg, cutoff, pad_to=pad_to)
        neighbors_flat = None
        neighbor_offsets = None

    sample = {
        "system_id": int(system_id),
        "traj_frame_idx": int(getattr(frame_now, "info", {}).get("frame_index", -1)),  # optional
        "time_lag": int(time_lag),
        "numbers": numbers,          # (N,)
        "masses": masses,            # (N,)
        "positions": pos_avg,        # (N,3)
        "momenta": p_avg,            # (N,3)
        "delta_q": delta_q,          # (N,3)
        "delta_p": delta_p,          # (N,3)
    }
    if neighbors_flat is not None:
        sample["neighbors_flat"] = neighbors_flat
        sample["neighbor_offsets"] = neighbor_offsets
    else:
        sample["neighbors_padded"] = neighbors_padded
        sample["neighbor_count"] = neighbor_count

    return sample

def write_sample_h5(h5_group, sample):
    """
    Write entries of `sample` to an HDF5 group.
    - Scalars (int/float/bool) and short strings -> stored as attributes
    - numpy arrays / lists -> stored as datasets with gzip compression
    - string arrays -> stored as dataset with variable-length UTF-8 dtype
    """
    # store obvious attrs first (keeps parity with original behavior)
    for k in ("system_id", "time_lag"):
        if k in sample:
            try:
                h5_group.attrs[k] = sample[k]
            except Exception:
                # fallback: convert to string
                h5_group.attrs[k] = str(sample[k])

    for k, v in sample.items():
        if k in ("system_id", "time_lag"):
            continue
        if v is None:
            continue

        # Convert to numpy where reasonable
        if isinstance(v, (int, float, bool)):
            h5_group.attrs[k] = v
            continue
        if isinstance(v, str):
            h5_group.attrs[k] = v
            continue
        if isinstance(v, bytes):
            try:
                h5_group.attrs[k] = v.decode("utf-8")
            except Exception:
                h5_group.attrs[k] = v

        arr = np.asarray(v)
        
        if arr.ndim == 0:
            try:
                h5_group.attrs[k] = arr.item()
            except Exception:
                h5_group.attrs[k] = str(arr)
            continue

        if arr.dtype.kind in ("U", "S", "O") and arr.dtype.kind != "f" and arr.dtype.kind != "i":
            dt = h5py.string_dtype(encoding="utf-8")
            if k in h5_group:
                del h5_group[k]
            h5_group.create_dataset(k, data=arr.astype('U'), dtype=dt, compression="gzip")
            continue

        if k in h5_group:
            del h5_group[k]
        h5_group.create_dataset(k, data=arr, compression="gzip")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "val", "test"])
    parser.add_argument("--traj-range", type=str, default=None, help="e.g. 1-8 or comma list")
    parser.add_argument("--traj-files", nargs="*", help="provide explicit filenames instead of numbering")
    parser.add_argument("--time-lags", type=str, default="128", help="comma separated time lags, e.g. 64,128")
    parser.add_argument("--out", default="dataset.h5")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=400, help="correlation_time stride")
    parser.add_argument("--max-speed-factor", type=float, default=10.0)
    parser.add_argument("--neighbor-mode", choices=["flat", "padded"], default="flat")
    parser.add_argument("--pad-to", type=int, default=None)
    parser.add_argument("--force-species", type=int, default=None, help="force atomic number (Z) for all atoms")
    parser.add_argument("--rescale-momenta", type=float, default=None, help="global scalar to multiply momenta")
    parser.add_argument("--time-reversal-augment", action="store_true")
    args = parser.parse_args(argv)

    if args.traj_files:
        traj_list = args.traj_files
    elif args.traj_range:
        parts = []
        for token in args.traj_range.split(","):
            if "-" in token:
                a, b = token.split("-")
                parts.extend(range(int(a), int(b)+1))
            else:
                parts.append(int(token))
        traj_list = [f"dump_{n}.lammpstrj" for n in parts]
    else:
        
        if args.mode == "train":
            traj_list = [f"dump_{n}.lammpstrj" for n in range(1,9)]
        elif args.mode == "val":
            traj_list = [f"dump_{n}.lammpstrj" for n in range(9,10)]
        else:
            traj_list = [f"dump_{n}.lammpstrj" for n in range(10,11)]

    time_lags = [int(x) for x in args.time_lags.split(",")]
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    sample_counter = 0
    with h5py.File(out_path, "w") as hf:
        samp_grp = hf.create_group("samples")
        for tfile in traj_list:
            print("Reading", tfile)
            traj = ase.io.read(tfile, index=":")
            L = len(traj)
            for i in range(0, L - max(time_lags), args.stride):
                for time_lag in time_lags:
                    f0 = traj[i]
                    f1 = traj[i + time_lag]
                    sample = make_sample(f0, f1, time_lag, sample_counter,
                                         cutoff=args.cutoff,
                                         max_speed_factor=args.max_speed_factor,
                                         neighbor_mode=args.neighbor_mode,
                                         pad_to=args.pad_to,
                                         force_species=args.force_species,
                                         rescale_momenta=args.rescale_momenta)
                    if sample is None:
                        # skipped by filter
                        continue
                    g = samp_grp.create_group(f"{sample_counter:08d}")
                    write_sample_h5(g, sample)
                    g.attrs["source_file"] = str(tfile)
                    g.attrs["frame_idx"] = int(i)
                    g.attrs["time_lag"] = int(time_lag)
                    sample_counter += 1

                    if args.time_reversal_augment:
                        sample_tr = dict(sample)
                        sample_tr["momenta"] = -sample_tr["momenta"]
                        sample_tr["delta_p"] = -sample_tr["delta_p"]
                        g2 = samp_grp.create_group(f"{sample_counter:08d}")
                        write_sample_h5(g2, sample_tr)
                        g2.attrs["source_file"] = str(tfile)
                        g2.attrs["frame_idx"] = int(i)
                        g2.attrs["time_lag"] = int(time_lag)
                        g2.attrs["augment"] = "time_reversal"
                        sample_counter += 1

    print("Done. Written samples:", sample_counter)

if __name__ == "__main__":
    main()

