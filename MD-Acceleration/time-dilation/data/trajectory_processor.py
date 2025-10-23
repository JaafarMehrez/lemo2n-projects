"""
trajectory_processor.py

A flexible trajectory reader/processor that supports multiple input formats (via
optional backends ASE, MDAnalysis, mdtraj) and writes processed trajectories to
HDF5, NPZ, and extended XYZ formats (and optionally Zarr) suitable for dataset creation.

Features:
- Reads common trajectory formats: LAMMPS dump (lammpstrj), XYZ, DCD, XTC, TRR, PDB, GRO,
  and many others supported by ASE/MDAnalysis/mdtraj.
- Detects available backend automatically and falls back gracefully.
- Keeps both wrapped (as-read) and unwrapped positions.
- Robust unwrapping for variable box (NPT) using fractional-coordinate tracking.
- Computes displacements and, optionally, recomputes velocities from unwrapped
  positions using finite differences (central difference where possible).
- Supports optional per-atom partial charges and stores both original and
  post-processed ("updated") charges.
- Writes output to HDF5 (.h5) using a compact layout, to NPZ packages, and to
  extended XYZ format. Optionally supports Zarr if available.

Usage (example):
    proc = TrajectoryProcessor()
    data = proc.load('traj.lammpstrj', fmt='lammps')
    proc.process(recompute_velocities=True, dt=1.0)
    proc.save('processed.h5', format='hdf5')
    proc.save('traj.xyz', format='extxyz')

The implementation intentionally uses different variable names and algorithms
compared to other sample code; variable and class names are distinct.

Author: Jaafar Mehrez (jaafarmehrez@sjtu.edu.cn)
"""

from __future__ import annotations

import os
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Lazy imports for optional backends
_try_ase = False
_try_mdanalysis = False
_try_mdtraj = False
_try_zarr = False
try:
    import ase.io
    _try_ase = True
except Exception:
    pass
try:
    import MDAnalysis as mda
    _try_mdanalysis = True
except Exception:
    pass
try:
    import mdtraj as md
    _try_mdtraj = True
except Exception:
    pass
try:
    import zarr
    _try_zarr = True
except Exception:
    pass

# HDF5 writer
try:
    import h5py
except Exception as e:
    raise ImportError("h5py is required to write HDF5 output: install h5py") from e


class FrameBatch:
    """Container for trajectory data after loading and processing.

    Attributes
    ----------
    pos_wrapped : np.ndarray
        shape (n_frames, n_atoms, 3)
    pos_unwrapped : np.ndarray
        shape (n_frames, n_atoms, 3)
    vel : Optional[np.ndarray]
        as-read velocities, shape (n_frames, n_atoms, 3) or None
    vel_corrected : Optional[np.ndarray]
        recomputed velocities from unwrapped positions (if requested)
    atom_charges : Optional[np.ndarray]
        partial charges per frame, shape (n_frames, n_atoms) or None
    atom_charges_corrected : Optional[np.ndarray]
        post-processed charges (if any)
    box : Optional[np.ndarray]
        shape (n_frames, 3, 3) box matrices (row-wise vectors) if available
    times : Optional[np.ndarray]
        frame times in physical units if available
    atom_types : Optional[np.ndarray]
        atom type information, shape (n_atoms,) or None
    metadata : dict
        arbitrary metadata
    """

    def __init__(self):
        self.pos_wrapped = None
        self.pos_unwrapped = None
        self.vel = None
        self.vel_corrected = None
        self.atom_charges = None
        self.atom_charges_corrected = None
        self.box = None
        self.times = None
        self.atom_types = None
        self.meta: Dict[str, Any] = {}


class TrajectoryProcessor:
    """Main class to load, process, and save trajectory data.

    The object attempts to autodetect a capable reader from installed
    dependencies. For exotic formats or constrained environments, the user
    can explicitly select a backend via the `backend` argument.
    """

    def __init__(self, backend: Optional[str] = None):
        self.backend = backend
        self._preferred = backend
        self.batch = FrameBatch()

    # ----------------------------- Loading ---------------------------------
    def load(self, path: str, fmt: Optional[str] = None, backend: Optional[str] = None) -> FrameBatch:
        """Load a trajectory into memory using an available backend.

        Parameters
        ----------
        path : str
            Path to trajectory file (or to a directory with frames)
        fmt : Optional[str]
            Hint for format (e.g. 'lammps', 'xyz', 'dcd')
        backend : Optional[str]
            Force a particular backend: 'ase', 'mda', 'mdtraj'
        """
        chosen = backend or self._preferred
        loaders = []
        if chosen is None or chosen == 'ase':
            loaders.append(('ase', self._load_with_ase))
        if chosen is None or chosen == 'mda':
            loaders.append(('mda', self._load_with_mdanalysis))
        if chosen is None or chosen == 'mdtraj':
            loaders.append(('mdtraj', self._load_with_mdtraj))

        last_err = None
        for name, func in loaders:
            try:
                func(path, fmt)
                self.batch.meta['reader_used'] = name
                return self.batch
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"Failed to load trajectory with available backends: {last_err}")

    def _load_with_ase(self, path: str, fmt: Optional[str] = None):
        if not _try_ase:
            raise RuntimeError('ASE not installed')
        # ASE can read many single-file formats with ase.io.read(..., index=slice(None))
        try:
            atoms_list = ase.io.read(path, index=slice(None))
        except Exception as e:
            raise

        if isinstance(atoms_list, ase.Atoms):
            atoms_list = [atoms_list]

        nframes = len(atoms_list)
        natoms = len(atoms_list[0].get_positions())

        positions = np.empty((nframes, natoms, 3), dtype=float)
        boxes = np.empty((nframes, 3, 3), dtype=float)
        velocities = None
        charges = None
        atom_types = None
        times = []

        for i, at in enumerate(atoms_list):
            positions[i] = np.asarray(at.get_positions())
            cell = at.get_cell()
            if cell is None:
                boxes[i] = np.eye(3)
            else:
                boxes[i] = np.asarray(cell)
            # ASE sometimes stores velocities in arrays
            try:
                vel = at.get_velocities()
                if vel is not None:
                    if velocities is None:
                        velocities = np.zeros((nframes, natoms, 3), dtype=float)
                    velocities[i] = np.asarray(vel)
            except Exception:
                pass
            # ASE can store atomic charges in 'arrays' under Atoms.info or at.arrays
            try:
                if 'charges' in at.arrays:
                    if charges is None:
                        charges = np.zeros((nframes, natoms), dtype=float)
                    charges[i] = np.asarray(at.arrays['charges'])
            except Exception:
                pass
            # Get atom types from first frame
            if i == 0:
                try:
                    atom_types = np.asarray(at.get_chemical_symbols())
                except Exception:
                    try:
                        atom_types = np.asarray(at.get_atomic_numbers())
                    except Exception:
                        # Fallback: use sequential numbers
                        atom_types = np.arange(1, natoms + 1)

        self.batch.pos_wrapped = positions
        self.batch.box = boxes
        self.batch.vel = velocities
        self.batch.atom_charges = charges
        self.batch.atom_types = atom_types
        self.batch.meta['source_path'] = os.path.abspath(path)

    def _load_with_mdanalysis(self, path: str, fmt: Optional[str] = None):
        if not _try_mdanalysis:
            raise RuntimeError('MDAnalysis not installed')
        u = mda.Universe(path)
        nframes = len(u.trajectory)
        natoms = u.atoms.n_atoms
        positions = np.empty((nframes, natoms, 3), dtype=float)
        boxes = np.empty((nframes, 3, 3), dtype=float)
        velocities = None
        charges = None
        atom_types = None
        times = np.empty((nframes,), dtype=float)

        # Get atom types/names
        try:
            atom_types = np.array([atom.type for atom in u.atoms])
        except Exception:
            try:
                atom_types = np.array([atom.name for atom in u.atoms])
            except Exception:
                atom_types = np.arange(1, natoms + 1)

        for fi, ts in enumerate(u.trajectory):
            positions[fi] = u.atoms.positions.copy()
            # box in MDAnalysis is ts.dimensions (6 or 9). If 6, convert to matrix
            dims = ts.dimensions
            if len(dims) >= 9:
                boxes[fi] = np.reshape(dims[:9], (3, 3))
            elif len(dims) >= 6:
                # a, b, c, alpha, beta, gamma
                a, b, c, alpha, beta, gamma = dims[:6]
                # simplified orthorhombic assumption if angles are 90
                boxes[fi] = np.diag([a, b, c])
            else:
                boxes[fi] = np.eye(3)
            try:
                vel = u.atoms.velocities
                if vel is not None:
                    if velocities is None:
                        velocities = np.zeros((nframes, natoms, 3), dtype=float)
                    velocities[fi] = vel.copy()
            except Exception:
                pass
            times[fi] = ts.time
            # extra attributes like partial_charges can sometimes be in atomgroups; try to find
            try:
                if hasattr(u.atoms, 'charges'):
                    if charges is None:
                        charges = np.zeros((nframes, natoms), dtype=float)
                    charges[fi] = u.atoms.charges.copy()
            except Exception:
                pass

        self.batch.pos_wrapped = positions
        self.batch.box = boxes
        self.batch.vel = velocities
        self.batch.atom_charges = charges
        self.batch.atom_types = atom_types
        self.batch.times = times
        self.batch.meta['source_path'] = os.path.abspath(path)

    def _load_with_mdtraj(self, path: str, fmt: Optional[str] = None):
        if not _try_mdtraj:
            raise RuntimeError('mdtraj not installed')
        traj = md.load(path)
        positions = traj.xyz.copy()  # shape (n_frames, n_atoms, 3)
        nframes = positions.shape[0]
        natoms = positions.shape[1]
        boxes = np.zeros((nframes, 3, 3), dtype=float)
        
        # Get atom types/names
        try:
            atom_types = np.array([atom.element.symbol for atom in traj.topology.atoms])
        except Exception:
            try:
                atom_types = np.array([atom.name for atom in traj.topology.atoms])
            except Exception:
                atom_types = np.arange(1, natoms + 1)
                
        for i in range(nframes):
            # mdtraj stores unitcell_lengths and angles
            if traj.unitcell_lengths is not None:
                L = traj.unitcell_lengths[i]
                boxes[i] = np.diag(L)
            else:
                boxes[i] = np.eye(3)
        velocities = None
        charges = None
        # mdtraj does not store per-atom charges by default

        self.batch.pos_wrapped = positions
        self.batch.box = boxes
        self.batch.vel = velocities
        self.batch.atom_charges = charges
        self.batch.atom_types = atom_types
        self.batch.meta['source_path'] = os.path.abspath(path)

    # --------------------------- Processing --------------------------------
    def process(self,
                recompute_velocities: bool = False,
                dt: Optional[float] = None,
                remove_com_motion: bool = True,
                handle_charges: bool = True):
        """Process the loaded trajectory.

        Steps performed:
        - Unwrap positions using fractional-coordinate nearest-image method which
          works even for changing box sizes (NPT).
        - Optionally recompute velocities from unwrapped positions using central
          differences (requires dt or derivable from time data).
        - Optionally remove center-of-mass motion from velocities.
        - Preserve original charges if present and optionally create a "corrected"
          charge array (currently identity transform, place for custom routines).

        Parameters
        ----------
        recompute_velocities : bool
            If True, velocities will be recomputed from unwrapped positions.
        dt : Optional[float]
            Time-step in same units as frame times. If None, attempt to infer
            from batch.times. Required to recompute velocities.
        remove_com_motion : bool
            If True, subtract COM velocity from per-atom velocities.
        handle_charges : bool
            If True, copy atom_charges to atom_charges_corrected (placeholder
            for user-supplied charge processing).
        """
        if self.batch.pos_wrapped is None:
            raise RuntimeError('No trajectory loaded')

        X = np.asarray(self.batch.pos_wrapped, dtype=float)
        nframes, natoms, _ = X.shape

        boxmats = self.batch.box
        if boxmats is None:
            # If no box provided, assume infinite box (no PBC) -> unwrapped = wrapped
            self.batch.pos_unwrapped = X.copy()
        else:
            self.batch.pos_unwrapped = _unwrap_positions_fractional(X, boxmats)

        # compute displacements (simple forward difference r[t+1]-r[t]) last frame zeros
        disp = np.empty_like(self.batch.pos_unwrapped)
        disp[:-1] = self.batch.pos_unwrapped[1:] - self.batch.pos_unwrapped[:-1]
        disp[-1] = np.zeros((natoms, 3))
        self.batch.meta['displacements'] = disp

        # velocities
        if recompute_velocities:
            if dt is None:
                if self.batch.times is not None:
                    # infer average dt
                    dts = np.diff(self.batch.times)
                    if np.any(dts <= 0):
                        raise RuntimeError('Non-positive time differences found; specify dt')
                    dt = float(np.mean(dts))
                else:
                    raise RuntimeError('dt not provided and times missing; cannot recompute velocities')
            vnew = _compute_vel_from_positions(self.batch.pos_unwrapped, dt)
            if remove_com_motion:
                vnew = _remove_com_velocity(vnew)
            self.batch.vel_corrected = vnew
        else:
            # keep original velocities in vel_corrected if present
            if self.batch.vel is not None:
                v = np.asarray(self.batch.vel)
                if remove_com_motion:
                    v = _remove_com_velocity(v)
                self.batch.vel_corrected = v

        if handle_charges:
            if self.batch.atom_charges is not None:
                self.batch.atom_charges_corrected = np.asarray(self.batch.atom_charges).copy()
            else:
                self.batch.atom_charges_corrected = None

    # ---------------------------- Saving ----------------------------------
    def save(self, outpath: str, format: str = 'auto', **kwargs):
        """Save the processed batch in the specified format.
        
        Parameters
        ----------
        outpath : str
            Output file path
        format : str
            Output format: 'hdf5', 'npz', 'extxyz', 'zarr', or 'auto' (detect from extension)
        **kwargs
            Additional format-specific options
        """
        if format == 'auto':
            format = self._detect_format(outpath)
            
        format = format.lower()
        if format in ('hdf5', 'h5'):
            self.save_hdf5(outpath, **kwargs)
        elif format == 'npz':
            self.save_npz(outpath, **kwargs)
        elif format in ('extxyz', 'xyz'):
            self.save_extxyz(outpath, **kwargs)
        elif format == 'zarr':
            self.save_zarr(outpath, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _detect_format(self, outpath: str) -> str:
        """Detect format from file extension."""
        ext = os.path.splitext(outpath)[1].lower()
        if ext in ('.h5', '.hdf5'):
            return 'hdf5'
        elif ext == '.npz':
            return 'npz'
        elif ext in ('.xyz', '.extxyz'):
            return 'extxyz'
        elif ext in ('.zarr',):
            return 'zarr'
        else:
            # Default to HDF5 for unknown extensions
            return 'hdf5'

    def save_hdf5(self, outpath: str, compress: bool = True):
        """Save the processed batch to an HDF5 file.

        Layout:
        /positions_wrapped    (n_frames, n_atoms, 3)
        /positions_unwrapped  (n_frames, n_atoms, 3)
        /vel_corrected        (n_frames, n_atoms, 3)  # if present
        /charges              (n_frames, n_atoms)     # if present
        /box                  (n_frames, 3, 3)
        /times                (n_frames,)
        /metadata             (attrs)
        """
        if self.batch.pos_unwrapped is None:
            raise RuntimeError('Process the trajectory before saving')

        with h5py.File(outpath, 'w') as f:
            kwargs = {'compression': 'gzip'} if compress else {}
            f.create_dataset('positions_wrapped', data=self.batch.pos_wrapped, **kwargs)
            f.create_dataset('positions_unwrapped', data=self.batch.pos_unwrapped, **kwargs)
            if self.batch.vel_corrected is not None:
                f.create_dataset('velocities', data=self.batch.vel_corrected, **kwargs)
            if self.batch.atom_charges is not None:
                f.create_dataset('charges', data=self.batch.atom_charges, **kwargs)
            if self.batch.atom_charges_corrected is not None:
                f.create_dataset('charges_corrected', data=self.batch.atom_charges_corrected, **kwargs)
            if self.batch.box is not None:
                f.create_dataset('box', data=self.batch.box, **kwargs)
            if self.batch.times is not None:
                f.create_dataset('times', data=self.batch.times, **kwargs)
            if self.batch.atom_types is not None:
                # Store atom types as variable-length strings
                atom_types_str = [str(t) for t in self.batch.atom_types]
                f.create_dataset('atom_types', data=atom_types_str, dtype=h5py.special_dtype(vlen=str))
            # write simple metadata as attributes
            for k, v in self.batch.meta.items():
                try:
                    f.attrs[k] = str(v)
                except Exception:
                    pass

    def save_npz(self, outpath: str):
        """Save core arrays to a compressed numpy .npz file.

        Keys: pos_wrapped, pos_unwrapped, velocities, charges, charges_corrected, box, times
        """
        arrs = {'pos_wrapped': self.batch.pos_wrapped, 'pos_unwrapped': self.batch.pos_unwrapped}
        if self.batch.vel_corrected is not None:
            arrs['velocities'] = self.batch.vel_corrected
        if self.batch.atom_charges is not None:
            arrs['charges'] = self.batch.atom_charges
        if self.batch.atom_charges_corrected is not None:
            arrs['charges_corrected'] = self.batch.atom_charges_corrected
        if self.batch.box is not None:
            arrs['box'] = self.batch.box
        if self.batch.times is not None:
            arrs['times'] = self.batch.times
        if self.batch.atom_types is not None:
            arrs['atom_types'] = self.batch.atom_types
        np.savez_compressed(outpath, **arrs)

    def save_extxyz(self, outpath: str, use_unwrapped: bool = True, include_velocities: bool = True, 
                   include_charges: bool = True, include_box: bool = True):
        """Save the processed trajectory in extended XYZ format.
        
        Extended XYZ format can store atomic positions, velocities, charges, and cell information.
        
        Parameters
        ----------
        outpath : str
            Output .xyz file path
        use_unwrapped : bool
            If True, use unwrapped positions; otherwise use wrapped positions
        include_velocities : bool
            If True, include velocity information
        include_charges : bool
            If True, include charge information  
        include_box : bool
            If True, include periodic box information
        """
        if self.batch.pos_unwrapped is None:
            raise RuntimeError('Process the trajectory before saving')
            
        if self.batch.atom_types is None:
            raise RuntimeError('Atom type information required for XYZ format')
            
        n_frames, n_atoms, _ = self.batch.pos_unwrapped.shape
        
        with open(outpath, 'w') as f:
            for frame_idx in range(n_frames):
                # Write number of atoms
                f.write(f"{n_atoms}\n")
                
                # Write properties line
                properties = []
                
                # Add box information if available
                if include_box and self.batch.box is not None:
                    box = self.batch.box[frame_idx]
                    # Convert 3x3 matrix to 9 numbers in row-major order
                    box_flat = ' '.join(str(x) for x in box.ravel())
                    properties.append(f'Lattice="{box_flat}"')
                
                # Add time if available
                if self.batch.times is not None:
                    properties.append(f'Time={self.batch.times[frame_idx]}')
                
                # Add Properties field describing the columns
                prop_columns = ["species:S:1", "pos:R:3"]
                if include_velocities and self.batch.vel_corrected is not None:
                    prop_columns.append("vel:R:3")
                if include_charges and self.batch.atom_charges_corrected is not None:
                    prop_columns.append("charge:R:1")
                
                properties.append(f'Properties={"".join(prop_columns)}')
                
                f.write(' '.join(properties) + '\n')
                
                # Write atom data
                positions = self.batch.pos_unwrapped[frame_idx] if use_unwrapped else self.batch.pos_wrapped[frame_idx]
                
                for atom_idx in range(n_atoms):
                    line_parts = [str(self.batch.atom_types[atom_idx])]
                    
                    # Add position
                    line_parts.extend(f"{x:.8f}" for x in positions[atom_idx])
                    
                    # Add velocity if available
                    if include_velocities and self.batch.vel_corrected is not None:
                        line_parts.extend(f"{v:.8f}" for v in self.batch.vel_corrected[frame_idx, atom_idx])
                    
                    # Add charge if available
                    if include_charges and self.batch.atom_charges_corrected is not None:
                        line_parts.append(f"{self.batch.atom_charges_corrected[frame_idx, atom_idx]:.8f}")
                    
                    f.write(' '.join(line_parts) + '\n')

    def save_zarr(self, outpath: str):
        if not _try_zarr:
            raise RuntimeError('zarr not installed')
        store = zarr.open(outpath, mode='w')
        store.create_dataset('pos_wrapped', data=self.batch.pos_wrapped)
        store.create_dataset('pos_unwrapped', data=self.batch.pos_unwrapped)
        if self.batch.vel_corrected is not None:
            store.create_dataset('velocities', data=self.batch.vel_corrected)
        if self.batch.atom_charges is not None:
            store.create_dataset('charges', data=self.batch.atom_charges)
        if self.batch.box is not None:
            store.create_dataset('box', data=self.batch.box)


# --------------------- Helper functions ---------------------------------

def _unwrap_positions_fractional(positions: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Unwrap a trajectory of positions using fractional coordinates.

    This function works when the cell changes between frames by converting
    Cartesian positions to fractional coordinates via the inverse of the box
    matrix for each frame and tracking integer image changes between frames.

    Parameters
    ----------
    positions : (n_frames, n_atoms, 3) array
    boxes : (n_frames, 3, 3) array of box matrices (row vectors)

    Returns
    -------
    unwrapped : (n_frames, n_atoms, 3)
    """
    nframes, natoms, _ = positions.shape
    frac = np.empty_like(positions)
    inv_boxes = np.empty_like(boxes)
    for t in range(nframes):
        B = boxes[t]
        try:
            inv_boxes[t] = np.linalg.inv(B)
        except Exception:
            # fallback to diagonal assumption
            inv_boxes[t] = np.diag(1.0 / np.diag(B))
        frac[t] = positions[t] @ inv_boxes[t].T

    # cumulative integer image shifts for each atom
    cum_shifts = np.zeros((natoms, 3), dtype=int)
    unwrapped = np.empty_like(positions)
    unwrapped[0] = positions[0].copy()

    for t in range(1, nframes):
        # delta between fractional coords
        df = frac[t] - frac[t - 1]
        # bring df to nearest image by rounding
        shifts = np.round(df).astype(int)
        cum_shifts += shifts  # broadcast per-atom
        # unwrapped fractional coords
        frac_unwrapped = frac[t] + cum_shifts
        # convert back to cartesian using current box (keeps coordinates consistent
        # with the current frame's box vectors)
        unwrapped[t] = frac_unwrapped @ boxes[t].T

    return unwrapped


def _compute_vel_from_positions(positions: np.ndarray, dt: float) -> np.ndarray:
    """Compute velocities from unwrapped positions using central differences.

    Uses forward/backward differences at the edges.
    """
    nframes = positions.shape[0]
    vel = np.zeros_like(positions)
    if nframes == 1:
        return vel
    # central differences interior
    vel[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt)
    # endpoints: forward/backward
    vel[0] = (positions[1] - positions[0]) / dt
    vel[-1] = (positions[-1] - positions[-2]) / dt
    return vel


def _remove_com_velocity(velocities: np.ndarray) -> np.ndarray:
    """Subtract center-of-mass velocity per frame.

    Assumes uniform masses unless mass info is provided elsewhere.
    """
    # uniform mass simplification
    com = np.mean(velocities, axis=1, keepdims=True)
    return velocities - com


# --------------------------- CLI ---------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Trajectory reader & processor')
    parser.add_argument('input', help='Input trajectory file')
    parser.add_argument('--backend', choices=['ase', 'mda', 'mdtraj'], help='Force backend')
    parser.add_argument('--fmt', help='Format hint (e.g. lammps, xyz, dcd)')
    parser.add_argument('--dt', type=float, default=None, help='Time-step for velocity recomputation')
    parser.add_argument('--recompute-vel', action='store_true', help='Recompute velocities from positions')
    parser.add_argument('--out-format', choices=['hdf5', 'npz', 'extxyz', 'zarr'], 
                       default='hdf5', help='Output format')
    parser.add_argument('--out', default='processed', help='Output path (extension may be added)')
    args = parser.parse_args()

    # Add appropriate extension if not present
    outpath = args.out
    if args.out_format == 'hdf5' and not outpath.endswith(('.h5', '.hdf5')):
        outpath += '.h5'
    elif args.out_format == 'npz' and not outpath.endswith('.npz'):
        outpath += '.npz'
    elif args.out_format == 'extxyz' and not outpath.endswith(('.xyz', '.extxyz')):
        outpath += '.xyz'
    elif args.out_format == 'zarr' and not outpath.endswith('.zarr'):
        outpath += '.zarr'

    tp = TrajectoryProcessor(backend=args.backend)
    tp.load(args.input, fmt=args.fmt)
    tp.process(recompute_velocities=args.recompute_vel, dt=args.dt)
    tp.save(outpath, format=args.out_format)

    print(f"Saved processed trajectory to {outpath}")
