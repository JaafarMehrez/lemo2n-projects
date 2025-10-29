"""
processor of MD trajectories, this code is influenced by 
https://github.com/IBM/trajcast/blob/main/trajcast/data/trajectory.py

the modifications include:
1. Adding partial_charge and update_partial_charge keys to the computed fields
2. Fix the problem of mapping the partial charges instead of the atomic number in the resulting trajectory
3. Replacing npz file format with hdf5 for large trajectory files

Modified by: Jaafar Mehrez (jaafarmehrez@sjtu.edu.cn)
"""

import os
from typing import Dict, List, Optional, Set, Union

import ase
import ase.io
import numpy as np

import h5py

from _keys import (
    ASE_ARRAY_FIELDS,
    ASE_INFO_FIELDS,
    CELL_KEY,
    DISPLACEMENTS_KEY,
    FORCES_KEY,
    INPUT_KEY_MAPPING,
    PBC_KEY,
    POSITIONS_KEY,
    TIMESTEP_KEY,
    TOTAL_ENERGY_KEY,
    UPDATE_KEY,
    VELOCITIES_KEY,
    PARTIAL_CHARGES_KEY,
    UPDATE_PARTIAL_CHARGES_KEY,
    ATOMIC_NUMBERS_KEY,
)
from atomic_computes import align_vectors_with_periodicity
from misc import (
    convert_ase_atoms_to_dictionary,
    invert_dictionary,
    string2index,
    truncate_dictionary,
)

from _lammps import lammps_dump_to_ase_atoms

_TRAJECTORY_FIELDS: Set[str] = {
    POSITIONS_KEY,
    PBC_KEY,
    CELL_KEY,
    FORCES_KEY,
    VELOCITIES_KEY,
    TOTAL_ENERGY_KEY,
    DISPLACEMENTS_KEY,
    TIMESTEP_KEY,
    PARTIAL_CHARGES_KEY,
    UPDATE_PARTIAL_CHARGES_KEY,
    ATOMIC_NUMBERS_KEY,
}


class Trajectory:

    def __init__(
        self,
        data: Union[List[ase.Atoms], Dict] = None,
        available_fields: Set[str] = _TRAJECTORY_FIELDS,
    ):

        self.available_fields = available_fields
        self.data = data

    def _validate_chosen_fields(self, chosen_fields: Set[str]) -> bool:
        if not chosen_fields.issubset(
            self.available_fields
        ) or not chosen_fields.issuperset(
            set((POSITIONS_KEY, ATOMIC_NUMBERS_KEY, PARTIAL_CHARGES_KEY, PBC_KEY, CELL_KEY))
        ):
            return False
        else:
            return True

    def compute_additional_fields(
        self,
        add_fields: Set[str] = {DISPLACEMENTS_KEY},
        time_step: int = 1,
        time_step_in_fs: float = None,
        truncate: bool = True,
    ):
        time_between_frames = (
            self.time_between_frames if hasattr(self, "time_between_frames") else 1.0
        )

        if time_step_in_fs:
            time_step = int(time_step_in_fs / time_between_frames)

        if DISPLACEMENTS_KEY in add_fields:
            self.data = compute_atomic_displacement_vectors(
                trajectory_data=self.data,
                time_step=time_step,
                time_between_frames=time_between_frames,
                key_mapping=self.mapping_available_fields,
            )
            self.available_fields.update({DISPLACEMENTS_KEY, TIMESTEP_KEY})

        update_fields = [field for field in add_fields if UPDATE_KEY in field]
        if update_fields:
            raw_fields = [field.split(f"{UPDATE_KEY}_")[-1] for field in update_fields]
            if not set(raw_fields).issubset(self.available_fields):
                raise KeyError(
                    "Some of the update fields cannot be used as we do not have the original field either."
                )
            self.data = get_desired_field_values_of_next_frame(
                fields=raw_fields,
                trajectory_data=self.data,
                time_step=time_step,
                key_mapping=self.mapping_available_fields,
            )

            self.available_fields.update(set(update_fields))

        if truncate:
            self._truncate_trajectory(time_step=time_step)


class ASETrajectory(Trajectory):
    def __init__(
        self,
        ase_atoms_list: List[ase.Atoms],
        key_mapping: Optional[Dict[str, str]] = invert_dictionary(INPUT_KEY_MAPPING),
        time_between_frames: Optional[float] = 1.0,
        apply_wrapping: Optional[bool] = False,
        apply_unwrapping: Optional[bool] = False,
    ):
        super().__init__()

        self.data = ase_atoms_list
        self.n_frames = len(ase_atoms_list)
        dictionary = convert_ase_atoms_to_dictionary(self.data[0], rename=False)

        self.mapping_available_fields = {
            key_mapping.get(key, key): key for key in dictionary.keys()
        }
        self.available_fields = set(self.mapping_available_fields.keys())
        self.time_between_frames = time_between_frames

        if CELL_KEY in self.available_fields:
            self._guess_wrapping()

        if apply_unwrapping:
            self.unwrap()
        if apply_wrapping:
            self.wrap()

    def _guess_wrapping(self):
        scaled_pos = self.data[-1].get_scaled_positions(wrap=False)
        if np.min(scaled_pos) < 0 or np.max(scaled_pos) > 1:
            self._is_wrapped = False
        else:
            self._is_wrapped = True

    @property
    def is_wrapped(self):
        return self._is_wrapped

    @is_wrapped.setter
    def is_wrapped(self, value: bool):
        
        if not value:
            self.unwrap()
        else:
            self.wrap()

    def _truncate_trajectory(self, time_step):
        self.data = self.data[:-time_step]

    @classmethod
    def read_from_file(
        cls,
        root: str,
        filename: str,
        key_mapping: Optional[Dict[str, str]] = invert_dictionary(INPUT_KEY_MAPPING),
        frame_interval: Optional[float] = None,
        wrapper: Optional[str] = None,
        wrapper_kwargs: Optional[Dict] = None,
        apply_wrapping: Optional[bool] = False,
        apply_unwrapping: Optional[bool] = False,
        **ase_kwargs,
    ):
        
        path_to_file = os.path.join(root, filename)
        if not os.path.exists(path_to_file):
            raise FileNotFoundError(f"Path:{path_to_file} does not exist.")

        if not wrapper:
            ase_atoms_list = ase.io.read(path_to_file, **ase_kwargs)

            if hasattr(ase_atoms_list[0], "calc") and ase_atoms_list[0].calc:
                for frame in ase_atoms_list:
                    frame.arrays[FORCES_KEY] = frame.get_forces()

        else:
            ase_atoms_list = {"lammps": lammps_dump_to_ase_atoms}[wrapper](
                path_to_file=path_to_file, **wrapper_kwargs, **ase_kwargs
            )
            
        if not isinstance(ase_atoms_list, List):
            raise TypeError(
                f"Expected a list but received {type(ase_atoms_list).__name__}"
            )

        step_read = string2index(ase_kwargs.get("index")).step
        step_read = 1 if not step_read else step_read
        frame_interval = (
            frame_interval
            if frame_interval
            else ase_atoms_list[0].info.get(TIMESTEP_KEY, 1.0)
        )
        time_between_frames = step_read * frame_interval

        return cls(
            ase_atoms_list=ase_atoms_list,
            key_mapping=key_mapping,
            time_between_frames=time_between_frames,
            apply_wrapping=apply_wrapping,
            apply_unwrapping=apply_unwrapping,
        )

    def write_to_file(
        self,
        root: str = "./",
        filename_prefix: Optional[str] = "trajectory",
        chosen_fields: Optional[Set[str]] = set(),
        **ase_kwargs,
    ):
        
        if not os.path.exists(root):
            raise FileNotFoundError("Directory does not exist, please create it!")

        if not chosen_fields:
            chosen_fields = self.available_fields

        if not self._validate_chosen_fields(chosen_fields):
            raise KeyError(
                "Some elements of chosen fields are not available in the trajectory or minimum requirements (atomic numbers and positions) not satisfied."
            )

        modified_ase_atoms_list = self._modify_ase_atoms_list_based_on_chosen_fields(
            chosen_fields=chosen_fields
        )

        
        file_format = (
            ase_kwargs.get("format") if "format" in ase_kwargs.keys() else "extxyz"
        )
        path_to_file = os.path.join(root, f"{filename_prefix}.{file_format}")

        ase.io.write(path_to_file, modified_ase_atoms_list, **ase_kwargs)

    def _modify_ase_atoms_list_based_on_chosen_fields(
        self, chosen_fields: Set[str]
    ) -> List[ase.Atoms]:
        
        local_ase_atoms_list = self.data.copy()
        for frame in local_ase_atoms_list:
            
            frame.set_calculator()
            
            [
                frame.info.pop(self.mapping_available_fields[key])
                for key in ASE_INFO_FIELDS
                if {v: k for k, v in self.mapping_available_fields.items()}.get(key)
                in frame.info
                and key not in chosen_fields
            ]
            
            [
                frame.arrays.pop(self.mapping_available_fields[key])
                for key in ASE_ARRAY_FIELDS
                if {v: k for k, v in self.mapping_available_fields.items()}.get(key)
                in frame.arrays.keys()
                and key not in chosen_fields
            ]

        return local_ase_atoms_list

    def unwrap(self):
        if not self.data[0].__getattribute__(CELL_KEY):
            raise KeyError(
                "Cell not found, please define otherwise how do you expect to unwrap?"
            )

        coordinates = np.asarray([frame.positions for frame in self.data])

        if not np.all(self.data[0].cell == self.data[1].cell):
            raise NotImplementedError("Only NVT Ensemble so far.")

        displacement_vectors = np.diff(
            coordinates, prepend=coordinates[0][np.newaxis, :, :], axis=0
        )
        displacement_vectors_unwrapped = align_vectors_with_periodicity(
            displacement_vectors, self.data[0].cell
        )
        coordinates = coordinates[0] + np.cumsum(displacement_vectors_unwrapped, axis=0)

        for frame_index, frame in enumerate(self.data):
            frame.set_positions(coordinates[frame_index], apply_constraint=False)

        self._is_wrapped = False

    def wrap(self):
        if not self.data[0].__getattribute__(CELL_KEY):
            raise KeyError(
                "Cell not found, please define otherwise how do you expect to unwrap?"
            )

        for frame_index, frame in enumerate(self.data):
            frame.wrap()

        self._is_wrapped = True


class HDF5Trajectory(Trajectory):
    def __init__(
        self,
        hdf5_dictionary: Dict,
        key_mapping: Dict[str, str] = invert_dictionary(INPUT_KEY_MAPPING),
    ):
        super().__init__()

        self.data = hdf5_dictionary
        
        self.mapping_available_fields = {
            key_mapping.get(key, key): key for key in self.data.keys()
        }
        self.n_frames = self.data[self.mapping_available_fields[POSITIONS_KEY]].shape[0]
        self.available_fields = set(self.mapping_available_fields.keys())

    def _truncate_trajectory(self, time_step):
        self.data = truncate_dictionary(
            dictionary=self.data,
            n_values=self.n_frames - time_step,
        )

    @classmethod
    def read_from_file(
        cls,
        root: str,
        filename: str,
        indices: Optional[Union[str, List[int]]] = ":",
        key_mapping: Dict[str, str] = invert_dictionary(INPUT_KEY_MAPPING),
    ):
        import h5py
        
        path_to_file = os.path.join(root, filename)
        
        if not os.path.exists(path_to_file):
            raise FileNotFoundError(f"HDF5 file not found: {path_to_file}")
        
        hdf5_dictionary = {}
        with h5py.File(path_to_file, 'r') as h5file:
            
            if isinstance(indices, str):
                new_indices = string2index(indices)
            else:
                new_indices = indices
                
            for key in h5file.keys():
                dataset = h5file[key]
                if isinstance(dataset, h5py.Dataset):
                    hdf5_dictionary[key] = dataset[new_indices]

        return cls(hdf5_dictionary=hdf5_dictionary, key_mapping=key_mapping)

    def write_to_file(
        self,
        root: str = "./",
        filename_prefix: Optional[str] = "trajectory",
        chosen_fields: Optional[Set[str]] = None,
        **h5py_kwargs,
    ):
        import h5py
        
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

        if chosen_fields is None:
            chosen_fields = self.available_fields

        if not self._validate_chosen_fields(chosen_fields):
            raise KeyError(
                "Some elements of chosen fields are not available in the trajectory or minimum requirements (atomic numbers and positions) not satisfied."
            )

        path_to_file = os.path.join(root, f"{filename_prefix}.h5")

        with h5py.File(path_to_file, 'w') as h5file:
            for field in chosen_fields:
                if field in self.data:
                    data_key = self.mapping_available_fields[field]
                    h5file.create_dataset(data_key, data=self.data[data_key], **h5py_kwargs)

def compute_atomic_displacement_vectors(
    trajectory_data: Union[List[ase.Atoms], Dict],
    time_step: Optional[int] = 1,
    time_between_frames: Optional[float] = 1.0,
    key_mapping: Optional[Dict[str, str]] = {},
):

    if (isinstance(trajectory_data, list) and 
        isinstance(trajectory_data[0], ase.Atoms)):
        trajectory_type = "ase"
    elif isinstance(trajectory_data, dict):
        if (POSITIONS_KEY in trajectory_data or 
            key_mapping.get(POSITIONS_KEY) in trajectory_data):
            trajectory_type = "hdf5"
        
    else:
        trajectory_type = "unknown"
        
    n_frames = (
        len(trajectory_data)
        if trajectory_type == "ase"
        else len(trajectory_data.get(
            key_mapping.get(POSITIONS_KEY, POSITIONS_KEY), 
            trajectory_data.get(POSITIONS_KEY, [])
        ))
    )

    if time_step >= n_frames:
        raise ValueError(
            "Unphysical value for time_step which should be smaller than the number of frames"
        )

    positions = (
        np.asarray([frame.positions for frame in trajectory_data])
        if trajectory_type == "ase"
        else trajectory_data[key_mapping[POSITIONS_KEY]]
    )
    displacement_vectors = np.asarray(
        [
            frame2 - frame1
            for frame1, frame2 in zip(positions[:-time_step], positions[time_step:])
        ]
    )

    lattice_vectors = (
        trajectory_data[0].cell
        if trajectory_type == "ase"
        else trajectory_data.get(key_mapping.get(CELL_KEY), [])
    )
    pbc = (
        trajectory_data[0].pbc
        if trajectory_type == "ase"
        else trajectory_data.get(key_mapping.get(PBC_KEY), [False, False, False])
    )
    if all(pbc):
        displacement_vectors = align_vectors_with_periodicity(
            displacement_vectors, lattice_vectors
        )
    if trajectory_type == "ase":
        [
            frame.arrays.__setitem__(DISPLACEMENTS_KEY, displacement)
            for frame, displacement in zip(
                trajectory_data[:-time_step], displacement_vectors
            )
        ]
        [
            frame.info.__setitem__(TIMESTEP_KEY, time_between_frames * time_step)
            for frame in trajectory_data[:-time_step]
        ]
        
    elif  trajectory_type == "hdf5":
        trajectory_data[DISPLACEMENTS_KEY] = displacement_vectors
        trajectory_data[TIMESTEP_KEY] = np.array([time_between_frames * time_step])    
    else:
        trajectory_data[DISPLACEMENTS_KEY] = displacement_vectors
        trajectory_data[TIMESTEP_KEY] = np.array([time_between_frames * time_step])

    return trajectory_data


def get_desired_field_values_of_next_frame(
    fields: List,
    trajectory_data: Union[List[ase.Atoms], Dict],
    time_step: Optional[int] = 1,
    key_mapping: Optional[Dict[str, str]] = {},
):
    if (isinstance(trajectory_data, list) and 
        isinstance(trajectory_data[0], ase.Atoms)):
        trajectory_type = "ase"
    elif isinstance(trajectory_data, dict):
        if (POSITIONS_KEY in trajectory_data or 
            key_mapping.get(POSITIONS_KEY) in trajectory_data):
            trajectory_type = "hdf5"
    else:
        trajectory_type = "unknown"
    
    n_frames = (
        len(trajectory_data)
        if trajectory_type == "ase"
        else len(trajectory_data.get(key_mapping[POSITIONS_KEY]))
    )

    if time_step >= n_frames:
        raise ValueError(
            "Unphysical value for time_step which should be smaller than the number of frames"
        )

    if trajectory_type == "ase":
        for field in fields:
            new_field = f"{UPDATE_KEY}_{field}"
            if field in ASE_INFO_FIELDS:
                [
                    frame.info.__setitem__(
                        new_field,
                        trajectory_data[count + time_step].info[key_mapping[field]],
                    )
                    for count, frame in enumerate(trajectory_data[:-time_step])
                ]

            else:
                [
                    frame.arrays.__setitem__(
                        new_field,
                        trajectory_data[count + time_step].arrays[key_mapping[field]],
                    )
                    for count, frame in enumerate(trajectory_data[:-time_step])
                ]

    elif trajectory_type == "hdf5":
        for field in fields:
            new_field = f"{UPDATE_KEY}_{field}"
            field_key = key_mapping.get(field, field)
            if field_key in trajectory_data:
                trajectory_data[new_field] = trajectory_data[field_key][time_step:]
            else:
                raise KeyError(f"Field {field} not found in HDF5 data")

    return trajectory_data

