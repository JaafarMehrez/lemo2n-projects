"""
Dataset containing information (positions, species, etc.) of a collection of atomic configurations.

1. Added support for .h5 formats
2. Added local unit test functions

TODO 

1. Include partial charge during graph construction, refer to AtomicGraph
2. Add partial charge and updated partial charge to data augmentation

Future feature

1. Replace data augmentation with NN with built-in physical symmetry preservation

Modified by : Jaaar Mehrez
"""

import glob
import os
from typing import Dict, List, Optional, Union

import ase
import ase.io
import torch
from numpy import load
from torch_geometric.data import InMemoryDataset

from _keys import POSITIONS_KEY
from atomic_graph import AtomicGraph
from misc import (
    convert_ase_atoms_to_dictionary,
    convert_npz_to_dictionary,
    convert_hd5_to_dictionary,
    format_values_in_dictionary,
    guess_filetype,
)


class AtomicGraphDataset(InMemoryDataset):
    """each frame to be converted into an atomic graph"""

    def __init__(
        self,
        root: str,
        name: str,
        cutoff_radius: float,
        files: Optional[Union[List[str], str]] = "*",
        atom_type_mapper: Optional[Dict[int, int]] = {},
        time_reversibility: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        rename=True,
        **ase_kwargs,
    ):

        if rename:
            precision_str = str(torch.get_default_dtype())[-2:]
            time_reversibility_str = ";TR" if time_reversibility else ""
            self.name = f"{name};P:{precision_str}{time_reversibility_str}"
        else:
            self.name = name
        self.root = root
        self.cutoff_radius = cutoff_radius
        self.files = files
        self.atom_type_mapper = atom_type_mapper
        self.time_reversibility = time_reversibility

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.num_nodes = sum([config.num_nodes for config in self])

    @property
    def raw_file_names(self) -> List[str]:
        if isinstance(self.files, str):
            path_list = glob.glob(os.path.join(self.raw_dir, self.files))
            return [os.path.basename(path) for path in path_list]
        else:
            return self.files

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_file_names(self) -> List[str]:
        return [
            f"{self.name}.pt"
        ]

    @property
    def processed_dir(self) -> str:
        return self.root

    def process_pre_filter(self):
        pass

    def process_pre_transform(self):
        pass

    '''should perform the same thing for partial charge and updated partial charge -- later this augementation shall be removed'''
    
    def _augment_for_time_reversibility(self, atoms_dictionary):
        atoms_dictionary["update_velocities"], atoms_dictionary["velocities"] = (
            -atoms_dictionary["velocities"],
            -atoms_dictionary["update_velocities"],
        )
        atoms_dictionary["positions"] = (
            atoms_dictionary["positions"] + atoms_dictionary["displacements"]
        )
        atoms_dictionary["displacements"] = -atoms_dictionary["displacements"]

        rev_atomic_graph_data = AtomicGraph.from_atoms_dict(
            atoms_dict=atoms_dictionary,
            r_cut=self.cutoff_radius,
            atom_type_mapper=self.atom_type_mapper,
        )

        return rev_atomic_graph_data

    def process(self):
        data_list = []
        for count, raw_path in enumerate(self.raw_paths):
            filetype = guess_filetype(self.raw_file_names[count])
            if filetype == "npz":
                raw_data = load(raw_path)
                npz_dictionary = convert_npz_to_dictionary(raw_data)
                npz_dictionary = format_values_in_dictionary(npz_dictionary)
                for index in range(npz_dictionary[POSITIONS_KEY].shape[0]):
                    atoms_dictionary = {key: value[index] for key, value in npz_dictionary.items()}
                    atoms_dictionary = format_values_in_dictionary(atoms_dictionary)
                        # Add atomic masses if missing
                    if 'atomic_masses' not in atoms_dictionary and 'atomic_numbers' in atoms_dictionary:
                        try:
                            import numpy as np
                            from ase.data import atomic_masses as ase_atomic_masses
                            atomic_numbers = np.array(atoms_dictionary['atomic_numbers'])
                            atoms_dictionary['atomic_masses'] = ase_atomic_masses[atomic_numbers]
                        except Exception as e:
                            print(f"Warning: Could not compute atomic masses: {e}")

                        atomic_graph_data = AtomicGraph.from_atoms_dict(
                        atoms_dict=atoms_dictionary,
                        r_cut=self.cutoff_radius,
                        atom_type_mapper=self.atom_type_mapper,
                    )
                    data_list.append(atomic_graph_data)

            elif filetype == "h5":
                import h5py
                with h5py.File(raw_path, 'r') as h5_file:
                    hd5_dictionary = convert_hd5_to_dictionary(h5_file)
                hd5_dictionary = format_values_in_dictionary(hd5_dictionary)
                
                for index in range(hd5_dictionary[POSITIONS_KEY].shape[0]):
                    atoms_dictionary = {}
                    for key, value in hd5_dictionary.items():
                        # Only index if value is array and has enough dimensions
                        if hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] > index:
                            item = value[index]
                            # If item is still a scalar, convert to array if needed
                            if hasattr(item, 'shape') and item.shape == ():
                                item = item.reshape(1)
                            atoms_dictionary[key] = item
                        else:
                            atoms_dictionary[key] = value  # assign scalar directly
                    # Optionally: print shapes for debugging
                    # for k, v in atoms_dictionary.items():
                    #     print(f"{k}: {type(v)}, {getattr(v, 'shape', None)}")
                    atoms_dictionary = format_values_in_dictionary(atoms_dictionary)
                    # Add atomic masses if missing
                    if 'atomic_masses' not in atoms_dictionary and 'atomic_numbers' in atoms_dictionary:
                        try:
                            import numpy as np
                            import ase
                            from ase.data import atomic_masses as ase_atomic_masses
                            atomic_numbers = np.array(atoms_dictionary['atomic_numbers'])
                            atoms_dictionary['atomic_masses'] = ase_atomic_masses[atomic_numbers]
                        except Exception as e:
                            print(f"Warning: Could not compute atomic masses: {e}")
                            
                    atomic_graph_data = AtomicGraph.from_atoms_dict(
                        atoms_dict=atoms_dictionary,
                        r_cut=self.cutoff_radius,
                        atom_type_mapper=self.atom_type_mapper,
                    )
                    data_list.append(atomic_graph_data)

            else:
                raw_data = ase.io.read(raw_path, format=filetype, index=":")

                for index, config in enumerate(raw_data):
                    atoms_dictionary = convert_ase_atoms_to_dictionary(config)
                    atoms_dictionary = format_values_in_dictionary(atoms_dictionary)
                    rev_atoms_dictionary = atoms_dictionary.copy()

                    atomic_graph_data = AtomicGraph.from_atoms_dict(
                        atoms_dict=atoms_dictionary,
                        r_cut=self.cutoff_radius,
                        atom_type_mapper=self.atom_type_mapper,
                    )

                    # time-reversed configuration, that's for symmetry, what if the NN is symmetry-aware? 
                    if (
                        "update_velocities" in rev_atoms_dictionary
                        and self.time_reversibility
                    ):
                        rev_atomic_graph_data = self._augment_for_time_reversibility(
                            rev_atoms_dictionary
                        )
                        data_list.append(rev_atomic_graph_data)
                    data_list.append(atomic_graph_data)

        torch.save(self.collate(data_list=data_list), self.processed_paths[0])

'''Some testing'''
def test_extxyz_dataset():
    filename = "traj_with_displacement.extxyz"
    dataset = AtomicGraphDataset(
        root='./data/',
        name='test_extxyz',
        cutoff_radius=5.0,
        files="traj_with_displacement.extxyz",
        atom_type_mapper={14:1,8:2},
        time_reversibility=True,
        rename=True
    )

    print(f"Number of graphs: {len(dataset)}")
    print(f"First graph: {dataset[0]}")
    print(f"Number of nodes in first graph: {dataset[0].num_nodes}")

    return dataset

def test_h5_dataset():
    
    dataset = AtomicGraphDataset(
        root='./data/',
        name='test_h5',
        cutoff_radius=5.0,
        files="traj_hdf5.h5",
        atom_type_mapper={14:1,8:2},
        time_reversibility=True
    )
    
    print(f"Number of graphs: {len(dataset)}")
    print(f"First graph: {dataset[0]}")
    print(f"Number of nodes in first graph: {dataset[0].num_nodes}")
    return dataset

#test_extxyz_dataset()
test_h5_dataset()