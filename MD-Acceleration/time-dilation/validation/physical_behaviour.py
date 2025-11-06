"""
This is a collection of tests validating the structures and trajectories predicted
by the model obey the laws of physics. Generally, these functions do not need a reference
trajectory to compare to.

Author: Fabian Thiemann
Modified: Jaafar Mehrez (jaafarmehrez`sjtu.edu.cn)
"""

from typing import Optional, Tuple, List, Union
import numpy as np
from ase.units import J, kB, kg
from ase import Atoms
from ase.io import read

from utils.atomic_computes import (
    align_vectors_with_periodicity,
    cell_parameters_to_lattice_vectors,
)
from utils.misc import convert_units


def track_min_max_distances_in_molecule(
    trajectory: Union[List[Atoms], str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the time evolution of the minimum and maximum distance in a molecule.

    Args:
        trajectory: Can be either a list of ASE Atoms objects or a path to a trajectory file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Array of min and max distances.
    """
    # Load trajectory if path is provided
    if isinstance(trajectory, str):
        trajectory = read(trajectory, index=':')
    
    # based on the first frame we need identify the two atom pairs
    # one for the min distance and one for the max distance
    # start by computing vectors between all atoms
    first_frame = trajectory[0]
    positions = first_frame.positions
    vectors = positions[:, np.newaxis] - positions

    # make sure everything is in line with nearest image convention and pbc
    vectors = align_vectors_with_periodicity(
        vectors=vectors,
        lattice_vectors=cell_parameters_to_lattice_vectors(
            params=first_frame.cell.cellpar()
        ),
    )

    # now compute distances
    distances = np.linalg.norm(vectors, axis=2)

    # now find min and max indices
    # Mask diagonal to avoid zero distances (same atom)
    mask = ~np.eye(distances.shape[0], dtype=bool)
    min_distance = np.min(distances[mask])
    max_distance = np.max(distances)
    
    min_indices = np.argwhere(distances == min_distance)[0]
    max_indices = np.argwhere(distances == max_distance)[0]

    # instantiate arrays
    min_distances = []
    max_distances = []

    # now let's loop over the trajectory
    for frame in trajectory:
        # Get cell parameters for periodicity handling
        cell_params = frame.cell.cellpar()
        
        # save minimum distance
        min_vector = frame.positions[min_indices[0]] - frame.positions[min_indices[1]]
        min_vector = align_vectors_with_periodicity(
            min_vector.reshape(1, -1),  # Reshape for compatibility
            lattice_vectors=cell_parameters_to_lattice_vectors(params=cell_params),
        )
        min_distances.append(np.linalg.norm(min_vector))
        
        # save maximum distance
        max_vector = frame.positions[max_indices[0]] - frame.positions[max_indices[1]]
        max_vector = align_vectors_with_periodicity(
            max_vector.reshape(1, -1),  # Reshape for compatibility
            lattice_vectors=cell_parameters_to_lattice_vectors(params=cell_params),
        )
        max_distances.append(np.linalg.norm(max_vector))

    return np.asarray(min_distances), np.asarray(max_distances)


def track_average_distance_from_molecule_center_of_mass(
    trajectory: Union[List[Atoms], str]
) -> np.ndarray:
    """Computes the time evolution of average distance over all atoms in a molecule from its center of mass.
        Note: We assume an unwrapped trajectory here for now.
    Args:
        trajectory: Can be either a list of ASE Atoms objects or a path to a trajectory file.

    Returns:
        (np.ndarray): Array of distances from center of mass.
    """
    # Load trajectory if path is provided
    if isinstance(trajectory, str):
        trajectory = read(trajectory, index=':')

    # instantiate arrays
    mean_distances = []

    # now let's loop over the trajectory
    for frame in trajectory:
        # get center of mass using ASE
        COM = frame.get_center_of_mass()

        # next compute vectors from COM
        vectors = frame.positions - COM

        # get distances in line with pbcs
        distances = np.linalg.norm(
            align_vectors_with_periodicity(
                vectors,
                lattice_vectors=cell_parameters_to_lattice_vectors(
                    params=frame.cell.cellpar()
                ),
            ),
            axis=1,
        )

        # take average and save
        mean_distances.append(np.mean(distances))

    return np.asarray(mean_distances)


def track_instantaneous_temperature(
    trajectory: Union[List[Atoms], str],
    linear_momentum_fixed: Optional[bool] = False,
    angular_momentum_fixed: Optional[bool] = False,
    unit_velocity: Optional[str] = "angstroms/femtoseconds",
) -> np.ndarray:
    """Computes the time evolution of the instantanteous temperature based on the atomic velocities.

    Args:
        trajectory: Can be either a list of ASE Atoms objects or a path to a trajectory file.
        linear_momentum_fixed (Optional[bool], optional): True if the center of mass motion has been removed from the trajectory.
            Defaults to False to not remove the translational degrees of freedom of a molecule.
        angular_momentum_fixed (Optional[bool], optional): True if angular momentum is removed in trajectory generation.
            Defaults to False.
        unit_velocity (Optional[str], optional): Unit of the velocities in the trajectory. Defaults to "angstroms/femtoseconds".

    Raises:
        AttributeError: If trajectory frames don't have velocities.

    Returns:
        np.ndarray: Array of instantaneous temperatures.
    """
    # Load trajectory if path is provided
    if isinstance(trajectory, str):
        trajectory = read(trajectory, index=':')

    # instantiate array
    temperatures = []

    # get masses for each atom in kg per particle (ASE uses atomic mass units)
    first_frame = trajectory[0]
    masses = first_frame.get_masses() / kg  # Convert from amu to kg

    # get the conversion factor for the velocities
    units_distance_time = unit_velocity.replace(" ", "").split("/")
    conv_factor_velocity = convert_units(
        origin=units_distance_time[0], target="meter"
    ) / convert_units(origin=units_distance_time[1], target="seconds")

    # check the trajectory has velocities
    if not hasattr(first_frame, 'get_velocities') or first_frame.get_velocities() is None:
        raise AttributeError(
            "The trajectory does not have velocities, therefore we cannot compute the temperature."
        )

    # compute the degrees of freedom
    degrees_of_freedom = 3 * len(masses)

    if linear_momentum_fixed:
        degrees_of_freedom -= 3
    if angular_momentum_fixed:
        degrees_of_freedom -= 3

    # now let's loop over the trajectory
    for frame in trajectory:
        # get velocities from ASE Atoms object
        velocities = frame.get_velocities()
        
        # compute momenta in kg*m/s
        momenta = (
            masses * np.linalg.norm(velocities, axis=1) * conv_factor_velocity
        )

        # compute instantaneous temperature
        # D. Frenkel and B Smit; "Understanding Molecular Simulation: Algorithms to Applications", Academic Press, 2nd Edition, 2002, page 64, equation 4.1.2
        temp_inst = (
            np.sum(np.square(momenta) / masses) / (kB / J) / degrees_of_freedom
        )

        # append to list
        temperatures.append(temp_inst)

    return np.asarray(temperatures)
