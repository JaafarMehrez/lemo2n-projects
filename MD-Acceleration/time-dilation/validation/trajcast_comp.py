import copy
from typing import Optional, Tuple, List, Union
import numpy as np
from ase import Atoms
from ase.io import read

from utils.misc import get_least_common_multiple
from physical_behaviour import track_instantaneous_temperature


def compute_particle_position_error(
    predicted_trajectory: Union[List[Atoms], str],
    reference_trajectory: Union[List[Atoms], str],
    mode: Optional[str] = "mae",
    tolerance: Optional[float] = 1e-6,
) -> Tuple[np.ndarray, float]:
    """
    Compute position errors between predicted and reference trajectories.
    
    Args:
        predicted_trajectory: Predicted trajectory (list of Atoms or file path)
        reference_trajectory: Reference trajectory (list of Atoms or file path)
        mode: Error metric - "mae", "mse", or "rmse"
        tolerance: Numerical tolerance for position comparison
        
    Returns:
        Tuple of (error array, time between frames in fs)
    """
    # Load trajectories if file paths are provided
    if isinstance(predicted_trajectory, str):
        predicted_trajectory = read(predicted_trajectory, index=':')
    if isinstance(reference_trajectory, str):
        reference_trajectory = read(reference_trajectory, index=':')
    
    pred_traj = predicted_trajectory
    ref_traj = reference_trajectory

    # First we check whether both trajectories have the same starting point
    if not np.allclose(
        pred_traj[0].positions,
        ref_traj[0].positions,
        atol=tolerance
    ):
        raise ValueError(
            "For this test, trajectories need to start from same frame!"
        )

    # Check if velocities are identical in initial frame (if available)
    if (hasattr(pred_traj[0], 'get_velocities') and 
        hasattr(ref_traj[0], 'get_velocities') and
        pred_traj[0].get_velocities() is not None and
        ref_traj[0].get_velocities() is not None):
        
        if not np.allclose(
            pred_traj[0].get_velocities(),
            ref_traj[0].get_velocities(),
            atol=tolerance
        ):
            raise ValueError(
                "Check that velocities are also identical in initial frame!"
            )

    # Align trajectories (assuming equal timesteps for now - see note below)
    # For ASE, we need to handle timestep alignment differently
    min_frames = min(len(pred_traj), len(ref_traj))
    
    # For ASE trajectories, we assume equal timesteps or user provides alignment
    # In practice, you might want to add dt_pred and dt_ref as parameters
    time_between_frames = 1.0  # Default, should be provided by user
    
    # based on input, compute the required statistics
    error_function = {
        "mae": np.mean,
        "mse": lambda x: np.mean(np.square(x)),
        "rmse": lambda x: np.sqrt(np.mean(np.square(x))),
    }[mode]
    
    # now we loop over them
    errors = []
    for frame_idx in range(min_frames):
        frame_pred = pred_traj[frame_idx]
        frame_ref = ref_traj[frame_idx]
        
        errors.append(
            error_function(
                np.linalg.norm(frame_pred.positions - frame_ref.positions, axis=1)
            )
        )

    return (np.asarray(errors), time_between_frames * 1000)


def compare_instantaneous_temperature(
    predicted_trajectory: Union[List[Atoms], str],
    reference_trajectory: Union[List[Atoms], str],
    linear_momentum_fixed: Optional[bool] = False,
    unit_velocity_predicted: Optional[str] = "angstroms/femtoseconds",
    unit_velocity_reference: Optional[str] = "angstroms/femtoseconds",
    tolerance: Optional[float] = 1e-6,
) -> Tuple[np.ndarray, float]:
    """
    Compare instantaneous temperatures between predicted and reference trajectories.
    
    Args:
        predicted_trajectory: Predicted trajectory (list of Atoms or file path)
        reference_trajectory: Reference trajectory (list of Atoms or file path)
        linear_momentum_fixed: Whether linear momentum was fixed in simulation
        unit_velocity_predicted: Velocity units for predicted trajectory
        unit_velocity_reference: Velocity units for reference trajectory
        tolerance: Numerical tolerance for position comparison
        
    Returns:
        Tuple of (temperature difference array, time between frames)
    """
    # Load trajectories if file paths are provided
    if isinstance(predicted_trajectory, str):
        predicted_trajectory = read(predicted_trajectory, index=':')
    if isinstance(reference_trajectory, str):
        reference_trajectory = read(reference_trajectory, index=':')
    
    pred_traj = predicted_trajectory
    ref_traj = reference_trajectory

    # Check initial positions match
    if not np.allclose(
        pred_traj[0].positions,
        ref_traj[0].positions,
        atol=tolerance
    ):
        raise ValueError(
            "For this test, trajectories need to start from same frame!"
        )

    # Check initial velocities match (if available)
    if (hasattr(pred_traj[0], 'get_velocities') and 
        hasattr(ref_traj[0], 'get_velocities') and
        pred_traj[0].get_velocities() is not None and
        ref_traj[0].get_velocities() is not None):
        
        if not np.allclose(
            pred_traj[0].get_velocities(),
            ref_traj[0].get_velocities(),
            atol=tolerance
        ):
            raise ValueError(
                "Check that velocities are also identical in initial frame!"
            )

    # For ASE, use minimum common frames
    min_frames = min(len(pred_traj), len(ref_traj))
    time_between_frames = 1.0  # Should be provided by user

    # compute temperatures
    pred_temperature = track_instantaneous_temperature(
        pred_traj[:min_frames], linear_momentum_fixed, unit_velocity_predicted
    )

    ref_temperature = track_instantaneous_temperature(
        ref_traj[:min_frames], linear_momentum_fixed, unit_velocity_reference
    )

    return (np.abs(pred_temperature - ref_temperature), time_between_frames)


# Note: The original _align_trajectories function was MDAnalysis-specific
# For ASE, trajectory alignment would need to be handled differently
# since ASE doesn't store timestep information in a standardized way
