import torch
from trajcast.model.models import EfficientTrajCastModel
from trajcast.model.forecast import Forecast
import ase.io

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

torch.set_default_dtype(torch.float64)

# Initialize model
model = EfficientTrajCastModel.build_from_yaml("../config.yaml")

# Load state dictionary
model.load_state_dict(
    torch.load(
        "../model_params.pt",
        map_location=device,
    )
)

# Read the initial configuration (last frame from trajectory)
start_frame = ase.io.read(
    "../data/test.extxyz",
    index="-1",  # Last frame
)
start_frame.center()

# Remove prediction-related arrays if present
if "displacements" in start_frame.arrays:
    _ = start_frame.arrays.pop("displacements")
if "update_velocities" in start_frame.arrays:
    _ = start_frame.arrays.pop("update_velocities")

# Forecast protocol configuration
protocol = {
    "units": "real",  # Units convention matching LAMMPS
    "run": 10000,  # Number of steps to forecast
    "temperature": 300.0,  # Target temperature in Kelvin
    "extra_dof": 6,  # Degrees of freedom to subtract from 3N for temperature
    "timestep": 7.0,  # Prediction horizon (fs)
    "configuration": start_frame,  # Initial atomic configuration
    "model_type": "EfficientTrajCastModel",  # Model type
    "model": model,  # Trained model instance
    "thermostat": {
        "Tdamp": 70.0  # Thermostat damping parameter (fs)
    },
    "velocities": {  # Initial velocity assignment
        "Temperature": 300,  # Initial temperature for velocities
        "linear": True,  # Remove total linear momentum
        "angular": True,  # Remove total angular momentum
        "distribution": "gaussian",  # Velocity distribution
    },
    "write": {  # Output settings
        "filename": "./forecasted_traj.extxyz",  # Output file path
        "every": 10,  # Save every frame
    },
    "device": device,  # Computation device
    "seed": 42,  # Random seed for reproducibility
    "set_momenta": {  # Target momenta
        "linear": torch.zeros(3, device=device),
        "angular": torch.tensor([], device=device),
    },
    "zero_momentum": {  # Momentum removal during simulation
        "every": 100,  # Remove momenta every N steps
        "linear": False,  # Don't remove linear momentum
        "angular": True,  # Remove angular momentum
    },
}

# Create forecaster and generate trajectory
print("Starting trajectory forecast...")
forecaster = Forecast(protocol=protocol)
forecaster.generate_trajectory()
print(f"Trajectory saved to: {protocol['write']['filename']}")

from ase import io, visualize
traj = io.read("forecasted_traj.extxyz", index=":")
visualize.view(traj)

# For analysis with ASE
traj = io.read("forecasted_traj.extxyz", index=":")
print(f"Trajectory contains {len(traj)} frames")
print(f"Each frame has {len(traj[0])} atoms")
