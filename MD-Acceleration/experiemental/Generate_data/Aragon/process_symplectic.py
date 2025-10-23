import ase.io
import numpy as np
import torch
import tqdm
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import NeighborListOptions, systems_to_torch
from metatrain.utils.data.writers import DiskDatasetWriter
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
import copy
import sys
from ase.data import atomic_masses


type = sys.argv[1]
if type == "train":
    traj_range = range(1, 9)
elif type == "val":
    traj_range = range(9, 10)
elif type == "test":
    traj_range = range(10, 11)

time_lags = [128]
disk_dataset_writers = {time_lag: DiskDatasetWriter(f"argon_{time_lag}_{type}_symplectic.zip") for time_lag in time_lags}
correlation_time = 400

def write_to_dataset(frame_now, frame_ahead, time_lag, i, disk_dataset_writer):
    frame_now = copy.deepcopy(frame_now)
    frame_ahead = copy.deepcopy(frame_ahead)
    frame_now.numbers[:] = 18  # Argon has atomic number 18
    frame_now.arrays["numbers"][:] = 18  # Ensure numbers are set correctly
    frame_ahead.numbers[:] = 18  # Argon has atomic number 18
    frame_ahead.arrays["numbers"][:] = 18  # Ensure numbers are set correctly
    frame_now.arrays["momenta"][:] *= atomic_masses[18]/atomic_masses[1]
    frame_ahead.arrays["momenta"][:] *= atomic_masses[18]/atomic_masses[1]

    frame_average = copy.deepcopy(frame_now)
    frame_average.set_positions(
        (frame_now.get_positions() + frame_ahead.get_positions()) / 2.0
    )
    frame_average.set_momenta(
        (frame_now.get_momenta() + frame_ahead.get_momenta()) / 2.0
    )

    system = systems_to_torch(frame_average, dtype=torch.float64)
    system = get_system_with_neighbor_lists(
        system,
        [NeighborListOptions(cutoff=5.0, full_list=True, strict=True)],
    )
    system.add_data(
        "momenta",
        TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.tensor(frame_average.get_momenta(), dtype=torch.float64).unsqueeze(-1),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor([[i, j] for j in range(len(frame_average))]),
                    ),
                    components=[Labels(names="xyz", values=torch.tensor([[0], [1], [2]]))],
                    properties=Labels.single(),
                )
            ],
        )
    )
    masses = frame_average.get_masses()[:, np.newaxis]
    system.add_data(
        "masses",
        TensorMap(
            keys=Labels.single(),
            blocks = [
                TensorBlock(
                    values=torch.tensor(masses, dtype=torch.float64),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor([[i, j] for j in range(len(frame_now))]),
                    ),
                    components=[],
                    properties=Labels.single(),
                )
            ],
        )
    )

    distances = frame_ahead.get_positions() - frame_now.get_positions()
    if np.any(np.abs(distances) > 10.0 * 0.25 * time_lag):
        # cut anything over 10 angstrom/fs speed, these have to be wrong
        return False
    delta_q = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor(distances, dtype=torch.float64).unsqueeze(-1),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[i, j] for j in range(len(frame_average))]),
                ),
                components=[Labels(names="xyz", values=torch.tensor([[0], [1], [2]]))],
                properties=Labels.single(),
            )
        ],
    )
    delta_p = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor(frame_ahead.get_momenta()-frame_now.get_momenta(), dtype=torch.float64).unsqueeze(-1),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[i, j] for j in range(len(frame_average))]),
                ),
                components=[Labels(names="xyz", values=torch.tensor([[0], [1], [2]]))],
                properties=Labels.single(),
            )
        ],
    )
    disk_dataset_writer.write_sample(
        system,
        {
            f"mtt::delta_{time_lag}_q": delta_q,
            f"mtt::delta_{time_lag}_p": delta_p,
        }
    )
    return True

assert len(time_lags) == 1, "Only one time lag is supported at the moment"
# see structure counting below

structure_counter = 0
for trj_num in traj_range:
    print(trj_num)
    traj = ase.io.read(f'dump_{trj_num}.lammpstrj', index=':')
    traj_len = len(traj)
    for i in tqdm.tqdm(range(0, traj_len-max(time_lags), correlation_time)):
        for time_lag in time_lags:
            frame_now = traj[i]
            frame_ahead = traj[i+time_lag]
            written = write_to_dataset(frame_now, frame_ahead, time_lag, structure_counter, disk_dataset_writers[time_lag])
            if written:
                structure_counter += 1
            else:
                print("Not written")
            # frame_now_trev = copy.deepcopy(frame_now)
            # frame_ahead_trev = copy.deepcopy(frame_ahead)
            # frame_now_trev.set_momenta(-frame_now_trev.get_momenta())
            # frame_ahead_trev.set_momenta(-frame_ahead_trev.get_momenta())
            # written = write_to_dataset(frame_ahead_trev, frame_now_trev, time_lag, structure_counter+1, disk_dataset_writers[time_lag])
            # if written:
            #     structure_counter += 1
            # else:
            #     print("Not written")

for k in list(disk_dataset_writers.keys()):
    disk_dataset_writer = disk_dataset_writers.pop(k)
    del disk_dataset_writer
