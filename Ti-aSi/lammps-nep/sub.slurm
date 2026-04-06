#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --job-name=nep-Ti-aSi
#SBATCH --nodes=1
#SBATCH --nodelist=mgt
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --output=out.%j
#SBATCH --error=err.%j

module load mkl/latest
module load mpi/latest
module load cuda/11.6
module load conda/3-2023.09

export OMP_NUM_THREADS=4
export MP_NUM_THREADS=1

echo "Starting date : `date`"

mpirun -np $SLURM_NPROCS /dat/Jaafar/tools/lammps-nep/lammps-22Jul2025/build/lmp -in lmp.in > outlog

echo "Program finished with exit code $? at: `date`"
