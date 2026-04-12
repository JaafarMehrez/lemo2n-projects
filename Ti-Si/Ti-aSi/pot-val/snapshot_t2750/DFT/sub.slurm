#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --job-name=SCF-t2750
#SBATCH --nodes=1
#SBATCH --nodelist=mgt
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1

module load mkl mpi
module load cuda/12.4
module load pwmat/20240322

export I_MPI_FABRICS=ofi

mpirun -genvall -np $SLURM_NPROCS PWmat | tee output
