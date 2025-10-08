#!/bin/bash

#SBATCH --job-name=data-generate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g01 
#SBATCH --output=out.%j
#SBATCH --error=err.%j

module load conda/3-2023.09
module load cuda/11.6
module load lammps-23Jun2022

export OMP_NUM_THREADS=4

echo "Starting date : `date`"

for i in {1..10}
do
    sed "s/SEED/$i/g" argon.in > argon_$i.in
    lmp_mpi -in argon_$i.in > argon_$i.out
done

echo "Program finished with exit code $? at: `date`"
