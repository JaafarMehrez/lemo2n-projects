#!/bin/bash

#SBATCH --job-name=traj-gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g02 
#SBATCH --output=out.%j
#SBATCH --error=err.%j

module load conda/3-2023.09
module load mpi/2021.5.1

export OMP_NUM_THREADS=4

echo "Starting date : `date`"

#for i in {1..10}
#do
#    sed "s/SEED/$i/g" lmp.in > lmp_$i.in
#    lmp -in lmp_$i.in > lmp_$i.out
#done

lmp -in lmp.in > lmp.out

echo "Program finished with exit code $? at: `date`"
