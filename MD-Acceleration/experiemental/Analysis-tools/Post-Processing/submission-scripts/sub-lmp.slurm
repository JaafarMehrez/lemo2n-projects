#!/bin/bash
#SBATCH --account=def-nike-ab
#SBATCH --mem=4G      # memory; default unit is megabytes.
#SBATCH --time=0-23:00           # time (DD-HH:MM).
#SBATCH --cpus-per-task=4
#SBATCH --output=out.%j
#SBATCH --error=err.%j

module load StdEnv/2023
module load intel/2023.2.1
module load openmpi/4.1.5
module load gcc/12.3
module load lammps-omp/20250722

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Starting run at: `date`"

echo "=========================================="
echo "Running 5 NVE Production Simulations"
echo "Each: 100 ps (200,000 steps at 0.5 fs)"
echo "=========================================="

for i in {1..5}; do
    DATA_FILE="frame_${i}.dump"

    if [ ! -f "$DATA_FILE" ]; then
        echo "Error: $DATA_FILE not found! Run conversion first."
        exit 1
    fi

    echo ""
    echo "Starting NVE simulation $i from $DATA_FILE"
    echo "------------------------------------------"

    # Create LAMMPS input file for this run
    cat > nve_${i}.in << EOF
# SiO2 NVE Production Run ${i}
# Starting from extracted configuration
# Total: 100 ps (200,000 steps at 0.5 fs)

units real
boundary p p p
atom_style full
timestep 0.5
neighbor 2.0 bin
neigh_modify delay 0 every 5 check yes

# ===== LOAD EXTRACTED CONFIGURATION =====
read_data ${DATA_FILE}

# ===== ReaxFF FORCE FIELD =====
pair_style reaxff NULL checkqeq yes
pair_coeff * * ffield.reax Si O H

# QEQ for charge equilibration
fix qeq all qeq/reax 1 0.0 10.0 1.0e-3 reaxff

# ===== NVE ENSEMBLE (NO THERMOSTAT) =====
# Pure NVE for unbiased trajectory (as in paper)
fix nve_integrator all nve

# Remove center-of-mass motion
fix momentum all momentum 100 linear 1 1 1 angular

# ===== OUTPUT SETTINGS =====
thermo 10000
thermo_style custom step temp pe ke etotal

# ===== CRITICAL: DUMP EVERY STEP =====
# TrajCast needs high-frequency data (0.5 fs intervals)
# We'll dump every step for maximum Δt flexibility
dump nve_dump all custom 5 nve_${i}.lammpstrj id type x y z vx vy vz
dump_modify nve_dump format float %.16f

# ===== RUN PRODUCTION =====
# 100 ps = 200,000 steps at 0.5 fs timestep
run 200000

# Write final restart
write_restart final_nve_${i}.restart

print "NVE simulation ${i} complete!"
print "Trajectory saved to: nve_${i}.lammpstrj"
print "Total steps: 200,000 (100 ps at 0.5 fs)"
EOF

    # Run the simulation
    echo "Running simulation ${i}..."
    lmp -sf omp -in nve_${i}.in > nve_${i}.out

    # Check if successful
    if [ -f "nve_${i}.lammpstrj" ]; then
        # Count frames in trajectory
        FRAMES=$(grep -c "ITEM: TIMESTEP" nve_${i}.lammpstrj)
        echo "  ✓ Success: Generated nve_${i}.lammpstrj with ${FRAMES} frames"
    else
        echo "  ✗ Error: Failed to generate nve_${i}.lammpstrj"
        echo "  Check nve_${i}.log for errors"
    fi
done

echo ""
echo "=========================================="
echo "All 5 NVE simulations completed!"
echo "=========================================="
echo "Generated files:"
echo "  Trajectories: nve_1.lammpstrj ... nve_5.lammpstrj"
echo "  Log files: nve_1.log ... nve_5.log"
echo "  Restart files: final_nve_1.restart ... final_nve_5.restart"
echo ""
echo "Each trajectory: 200,000 frames (100 ps at 0.5 fs)"
echo "Ready for TrajCast preprocessing!"

echo "Program finished with exit code $? at: `date`"
