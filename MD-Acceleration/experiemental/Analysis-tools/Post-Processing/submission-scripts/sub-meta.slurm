#!/bin/bash
#SBATCH --account=def-nike-ab
#SBATCH --mem=4G
#SBATCH --time=0-23:00
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
echo "Running 5 Metadynamics Simulations"
echo "=========================================="

for i in {1..5}; do
    DATA_FILE="frame_${i}.dump"

    if [ ! -f "$DATA_FILE" ]; then
        echo "Error: $DATA_FILE not found! Run conversion first."
        exit 1
    fi

    echo ""
    echo "Starting metadynamics simulation $i from $DATA_FILE"
    echo "------------------------------------------"

    # Create plumed input file for this run
    cat > plumed_${i}.dat << EOF
# Well-tempered metadynamics on the Si-H bond length with restricted sampling
# For simulation ${i}

UNITS LENGTH=A TIME=fs ENERGY=kj/mol

# CV definitions
# IMPORTANT: Check if these atom indices are consistent across all frames!
sih: DISTANCE ATOMS=153,469
angle_phi: ANGLE ATOMS=469,153,94

# Restrict bond distance to within 4Å as mentioned in paper
UPPER_WALLS ARG=sih AT=4.0 KAPPA=1000.0 EXP=2 EPS=1 OFFSET=0

METAD ARG=sih,angle_phi SIGMA=0.5,0.5 HEIGHT=1 PACE=50 BIASFACTOR=80 TEMP=300 FILE=HILLS_${i}

# Print CVs during metadynamics
PRINT ARG=sih,angle_phi STRIDE=100 FILE=COLVAR_METAD_${i}
EOF

    # Create LAMMPS input file - EXACTLY matching reference structure
    cat > meta_${i}.in << EOF
# SiO2 Metadynamics Run ${i}
# Following exact protocol from reference input

variable t_lo equal 300
variable seed equal 12345

units real
boundary p p p
atom_style full
timestep 0.5
neighbor 2.0 bin
neigh_modify delay 0 every 5 check yes

# ===== LOAD CONFIGURATION =====
# Note: Using read_data which expects a LAMMPS data file
# If frame_${i}.dump is a dump file, you might need to convert it first
read_data ${DATA_FILE}

# ===== FORCE FIELD =====
pair_style reaxff NULL checkqeq yes
pair_coeff * * ffield.reax Si O H

# ===== QEQ AND INITIAL RUN =====
fix 1 all qeq/reax 1 0.0 10.0 1.0e-3 reaxff
thermo 100
run 1000
dump production all custom 10 dump_init_${i}.lammpstrj id type x y z vx vy vz
dump_modify production format float %.16f

# ===== MINIMIZATION =====
minimize 1.0e-4 1.0e-6 1 10

# ===== VELOCITY INITIALIZATION =====
velocity all create \${t_lo} \${seed} rot yes dist gaussian

# ===== NVT EQUILIBRATION (100 ps) =====
fix 2 all nvt temp \${t_lo} \${t_lo} 100
run 100000

# ===== METADYNAMICS PRODUCTION (1 ns) =====
# Note: This runs WITH NVT thermostat (fix 2 remains active)
fix 3 all plumed plumedfile plumed_${i}.dat outfile plumed_${i}.log
run 200000

# ===== CLEAN UP METADYNAMICS =====
unfix 2
unfix 3

# ===== FINAL NVT EQUILIBRATION (50 ps) =====
fix 2 all nvt temp 300.0 300.0 100.0
run 100000

write_data      SiO2_metadynamics_output_${i}.data

print "Metadynamics simulation ${i} complete!"
print "Total steps:"
print "  - Initial NVT:  100,000 steps (50  ps)"
print "  - Metadynamics: 200,000 steps (100 ps)"
print "  - Final NVT:    100,000 steps (50  ps)"
print "Output: SiO2_metadynamics_output_${i}.data"
EOF

    # Run the simulation
    echo "Running metadynamics simulation ${i}..."
    echo "Command: lmp -sf omp -in meta_${i}.in > meta_${i}.out"
    lmp -sf omp -in meta_${i}.in > meta_${i}.out

    # Check if successful
    if [ $? -eq 0 ]; then
        echo "  ✓ LAMMPS execution completed for simulation ${i}"

        # Check for key output files
        if [ -f "COLVAR_METAD_${i}" ] && [ -f "HILLS_${i}" ]; then
            echo "  ✓ PLUMED output files generated"
            echo "    - COLVAR_METAD_${i} (CV values)"
            echo "    - HILLS_${i} (Gaussian hills)"
        else
            echo "    Warning: Some PLUMED output files missing"
        fi

        if [ -f "SiO2_metadynamics_output_${i}.data" ]; then
            echo "  ✓ Final structure saved"
        fi

        # Extract timing information if available
        if [ -f "meta_${i}.out" ]; then
            # Look for the final timing info
            echo "  Final stats from LAMMPS log:"
            tail -5 meta_${i}.out | grep -E "(Loop|Total)"
        fi
    else
        echo "  ✗ Error: LAMMPS failed for simulation ${i}"
        echo "  Check meta_${i}.out for error messages"
    fi
done

echo ""
echo "=========================================="
echo "Summary of all 5 metadynamics simulations:"
echo "=========================================="
echo ""
echo "For each simulation, the following protocol was followed:"
echo "1. Read initial configuration from frame_X.dump"
echo "2. Initial minimization (1.0e-4 1.0e-6 1 10)"
echo "3. Velocity initialization at 300K"
echo "4. NVT equilibration:       100,000 steps (50  ps)"
echo "5. Metadynamics production: 200,000 steps (100 ps) with NVT"
echo "6. Final NVT equilibration: 100,000 steps (50  ps)"
echo "7. Write final structure"
echo ""
echo "Generated files for each simulation (X=1..5):"
echo "  - meta_X.in: LAMMPS input script"
echo "  - meta_X.out: LAMMPS output log"
echo "  - plumed_X.dat: PLUMED input"
echo "  - plumed_X.log: PLUMED log"
echo "  - COLVAR_METAD_X: Collective variable trajectories"
echo "  - HILLS_X: Gaussian hills deposited"
echo "  - SiO2_metadynamics_output_X.data: Final structure"
echo "  - dump_init_X.lammpstrj: Initial trajectory (optional)"
echo "  - final_traj_X.lammpstrj: Final trajectory (optional)"
echo ""
echo "Program finished with exit code $? at: `date`"
