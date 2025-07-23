#!/bin/bash

N_VALUES=(12000 14000 18000 20000)
P_VALUES=(1 2 4 8 16 32)

for N in "${N_VALUES[@]}"; do
    for P in "${P_VALUES[@]}"; do
        JOB_NAME="mm_${N}_P${P}"
        OUTPUT_FILE="runs/mm_par_${N}_P${P}_%j.out"
        ERROR_FILE="runs/mm_par_${N}_P${P}_%j.err"

        cat << EOF > "${JOB_NAME}.sh"
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -o ${OUTPUT_FILE}
#SBATCH -e ${ERROR_FILE}
#SBATCH --mail-user fmazzarotto+capri@proton.me
#SBATCH --mail-type END
#SBATCH -n ${P}
#SBATCH -p allgroups
#SBATCH -t 01:00:00
#SBATCH --mem 15G

cd \$SLURM_SUBMIT_DIR
mpirun -np \$SLURM_NTASKS ./a.out ${N} ${N} ${N}
EOF

        sbatch "${JOB_NAME}.sh"
        rm -f "${JOB_NAME}.sh"
    done
done

