#!/bin/bash
#Slurm submission script for running a group of simulations in geoclaw in parallel

#### Resource Request: ####
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32 
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH -p normal_q
#SBATCH -A tsimp_slr
#load modules
module load parallel/20200522-GCCcore-9.3.0

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

export JOBS_PER_NODE=$(( $SLURM_CPUS_ON_NODE / $SLURM_CPUS_PER_TASK ))
export FFLAGS='-O2 -fopenmp'
export OMP_NUM_THREADS=32

#bash
#scontrol show hostname > ./node_list_${SLURM_JOB_ID}
#parallel --jobs 4 --delay 45 srun ./command.txt arg1:{1} ::: 'seq breach no_breach' ::: 'seq {480..490}'
#parallel -j$SLURM_NTASKS  < command.txt


# Tutorial
my_parallel="parallel --delay .25 -j $SLURM_NTASKS"
echo $SLURM_NTASKS
my_srun='srun --exclusive  --export=all --ntasks=1 --ntasks-per-node=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK'
$my_parallel "$my_srun python random_breach.py {1}"   ::: {26..200} #:::::: wd_last{0001..0050} #::: 1.5 1.0
exit;
