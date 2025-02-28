#!/bin/bash -x

#SBATCH -c 1                # Number of cores
#SBATCH -p shared,tambe,serial_requeue
#SBATCH -t 04:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem=8000          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o joblogs/%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e joblogs/%A_%a.err  # File to which STDERR will be written, %j inserts jobid

set -x

module load python/3.10.13-fasrc01
module load intel/24.0.1-fasrc01
module load openmpi/5.0.2-fasrc01
source activate llm

data="continuous_state"
save_string="synthetic"
N=21
B=15.0
n_train_epochs=901
seed=0
cdir="."
num_atoms=51
target_q_update_freq=10
noise_level=1.0
n_noisy_arms=17
no_comm_epochs=300
noise_shape=1

bash run/synthetic/run_synthetic.sh ${cdir} ${SLURM_ARRAY_TASK_ID} ${data} ${save_string} ${N} ${B} ${n_train_epochs} ${num_atoms} ${target_q_update_freq} ${noise_level} ${n_noisy_arms} ${no_comm_epochs} ${noise_shape}