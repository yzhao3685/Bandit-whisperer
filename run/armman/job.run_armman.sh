data="armman"
save_string="armman"
N=50
B=20.0
n_train_epochs=201
seed=0
cdir="."
num_atoms=51
target_q_update_freq=10
noise_level=1
n_noisy_arms=38
no_comm_epochs=20
noise_shape=1

bash run/armman/run_armman.sh ${cdir} ${seed} ${data} ${save_string} ${N} ${B} ${n_train_epochs} ${num_atoms} ${target_q_update_freq} ${noise_level} ${n_noisy_arms} ${no_comm_epochs} ${noise_shape}