data="sis"
save_string="sis"
N=20
B=16.0
n_train_epochs=201
seed=0
cdir="."
pop_size=50
num_atoms=51
target_q_update_freq=10
noise_level=1.0
n_noisy_arms=10
no_comm_epochs=20
noise_shape=1

bash run/sis/run_sis.sh ${cdir} ${seed} ${data} ${save_string} ${N} ${B} ${n_train_epochs} ${pop_size} ${num_atoms} ${target_q_update_freq} ${noise_level} ${n_noisy_arms} ${no_comm_epochs} ${noise_shape}

