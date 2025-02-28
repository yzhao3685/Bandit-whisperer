
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import torch
from torch.optim import Adam, SGD
import time
import robust_rmab.algos.rmabppo.rmabppo_core as core
from robust_rmab.environments.bandit_env_robust import ARMMANRobustEnv, SISRobustEnv, ContinuousStateExampleEnv, CounterExampleEnv
import copy
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn import metrics
import csv
import pandas as pd
import matplotlib.pyplot as plt

class VDN_buffer:
    def __init__(self, n_epochs, N):
        self.ptr = 0
        self.influence_buffer = np.zeros(n_epochs)
        self.comm_strat_buffer = np.zeros((n_epochs, N))

    def store(self, influence=0, comm_strat=[0]):
        self.influence_buffer[self.ptr] = influence
        self.comm_strat_buffer[self.ptr] = comm_strat
        self.ptr += 1

class RMABPPO_Buffer:
    """
    A buffer for storing trajectories experienced by a RMABPPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, tp_feats, act_dim, N, act_type, steps_per_epoch, one_hot_encode=True, gamma=0.99,
                 n_epochs=200, indices_to_perturb=[], get_buffer_size=4000):
        self.N = N
        self.obs_dim = obs_dim
        self.one_hot_encode = one_hot_encode
        self.size = steps_per_epoch * n_epochs * 11 # to accommodate extra data for comm
        self.features_buf = np.zeros(core.combined_shape(self.size, (N, tp_feats)), dtype=np.float32)
        self.opt_in_buf = np.ones(core.combined_shape(self.size, N), dtype=np.float32)
        self.obs_buf = np.zeros(core.combined_shape(self.size, N), dtype=np.float32)
        self.ohs_buf = np.zeros(core.combined_shape(self.size, (N, obs_dim)), dtype=np.float32)
        self.act_buf = np.zeros((self.size, N), dtype=np.float32)
        self.oha_buf = np.zeros(core.combined_shape(self.size, (N, act_dim)), dtype=np.float32)
        self.rew_buf = np.zeros((self.size,N), dtype=np.float32)
        self.gamma = gamma
        self.ptr = 0
        self.steps_per_epoch = steps_per_epoch
        self.act_type = act_type
        self.act_dim = act_dim
        self.indices_to_perturb = indices_to_perturb
        self.get_buffer_size = get_buffer_size


    def store(self, obs, features, opt_in, act, rew):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        ohs = np.zeros((self.N, self.obs_dim))
        if self.one_hot_encode:
            for i in range(self.N):
                ohs[i, int(obs[i])] = 1
        self.ohs_buf[self.ptr] = ohs
        self.features_buf[self.ptr] = features
        self.opt_in_buf[self.ptr] = opt_in

        self.act_buf[self.ptr] = act
        oha = np.zeros((self.N, self.act_dim))
        for i in range(self.N):
            oha[i, int(act[i])] = 1
        self.oha_buf[self.ptr] = oha

        self.rew_buf[self.ptr] = rew
        self.ptr += 1

    def get(self, num_batch=-1):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        start_idx = max(0, self.ptr - self.get_buffer_size)
        cum_rew = np.zeros(self.N)
        if num_batch > 0:
            cum_rew = np.zeros((num_batch, self.N))
            for j in range(num_batch):  # reward more than 20 steps ago will not matter in cumsum
                for arm_idx in range(self.N):
                    rews = self.rew_buf[self.ptr - (j+1) * self.steps_per_epoch : self.ptr - j * self.steps_per_epoch, arm_idx]
                    cum_rew[j, arm_idx] = sum(rews) #core.discount_cumsum(rews, self.gamma)[0]

        data = dict(obs=self.obs_buf[start_idx:self.ptr, :],
                    act=self.act_buf[start_idx:self.ptr, :],
                    oha=self.oha_buf[start_idx:self.ptr, :],
                    ohs=self.ohs_buf[start_idx:self.ptr, :],
                    rews=self.rew_buf[start_idx:self.ptr, :],
                    cum_rew=cum_rew)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

class AgentOracle:

    def __init__(self, data, N, S, A, B, seed, REWARD_BOUND, agent_kwargs=dict(),
        home_dir="", exp_name="", sampled_nature_parameter_ranges=None, robust_keyword="",
        pop_size=0, one_hot_encode=True, non_ohe_obs_dim=None, state_norm=None, opt_in_rate=None,
                 target_q_update_freq=10, num_atoms=21):

        self.data = data
        self.home_dir = home_dir
        self.exp_name = exp_name
        self.REWARD_BOUND = REWARD_BOUND
        self.N = N
        self.S = S
        self.A = A
        self.B = B
        self.seed=seed
        self.sampled_nature_parameter_ranges = sampled_nature_parameter_ranges
        self.robust_keyword = robust_keyword
        self.opt_in_rate = opt_in_rate
        self.target_q_update_freq = target_q_update_freq
        self.num_atoms = num_atoms
        self.num_neighbors = 1 # out of N arms, how many should we consider as neighbors and have a communication channel

        self.pop_size = pop_size
        self.one_hot_encode = one_hot_encode
        self.non_ohe_obs_dim = non_ohe_obs_dim
        self.state_norm = state_norm
        self.noise_level = agent_kwargs['noise_level']
        self.noise_shape = agent_kwargs['noise_shape']
        self.n_noisy_arms = agent_kwargs['n_noisy_arms']
        self.no_comm_epochs = agent_kwargs['no_comm_epochs']

        if data == 'armman':
            self.env_fn = lambda : ARMMANRobustEnv(N,B,seed, agent_kwargs['noise_level'], agent_kwargs['noise_shape'])

        if data == 'sis':
            self.env_fn = lambda : SISRobustEnv(N,B,pop_size,seed, agent_kwargs['noise_level'], agent_kwargs['noise_shape'])

        if data == 'continuous_state':
            self.env_fn = lambda: ContinuousStateExampleEnv(N, B, seed, agent_kwargs['noise_level'], agent_kwargs['noise_shape'])

        if data == 'counterexample':
            self.env_fn = lambda: CounterExampleEnv(N, B, seed, agent_kwargs['noise_level'], agent_kwargs['noise_shape'])

        self.actor_critic=core.MLPActorCriticRMAB
        self.agent_kwargs=agent_kwargs

        self.strat_ind = 0

        # this won't work if we go back to MPI, but doing it now to simplify seeding
        self.env = self.env_fn()
        self.env.seed(seed)
        # self.env.sampled_parameter_ranges = self.sampled_nature_parameter_ranges


    # Todo - figure out parallelization with MPI -- not clear how to do this yet, so restrict to single cpu
    def best_response(self, nature_strats, nature_eq, add_to_seed):

        self.strat_ind += 1
        exp_name = '%s_n%ib%.1fd%sp%s'%(self.exp_name, self.N, self.B, self.data, self.pop_size)
        data_dir = os.path.join(self.home_dir, 'data')

        return self.best_response_per_cpu(nature_strats, nature_eq, add_to_seed, seed=self.seed, **self.agent_kwargs)

    # add_to_seed is obsolete
    def best_response_per_cpu(self, nature_strats, nature_eq, add_to_seed, actor_critic=core.MLPActorCriticRMAB, ac_kwargs=dict(), seed=0, 
            steps_per_epoch=4000, epochs=50, gamma=0.99, vdn_lr=2e-3,
            vf_lr=1e-3, train_v_iters=80,
            tp_transform=None,
            noise_level=1,
            noise_shape=1,
            n_noisy_arms=10,
            no_comm_epochs=200):

        # Instantiate environment
        env = self.env

        # env.sampled_parameter_ranges = self.sampled_nature_parameter_ranges
        obs_dim = env.observation_space.shape

        # set input dim size given transformation selected 
        # ac_kwargs["input_feat_dim"] refers to how many features generated from input tps
        if tp_transform is None or tp_transform=="None" or ac_kwargs["input_feat_dim"]==None: 
            print("Defaulting to ground truth transition prob. inputs")
            ac_kwargs["input_feat_dim"] = 4
            tp_transform = None 
        else: 
            print("[tp->feats] Applying {} with dim {}".format(tp_transform,ac_kwargs["input_feat_dim"]))

        # Create actor-critic module
        ac = actor_critic(env.observation_space, env.action_space, opt_in_rate=self.opt_in_rate,
            N = env.N, C = env.C, B = env.B, strat_ind=self.strat_ind,
            one_hot_encode = self.one_hot_encode, non_ohe_obs_dim = self.non_ohe_obs_dim,
            state_norm=self.state_norm, seed=seed, num_atoms=self.num_atoms, num_neighbors=self.num_neighbors,
            **ac_kwargs)

        ac.max_q = env.max_reward / (1 - gamma)  # properly update ac

        act_dim = ac.act_dim
        obs_dim = ac.obs_dim
        num_neighbors = self.num_neighbors


        qf_optimizers_topline = np.zeros(env.N, dtype=object)
        qf_optimizers_oracle = np.zeros(env.N, dtype=object)
        qf_optimizers_worst = np.zeros(env.N, dtype=object)
        qf_optimizers_random = np.zeros(env.N, dtype=object)
        qf_optimizers_control = np.zeros(env.N, dtype=object)
        qf_optimizers_comm = np.zeros(env.N, dtype=object)
        vdn_parameters = set()
        for arm_idx in range(env.N):
            vdn_parameters |= set(ac.vdn_net.q_net[arm_idx].parameters())
        vdn_optimizer = Adam(vdn_parameters, lr=vdn_lr)
        for i in range(env.N):
            qf_optimizers_oracle[i] = Adam(ac.q_list_oracle[i].parameters(), lr=vf_lr)
            qf_optimizers_worst[i] = Adam(ac.q_list_worst[i].parameters(), lr=vf_lr)
            qf_optimizers_random[i] = Adam(ac.q_list_random[i].parameters(), lr=vf_lr)
            qf_optimizers_topline[i] = Adam(ac.q_list_topline[i].parameters(), lr=vf_lr)
            qf_optimizers_control[i] = Adam(ac.q_list_control[i].parameters(), lr=vf_lr)
            qf_optimizers_comm[i] = Adam(ac.q_list_comm[i].parameters(), lr=vf_lr)


        def featurize_tp(transition_probs, transformation=None, out_dim=4, in_dim=4):
            N = transition_probs.shape[0]
            output_features = np.zeros((N, out_dim))
            np.random.seed(0)  # Set random seed for reproducibility
            if transformation == "linear":
                transformation_matrix = np.random.rand(in_dim, out_dim)
                output_features = np.dot(transition_probs, transformation_matrix)
            elif transformation == "nonlinear":
                transformation_matrix = np.random.rand(in_dim, out_dim)
                output_features = 1 / (1 + np.exp(-np.dot(transition_probs, transformation_matrix)))
            else:
                output_features[:, :min(in_dim, out_dim)] = transition_probs[:, :min(in_dim, out_dim)]
            return output_features


        def compute_loss_vdn(vdn_buffer):
            vdn_optimizer.zero_grad()
            num_data_pts = vdn_buffer.ptr
            influence_predictions = torch.zeros(num_data_pts)
            influence_observed = torch.from_numpy(vdn_buffer.influence_buffer[:num_data_pts])
            for i in range(num_data_pts):
                influence_predictions[i] = ac.vdn_net(torch.from_numpy(vdn_buffer.comm_strat_buffer[i]).float())
            vdn_loss = torch.pow(influence_predictions - influence_observed, 2).mean()
            vdn_loss.backward()
            vdn_optimizer.step()


        def compute_loss_q(data, target_q_list_choice, q_list_choice, optimizer_choice,
                           receiver_arms=[], sender_arms=[]):
            ohs, oha, obs, rews = data['ohs'], data['oha'], data['obs'], data['rews']
            full_obs = None
            if ac.one_hot_encode:
                full_obs = ohs
            else:
                obs = obs/self.state_norm
                obs = obs.reshape(obs.shape[0], obs.shape[1], 1)
                full_obs = obs
            softmax_layer = torch.nn.Softmax(dim=2)
            enc = OneHotEncoder()
            enc.fit_transform(np.arange(ac.act_dim).reshape((-1, 1))) # tell encoder what are the possible actions
            loss_list = np.zeros(env.N,dtype=object)

            batch_size = data['rews'].shape[0] - 1
            num_atoms = ac.num_atoms
            max_return = env.max_reward
            max_q = ac.max_q
            min_q = 0 # artificially increasing this min_q does not improve performance
            delta_z = (max_q - min_q) / (num_atoms - 1)# interval length of each bin in categorical DQN
            for i in range(env.N):
                optimizer_choice[i].zero_grad()
                x = torch.as_tensor(full_obs[:-1, i], dtype=torch.float32)
                distribution_predictions = (q_list_choice[i](x)).reshape((batch_size, ac.act_dim, ac.num_atoms))
                distribution_predictions = softmax_layer(distribution_predictions)
                act_idx = np.argmax(oha[:-1,i], axis=1)
                distribution_predictions = distribution_predictions[torch.arange(distribution_predictions.size(0)), act_idx]
                # projection step
                with torch.no_grad(): # do not compute gradient for this target_q
                    target_x = torch.as_tensor(full_obs[1:, i], dtype=torch.float32)
                    target_distribution = target_q_list_choice[i](target_x).reshape((batch_size, ac.act_dim, ac.num_atoms))
                    target_distribution = softmax_layer(target_distribution)
                    target_q_all = np.dot(target_distribution, np.arange(num_atoms) * delta_z + min_q) # map categories into q-values
                    act_idx = np.argmax(target_q_all, axis=1)
                    target_distribution = target_distribution[torch.arange(target_distribution.size(0)), act_idx]
                    support = torch.linspace(min_q, max_q, num_atoms)
                    rewards = rews[:-1, i].unsqueeze(1).expand_as(target_distribution)
                    support = support.unsqueeze(0).expand_as(target_distribution)
                    Tz = rewards + gamma * support
                    Tz += 0.0002 # integer valued Tz will cause proj_dist to not sum up to 1. add this to avoid entires that are zeros
                    Tz = Tz.clamp(min=min_q, max=max_q)
                    Tz -= 0.0001 # integer valued Tz will cause proj_dist to not sum up to 1. add this to avoid entries that are max_q
                    b_arr = (Tz - min_q) / delta_z
                    l_arr = b_arr.floor().long()
                    u_arr = b_arr.ceil().long()
                    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long().unsqueeze(1).expand(batch_size, num_atoms) # added to index to make sure we update the correct entry

                    proj_dist = torch.zeros(target_distribution.size())
                    proj_dist.view(-1).index_add_(0, (l_arr + offset).view(-1), (target_distribution * (u_arr.float() - b_arr)).view(-1))
                    proj_dist.view(-1).index_add_(0, (u_arr + offset).view(-1), (target_distribution * (b_arr - l_arr.float())).view(-1))

                # compute NCE loss
                distribution_predictions.clamp(min=0.001, max=0.999) # avoid log(0)
                loss_list[i] = - (proj_dist * distribution_predictions.log()).sum(1).mean()
                loss_list[i].backward()
                optimizer_choice[i].step()
            return loss_list

        with torch.no_grad():
            if self.data == 'armman':
                # env.T is already initialized for real world data. don't update_transition_prob
                env_parameters = env.features
                ac.feature_arr = featurize_tp(env_parameters, transformation=tp_transform,
                                              out_dim=ac_kwargs["input_feat_dim"],
                                              in_dim=4)
            elif self.data == 'sis':
                env.update_transition_probs(np.ones(env.N))  # initialize all transition probs
                T_matrix = env.param_setting  # for SIS env, 4 parameters encode the transition dynamics information
                ac.feature_arr = featurize_tp(T_matrix, transformation=tp_transform,
                                              out_dim=ac_kwargs["input_feat_dim"],
                                              in_dim=4)
            elif self.data == 'counterexample':
                env.update_transition_probs(np.ones(env.N))  # initialize all transition probs
                ac.feature_arr = featurize_tp(env.T, transformation=tp_transform, out_dim=ac_kwargs["input_feat_dim"],
                                          in_dim=2)
            else:
                env.update_transition_probs(np.ones(env.N))  # initialize all transition probs
                T_matrix = env.T[:,:,:,1]
                T_matrix = np.reshape(T_matrix, (T_matrix.shape[0], np.prod(T_matrix.shape[1:])))
                ac.feature_arr = featurize_tp(T_matrix, transformation=tp_transform, out_dim=ac_kwargs["input_feat_dim"],
                                          in_dim=4)


        # prepare for model saving and simulator
        ac.out_dim = ac_kwargs["input_feat_dim"]
        ac.env = env

        # preparation for learning from similar arm's information
        feature_distances = metrics.pairwise.pairwise_distances(ac.feature_arr, ac.feature_arr)
        feature_distances += np.diag(np.ones(env.N) * 100) # make sure the same arm is not selected
        # indices_to_perturb = np.arange(round(env.N / 2), env.N)
        indices_to_perturb = np.arange(N - self.n_noisy_arms, env.N)
        indices_no_noise = list(set(np.arange(env.N)) - set(indices_to_perturb))
        communication_channels = np.argsort(feature_distances, axis=1)
        nearest_neighbor_noise_free = [0] * env.N
        # nearest_neighbor_noisy = [0] * env.N
        nearest_neighbor_sender = [0] * env.N
        random_neighbor = [0] * env.N
        for receiver in range(env.N):
            min_dist_noise_free = np.inf
            min_dist_any = np.inf
            for sender in indices_no_noise: # sender arms are noise-free.
                if feature_distances[receiver][sender] < min_dist_noise_free:
                    min_dist_noise_free = feature_distances[receiver][sender]
                    nearest_neighbor_noise_free[receiver] = sender
            for sender in indices_to_perturb: # sender arms are noisy
                if feature_distances[receiver][sender] < min_dist_any:
                    min_dist_any = feature_distances[receiver][sender]
                    nearest_neighbor_sender[receiver] = sender
            neighbors_set = np.arange(env.N).tolist()
            neighbors_set.remove(receiver)
            random_neighbor[receiver] = np.random.choice(neighbors_set)
        # initialize epsilon-greedy for vdn
        init_epochs = int(epochs / 2) # start with exploring different communication strategies
        no_comm_epochs = self.no_comm_epochs
        vdn_epsilon = 0.3 # 0.15 for ce
        use_random_communication = ac.random_stream.binomial(n=1, p=vdn_epsilon * np.ones(epochs - init_epochs))
        # use_random_communication = ac.random_stream.binomial(n=1, p=vdn_epsilon *
        #                                               np.power(vdn_epsilon_decay,np.arange(epochs - init_epochs)))
        use_random_communication = np.append(np.ones(init_epochs), use_random_communication)
        comm_strat = []; influence = -1; learnt_comm_receiver_arms = []; learnt_comm_sender_arms = []
        vdn_buffer = VDN_buffer(epochs, env.N)

        # Set up experience buffer
        self.get_buffer_size = 12000
        if self.data == 'sis':
            self.get_buffer_size = 4000 # sis gets too slow if we use buffer_size=12000
        if self.data == 'counterexample':
            self.get_buffer_size = 12000
        local_steps_per_epoch = int(steps_per_epoch)
        buf_topline = RMABPPO_Buffer(obs_dim, ac_kwargs["input_feat_dim"], act_dim, env.N, ac.act_type, local_steps_per_epoch,
                                    one_hot_encode=self.one_hot_encode, gamma=gamma, n_epochs=epochs,
                                     indices_to_perturb=indices_to_perturb, get_buffer_size=self.get_buffer_size)
        buf_comm = copy.deepcopy(buf_topline)
        buf_new_comm = copy.deepcopy(buf_topline)
        buf_control = copy.deepcopy(buf_topline)
        buf_oracle_comm = copy.deepcopy(buf_topline)
        buf_worst_comm = copy.deepcopy(buf_topline)
        buf_random_comm = copy.deepcopy(buf_topline)
        # placeholder
        data_control = []
        data_comm = []
        fixed_use_behavioral = False

        def compute_comm(rand_comm=0):
            if rand_comm > 0:  # epsilon-greedy
                comm_strat = ac.random_stream.binomial(n=1, p=0.5, size=env.N)
            else:
                with torch.no_grad():  # comm_strat[i,j]==1 means arm i should get info from arm j
                    comm_strat = (ac.vdn_net.compute_best_action()).numpy()
            learnt_comm_receiver_arms = np.where(comm_strat > 0)[0]
            # learnt_comm_receiver_arms = indices_to_perturb # sanity check: what if we give the ground truth info
            learnt_comm_sender_arms = []  # important! reset before filling in
            for arm_idx in learnt_comm_receiver_arms:  # arms receiving information
                senders_ = np.setdiff1d(communication_channels[arm_idx], learnt_comm_receiver_arms,assume_unique=True)
                # other receivers should not send
                senders_ = senders_[:self.num_neighbors]  # pick similar arms to send
                learnt_comm_sender_arms.append(senders_)
            learnt_comm_sender_arms = np.array(learnt_comm_sender_arms).flatten()
            return learnt_comm_sender_arms, learnt_comm_receiver_arms, comm_strat

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):
            print('epoch ', epoch)
            # when computing influence, use more data
            comm_steps_to_run = local_steps_per_epoch
            inflence_epochs = 1
            if epoch % self.target_q_update_freq == 1 and epoch > no_comm_epochs + 1:
                comm_steps_to_run *= inflence_epochs

            # epsilon greedy policy
            ac.epsilon *= 0.999
            take_random_actions = ac.random_stream.binomial(n=1, p=ac.epsilon, size=comm_steps_to_run)

            # with communication
            o = env.reset()
            o = o.reshape(-1)
            for t in range(comm_steps_to_run):
                torch_o = torch.as_tensor(o, dtype=torch.float32)
                if influence < 0 or epoch < no_comm_epochs + 1 or epoch % self.target_q_update_freq == 1:
                    a_agent = ac.step(torch_o, ac.q_list_comm, take_random_actions=take_random_actions[t])
                else:
                    a_agent = ac.step(torch_o, ac.q_list_new_comm, take_random_actions=take_random_actions[t])
                next_o, r, d, _ = env.step(a_agent, ac.opt_in, perturb_reward=True, indices_to_perturb=indices_to_perturb)
                next_o = next_o.reshape(-1)
                buf_comm.store(o, ac.feature_arr, ac.opt_in, a_agent, r)
                o = next_o # Update obs (critical!)
            data_comm = copy.deepcopy(buf_comm.get())
            if epoch % self.target_q_update_freq == 1 and epoch > 1:
                data_comm = copy.deepcopy(buf_comm.get(num_batch=inflence_epochs))
            reward_current_comm_strat = data_comm['cum_rew'] # discount cumulative rewards when there is no communication

            # target-q update
            if epoch % self.target_q_update_freq == 0:
                with torch.no_grad():
                    for arm_idx in range(env.N):
                        ac.target_q_list_comm[arm_idx].q_net.load_state_dict(ac.q_list_comm[arm_idx].q_net.state_dict())

                if epoch > no_comm_epochs:
                    learnt_comm_sender_arms, learnt_comm_receiver_arms, comm_strat = compute_comm(rand_comm=use_random_communication[epoch])

            # compute influence
            if epoch % self.target_q_update_freq == 1 and epoch > no_comm_epochs + 1:
                # collect data using networks with new communication strategy
                o = env.reset()
                o = o.reshape(-1)
                for t in range(comm_steps_to_run):
                    torch_o = torch.as_tensor(o, dtype=torch.float32)
                    a_agent = ac.step(torch_o, ac.q_list_new_comm, take_random_actions=take_random_actions[t])
                    next_o, r, d, _ = env.step(a_agent, ac.opt_in, perturb_reward=True, indices_to_perturb=indices_to_perturb)
                    next_o = next_o.reshape(-1)
                    buf_new_comm.store(o, ac.feature_arr, ac.opt_in, a_agent, r)
                    o = next_o  # Update obs (critical!)
                reward_new_comm_strat = buf_new_comm.get(num_batch=inflence_epochs)['cum_rew']
                influence_arr = (torch.sum(reward_new_comm_strat - reward_current_comm_strat, axis=1)).numpy()

                # perc of correct communications
                counter_correct_comm = 0
                sender_arms, receiver_arms, _ = compute_comm(rand_comm=0)
                for arm_idx in range(len(receiver_arms)):  # arms receiving information
                    if sender_arms[arm_idx] not in indices_to_perturb and receiver_arms[arm_idx] in indices_to_perturb:
                        counter_correct_comm += 1
                perc_correct_comm = counter_correct_comm / len(receiver_arms)
                write_list = list(influence_arr) + [counter_correct_comm, perc_correct_comm]
                newpath = os.getcwd() + '/logs/results/%s_n%s_b%s_S%s_e%s_noisy%s' % (
                    self.exp_name, env.N, int(env.B), self.pop_size, self.noise_level, self.n_noisy_arms)
                if not os.path.exists(newpath):
                    time.sleep(self.seed / 10) # avoid that multiple seeds creating folders at the same time
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                fname = os.getcwd() + '/logs/results/%s_n%s_b%s_S%s_e%s_noisy%s/influence_seed%s.csv' % (
                    self.exp_name, env.N, int(env.B), self.pop_size, self.noise_level, self.n_noisy_arms, self.seed)
                if os.path.exists(fname):
                    with open(fname, 'a') as aggregate_file:
                        writer = csv.writer(aggregate_file)
                        writer.writerow(write_list)
                        aggregate_file.flush()
                else:
                    with open(fname, 'w') as aggregate_file:
                        writer = csv.writer(aggregate_file)
                        writer.writerow(write_list)
                        aggregate_file.flush()
                influence = np.average(influence_arr)
                print('influence ', influence)
                vdn_buffer.store(influence=influence, comm_strat=comm_strat)
                if vdn_buffer.ptr >= 2:
                    compute_loss_vdn(vdn_buffer)

            # update networks
            for v_iter in range(train_v_iters):
                loss_q_list = compute_loss_q(data_comm,
                                             target_q_list_choice=ac.target_q_list_comm,
                                             q_list_choice=ac.q_list_comm,
                                             optimizer_choice=qf_optimizers_comm)
            if epoch > no_comm_epochs:
                with torch.no_grad():
                    for arm_idx in range(env.N):  # arms receiving information
                        ac.q_list_new_comm[arm_idx].q_net.load_state_dict(ac.q_list_comm[arm_idx].q_net.state_dict())
                        if arm_idx in learnt_comm_receiver_arms:
                            sender_arm = learnt_comm_sender_arms[list(learnt_comm_receiver_arms).index(arm_idx)]
                            ac.q_list_new_comm[arm_idx].q_net.load_state_dict(ac.q_list_comm[sender_arm].q_net.state_dict())
            ##########################################################################################################
            # control group - no communication throughout the training
            if epoch == no_comm_epochs:
                buf_control = copy.deepcopy(buf_comm)
                with torch.no_grad():
                    for arm_idx in range(env.N):  # arms receiving information
                        qf_optimizers_control[arm_idx].load_state_dict(qf_optimizers_comm[arm_idx].state_dict())
                        ac.q_list_control[arm_idx].q_net.load_state_dict(ac.q_list_comm[arm_idx].q_net.state_dict())
                        ac.target_q_list_control[arm_idx].q_net.load_state_dict(ac.q_list_comm[arm_idx].q_net.state_dict())
            if epoch > no_comm_epochs:
                o = env.reset()  # crucial! else 'no communication' starts with bad states
                o = o.reshape(-1)
                for t in range(local_steps_per_epoch):
                    torch_o = torch.as_tensor(o, dtype=torch.float32)
                    a_agent = ac.step(torch_o, ac.q_list_control, take_random_actions=take_random_actions[t])
                    next_o, r, d, _ = env.step(a_agent, ac.opt_in, perturb_reward=True, indices_to_perturb=indices_to_perturb)
                    next_o = next_o.reshape(-1)
                    buf_control.store(o, ac.feature_arr, ac.opt_in, a_agent, r)
                    o = next_o # Update obs (critical!)
                data_control = copy.deepcopy(buf_control.get())
                with torch.no_grad():
                    if epoch % self.target_q_update_freq == 0:
                        for arm_idx in range(env.N):
                            ac.target_q_list_control[arm_idx].q_net = copy.deepcopy(ac.q_list_control[arm_idx].q_net)
                for i in range(train_v_iters):
                    loss_q_list = compute_loss_q(data_control, target_q_list_choice=ac.target_q_list_control,
                                                 q_list_choice=ac.q_list_control, optimizer_choice=qf_optimizers_control)

            ##########################################################################################################
            # oracle communication - use ground truth info on which arms are noisy
            if epoch == no_comm_epochs:
                buf_oracle_comm = copy.deepcopy(buf_comm)
                with torch.no_grad():
                    for arm_idx in range(env.N):  # arms receiving information
                        qf_optimizers_oracle[arm_idx].load_state_dict(qf_optimizers_comm[arm_idx].state_dict())
                        ac.q_list_oracle[arm_idx].q_net.load_state_dict(ac.q_list_comm[arm_idx].q_net.state_dict())
                        ac.target_q_list_oracle[arm_idx].q_net.load_state_dict(ac.q_list_comm[arm_idx].q_net.state_dict())
                        ac.q_list_oracle_beh[arm_idx].q_net.load_state_dict(ac.q_list_comm[arm_idx].q_net.state_dict())

            if epoch > no_comm_epochs:
                o = env.reset()
                o = o.reshape(-1)
                for t in range(local_steps_per_epoch):
                    torch_o = torch.as_tensor(o, dtype=torch.float32)
                    if fixed_use_behavioral:
                        a_agent = ac.step(torch_o, ac.q_list_oracle_beh, take_random_actions=take_random_actions[t])
                    else:
                        a_agent = ac.step(torch_o, ac.q_list_oracle, take_random_actions=take_random_actions[t])
                    next_o, r, d, _ = env.step(a_agent, ac.opt_in, perturb_reward=True, indices_to_perturb=indices_to_perturb)
                    next_o = next_o.reshape(-1)
                    buf_oracle_comm.store(o, ac.feature_arr, ac.opt_in, a_agent, r)
                    o = next_o # Update obs (critical!)
                data_oracle = copy.deepcopy(buf_oracle_comm.get())
                with torch.no_grad():
                    if epoch % self.target_q_update_freq == 0:
                        for arm_idx in range(env.N):
                            ac.target_q_list_oracle[arm_idx].q_net = copy.deepcopy(ac.q_list_oracle[arm_idx].q_net)

                for i in range(train_v_iters):
                    loss_q_list = compute_loss_q(data_oracle, target_q_list_choice=ac.target_q_list_oracle,
                                                 q_list_choice=ac.q_list_oracle, optimizer_choice=qf_optimizers_oracle)

                with torch.no_grad():
                    for arm_idx in range(env.N):
                        ac.q_list_oracle_beh[arm_idx].q_net = copy.deepcopy(ac.q_list_oracle[arm_idx].q_net)
                        if epoch > no_comm_epochs and arm_idx in indices_to_perturb:  # arms receiving information
                            sender_arm = nearest_neighbor_noise_free[arm_idx]
                            if fixed_use_behavioral:
                                ac.q_list_oracle_beh[arm_idx].q_net = copy.deepcopy(ac.q_list_oracle[sender_arm].q_net)
                            else:
                                ac.q_list_oracle[arm_idx].q_net = copy.deepcopy(ac.q_list_oracle[sender_arm].q_net)

            ##########################################################################################################
            # feature-based comm. send from nearest neighbor regardless of the sender is noisy or not
            if epoch == no_comm_epochs:
                buf_worst_comm = copy.deepcopy(buf_comm)
                with torch.no_grad():
                    for arm_idx in range(env.N):  # arms receiving information
                        qf_optimizers_worst[arm_idx].load_state_dict(qf_optimizers_comm[arm_idx].state_dict())
                        ac.q_list_worst[arm_idx].q_net.load_state_dict(ac.q_list_comm[arm_idx].q_net.state_dict())
                        ac.target_q_list_worst[arm_idx].q_net.load_state_dict(ac.q_list_comm[arm_idx].q_net.state_dict())
                        ac.q_list_worst_beh[arm_idx].q_net.load_state_dict(ac.q_list_comm[arm_idx].q_net.state_dict())

            if epoch > no_comm_epochs:
                o = env.reset()
                o = o.reshape(-1)
                for t in range(local_steps_per_epoch):
                    torch_o = torch.as_tensor(o, dtype=torch.float32)
                    if fixed_use_behavioral:
                        a_agent = ac.step(torch_o, ac.q_list_worst_beh, take_random_actions=take_random_actions[t])
                    else:
                        a_agent = ac.step(torch_o, ac.q_list_worst, take_random_actions=take_random_actions[t])
                    next_o, r, d, _ = env.step(a_agent, ac.opt_in, perturb_reward=True, indices_to_perturb=indices_to_perturb)
                    next_o = next_o.reshape(-1)
                    buf_worst_comm.store(o, ac.feature_arr, ac.opt_in, a_agent, r)
                    o = next_o # Update obs (critical!)
                data_worst_comm = copy.deepcopy(buf_worst_comm.get())
                with torch.no_grad():
                    if epoch % self.target_q_update_freq == 0:
                        for arm_idx in range(env.N):
                            ac.target_q_list_worst[arm_idx].q_net = copy.deepcopy(ac.q_list_worst[arm_idx].q_net)

                for i in range(train_v_iters):
                    loss_q_list = compute_loss_q(data_worst_comm, target_q_list_choice=ac.target_q_list_worst,
                                                 q_list_choice=ac.q_list_worst, optimizer_choice=qf_optimizers_worst)

                with torch.no_grad():
                    for arm_idx in range(env.N):
                        ac.q_list_worst_beh[arm_idx].q_net = copy.deepcopy(ac.q_list_worst[arm_idx].q_net)
                        if epoch > no_comm_epochs and arm_idx in indices_no_noise:  # arms receiving information
                            sender_arm = nearest_neighbor_sender[arm_idx]
                            if fixed_use_behavioral:
                                ac.q_list_worst_beh[arm_idx].q_net = copy.deepcopy(ac.q_list_worst[sender_arm].q_net)
                            else:
                                ac.q_list_worst[arm_idx].q_net = copy.deepcopy(ac.q_list_worst[sender_arm].q_net)

            ##########################################################################################################
            # random communication - fix a random comm pattern
            if epoch == no_comm_epochs:
                buf_random_comm = copy.deepcopy(buf_comm)
                with torch.no_grad():
                    for arm_idx in range(env.N):  # arms receiving information
                        qf_optimizers_random[arm_idx].load_state_dict(qf_optimizers_comm[arm_idx].state_dict())
                        ac.q_list_random[arm_idx].q_net.load_state_dict(ac.q_list_comm[arm_idx].q_net.state_dict())
                        ac.target_q_list_random[arm_idx].q_net.load_state_dict(ac.q_list_comm[arm_idx].q_net.state_dict())
                        ac.q_list_random_beh[arm_idx].q_net.load_state_dict(ac.q_list_comm[arm_idx].q_net.state_dict())

            if epoch > no_comm_epochs:
                o = env.reset()
                o = o.reshape(-1)
                for t in range(local_steps_per_epoch):
                    torch_o = torch.as_tensor(o, dtype=torch.float32)
                    if fixed_use_behavioral:
                        a_agent = ac.step(torch_o, ac.q_list_random_beh, take_random_actions=take_random_actions[t])
                    else:
                        a_agent = ac.step(torch_o, ac.q_list_random, take_random_actions=take_random_actions[t])
                    next_o, r, d, _ = env.step(a_agent, ac.opt_in, perturb_reward=True, indices_to_perturb=indices_to_perturb)
                    next_o = next_o.reshape(-1)
                    buf_random_comm.store(o, ac.feature_arr, ac.opt_in, a_agent, r)
                    o = next_o # Update obs (critical!)
                data_random_comm = copy.deepcopy(buf_random_comm.get())
                with torch.no_grad():
                    if epoch % self.target_q_update_freq == 0:
                        for arm_idx in range(env.N):
                            ac.target_q_list_random[arm_idx].q_net = copy.deepcopy(ac.q_list_random[arm_idx].q_net)

                for i in range(train_v_iters):
                    loss_q_list = compute_loss_q(data_random_comm, target_q_list_choice=ac.target_q_list_random,
                                                 q_list_choice=ac.q_list_random, optimizer_choice=qf_optimizers_random)

                with torch.no_grad():
                    for arm_idx in range(env.N):
                        ac.q_list_random_beh[arm_idx].q_net = copy.deepcopy(ac.q_list_random[arm_idx].q_net)
                        if epoch > no_comm_epochs and arm_idx in indices_no_noise:  # arms receiving information
                            sender_arm = random_neighbor[arm_idx] # double check this works
                            if fixed_use_behavioral:
                                ac.q_list_random_beh[arm_idx].q_net = copy.deepcopy(ac.q_list_random[sender_arm].q_net)
                            else:
                                ac.q_list_random[arm_idx].q_net = copy.deepcopy(ac.q_list_random[sender_arm].q_net)

            ##########################################################################################################
            # topline - no noise, no communication
            o = env.reset()  # crucial! else 'no communication' starts with bad states
            o = o.reshape(-1)
            for t in range(local_steps_per_epoch):
                torch_o = torch.as_tensor(o, dtype=torch.float32)
                a_agent = ac.step(torch_o, ac.q_list_topline, take_random_actions=take_random_actions[t])
                next_o, r, d, _ = env.step(a_agent, ac.opt_in)
                next_o = next_o.reshape(-1)
                buf_topline.store(o, ac.feature_arr, ac.opt_in, a_agent, r)
                o = next_o # Update obs (critical!)
            data_topline = copy.deepcopy(buf_topline.get())
            # print('topline: action ', a_agent, 'resources to noisy arms', sum(a_agent[indices_to_perturb])/sum(a_agent))
            with torch.no_grad():
                if epoch % self.target_q_update_freq == 0:
                    for arm_idx in range(env.N):
                        ac.target_q_list_topline[arm_idx].q_net = copy.deepcopy(ac.q_list_topline[arm_idx].q_net)
            for i in range(train_v_iters):
                loss_q_list = compute_loss_q(data_topline, target_q_list_choice=ac.target_q_list_topline,
                                             q_list_choice=ac.q_list_topline, optimizer_choice=qf_optimizers_topline)

            # evaluation
            if epoch % self.target_q_update_freq == 0 and epoch > 0:
                current_state = np.random.get_state()
                np.random.set_state(current_state)
                labels = ['comm', 'no comm', 'topline', 'oracle', 'worst', 'randomComm', 'random']
                N_TRIALS = 1
                np_seed_states = []; world_seed_states = []
                for i_trial in range(N_TRIALS):
                    np.random.rand()
                    env.random_stream.rand()
                    np_seed_states.append(np.random.get_state())
                    world_seed_states.append(env.random_stream.get_state())
                eval_rews = np.zeros((N_TRIALS, len(labels)))
                comm_budget_noisy = np.zeros((N_TRIALS, comm_steps_to_run))
                no_comm_budget_noisy = np.zeros((N_TRIALS, comm_steps_to_run))
                counter_comm_opt_state = np.zeros(env.N)
                counter_no_comm_opt_state = np.zeros(env.N)
                for i_trial in range(N_TRIALS):
                    np_seed_state_for_trial = np_seed_states[i_trial]
                    world_seed_state_for_trial = world_seed_states[i_trial]
                    np.random.set_state(np_seed_state_for_trial)
                    env.random_stream.set_state(world_seed_state_for_trial)
                    o = env.reset()
                    o = o.reshape(-1)
                    for t in range(10):
                        torch_o = torch.as_tensor(o, dtype=torch.float32)
                        a_agent = ac.act_test(torch_o, method='comm')
                        next_o, r, d, _ = env.step(a_agent, ac.opt_in)
                        next_o = next_o.reshape(-1)
                        o = next_o  # Update obs (critical!)
                        eval_rews[i_trial, 0] += sum(r)

                    if epoch > no_comm_epochs:
                        np_seed_state_for_trial = np_seed_states[i_trial]
                        world_seed_state_for_trial = world_seed_states[i_trial]
                        np.random.set_state(np_seed_state_for_trial)
                        env.random_stream.set_state(world_seed_state_for_trial)
                        o = env.reset()
                        o = o.reshape(-1)
                        for t in range(10):
                            torch_o = torch.as_tensor(o, dtype=torch.float32)
                            a_agent = ac.act_test(torch_o, method='control')
                            next_o, r, d, _ = env.step(a_agent, ac.opt_in)
                            next_o = next_o.reshape(-1)
                            o = next_o  # Update obs (critical!)
                            eval_rews[i_trial, 1] += sum(r)

                    np_seed_state_for_trial = np_seed_states[i_trial]
                    world_seed_state_for_trial = world_seed_states[i_trial]
                    np.random.set_state(np_seed_state_for_trial)
                    env.random_stream.set_state(world_seed_state_for_trial)
                    o = env.reset()
                    o = o.reshape(-1)
                    for t in range(10):
                        torch_o = torch.as_tensor(o, dtype=torch.float32)
                        a_agent = ac.act_test(torch_o, method='topline')
                        next_o, r, d, _ = env.step(a_agent, ac.opt_in)
                        next_o = next_o.reshape(-1)
                        o = next_o  # Update obs (critical!)
                        eval_rews[i_trial, 2] += sum(r)

                    if epoch > no_comm_epochs:
                        np_seed_state_for_trial = np_seed_states[i_trial]
                        world_seed_state_for_trial = world_seed_states[i_trial]
                        np.random.set_state(np_seed_state_for_trial)
                        env.random_stream.set_state(world_seed_state_for_trial)
                        o = env.reset()
                        o = o.reshape(-1)
                        for t in range(10):
                            torch_o = torch.as_tensor(o, dtype=torch.float32)
                            a_agent = ac.act_test(torch_o, method='oracle')
                            next_o, r, d, _ = env.step(a_agent, ac.opt_in)
                            next_o = next_o.reshape(-1)
                            o = next_o  # Update obs (critical!)
                            eval_rews[i_trial, 3] += sum(r)

                    if epoch > no_comm_epochs:
                        np_seed_state_for_trial = np_seed_states[i_trial]
                        world_seed_state_for_trial = world_seed_states[i_trial]
                        np.random.set_state(np_seed_state_for_trial)
                        env.random_stream.set_state(world_seed_state_for_trial)
                        o = env.reset()
                        o = o.reshape(-1)
                        for t in range(10):
                            torch_o = torch.as_tensor(o, dtype=torch.float32)
                            a_agent = ac.act_test(torch_o, method='worst')
                            next_o, r, d, _ = env.step(a_agent, ac.opt_in)
                            next_o = next_o.reshape(-1)
                            o = next_o  # Update obs (critical!)
                            eval_rews[i_trial, 4] += sum(r)

                    if epoch > no_comm_epochs:
                        np_seed_state_for_trial = np_seed_states[i_trial]
                        world_seed_state_for_trial = world_seed_states[i_trial]
                        np.random.set_state(np_seed_state_for_trial)
                        env.random_stream.set_state(world_seed_state_for_trial)
                        o = env.reset()
                        o = o.reshape(-1)
                        for t in range(10):
                            torch_o = torch.as_tensor(o, dtype=torch.float32)
                            a_agent = ac.act_test(torch_o, method='randomComm')
                            next_o, r, d, _ = env.step(a_agent, ac.opt_in)
                            next_o = next_o.reshape(-1)
                            o = next_o  # Update obs (critical!)
                            eval_rews[i_trial, 5] += sum(r)

                    np_seed_state_for_trial = np_seed_states[i_trial]
                    world_seed_state_for_trial = world_seed_states[i_trial]
                    np.random.set_state(np_seed_state_for_trial)
                    env.random_stream.set_state(world_seed_state_for_trial)
                    o = env.reset()
                    o = o.reshape(-1)
                    for t in range(10):
                        torch_o = torch.as_tensor(o, dtype=torch.float32)
                        a_agent = ac.step(torch_o, ac.q_list_topline, take_random_actions=1)
                        next_o, r, d, _ = env.step(a_agent, ac.opt_in)
                        next_o = next_o.reshape(-1)
                        o = next_o  # Update obs (critical!)
                        eval_rews[i_trial, -1] += sum(r)

                # write testing results in a csv
                newpath = os.getcwd() + '/logs/results/%s_n%s_b%s_S%s_e%s_noisy%s' % (
                    self.exp_name, env.N, int(env.B), self.pop_size, self.noise_level, self.n_noisy_arms)
                if not os.path.exists(newpath):
                    time.sleep(self.seed / 10) # avoid that multiple seeds creating folders at the same time
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                fname = os.getcwd() + '/logs/results/%s_n%s_b%s_S%s_e%s_noisy%s/eval_seed%s.csv' % (
                    self.exp_name, env.N, int(env.B), self.pop_size, self.noise_level, self.n_noisy_arms, self.seed)
                values = np.round(np.mean(eval_rews, axis=0), 4)

                if epoch <= no_comm_epochs:
                    values[1] = copy.deepcopy(values[0]) # no comm <- comm
                    values[3] = copy.deepcopy(values[0]) # oracle <- comm
                    values[4] = copy.deepcopy(values[0]) # worst <- comm
                    values[5] = copy.deepcopy(values[0]) # randomComm <- comm

                print('return ', values)
                if os.path.exists(fname):
                    with open(fname, 'a') as aggregate_file:
                        writer = csv.writer(aggregate_file)
                        writer.writerow(values)
                        aggregate_file.flush()
                else:
                    with open(fname, 'w') as aggregate_file:
                        writer = csv.writer(aggregate_file)
                        writer.writerow(labels)
                        writer.writerow(values)
                        aggregate_file.flush()
        return ac




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--hid', type=int, default=64, help="Number of units in each layer of the neural networks used for the Oracles")
    parser.add_argument('-l', type=int, default=2, help="Depth of the neural networks used for Agent and Nature Oracles (i.e., layers)")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--seed', '-s', type=int, default=0, help="Seed")
    parser.add_argument('--cpu', type=int, default=1, help="Number of processes for mpi")
    
    parser.add_argument('--exp_name', type=str, default='experiment', help="Experiment name")
    parser.add_argument('-N', type=int, default=5, help="Number of arms")
    parser.add_argument('-S', type=int, default=4, help="Number of states in each arm (when applicable, e.g., SIS)")
    parser.add_argument('-A', type=int, default=2, help="Number of actions in each arm (not currently implemented)")
    parser.add_argument('-B', type=float, default=1.0, help="Budget per round")
    parser.add_argument('--reward_bound', type=int, default=1, help="Rescale rewards to this value (only some environments)")
    parser.add_argument('--save_string', type=str, default="")
    parser.add_argument('--agent_steps', type=int, default=10, help="Number of rollout steps between epochs")
    parser.add_argument('--agent_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--agent_vdn_lr', type=float, default=2e-3, help="Learning rate for vdn network")
    parser.add_argument('--agent_vf_lr', type=float, default=2e-3, help="Learning rate for critic network")
    parser.add_argument('--agent_train_vf_iters', type=int, default=10, help="Training iterations to run per epoch")
    parser.add_argument('--agent_tp_transform', type=str, default=None, help="Type of transform to apply to transition probabilities, if any")
    parser.add_argument('--agent_tp_transform_dims', type=int, default=None, help="Number of output features to generate from input tps; only used if tp_transform is True") 
    parser.add_argument('--pop_size', type=int, default=0)
    parser.add_argument('--noise_level', type=float, default=1)
    parser.add_argument('--noise_shape', type=float, default=1)
    parser.add_argument('--n_noisy_arms', type=int, default=10)
    parser.add_argument('--no_comm_epochs', type=int, default=200)
    parser.add_argument('--num_atoms', type=int, default=21) # 0.99999 is effectively removing scheduler
    parser.add_argument('--target_q_update_freq', type=int, default=10)
    parser.add_argument('--home_dir', type=str, default='.', help="Home directory for experiments")
    parser.add_argument('--cannon', type=int, default=0, help="Flag used for running experiments on batched slurm-based HPC resources. Leave at 0 for small experiments.")
    parser.add_argument('-d', '--data', default='continuous_state', type=str, help='Environment selection',
                        choices=[
                                    'armman',
                                    'sis',
                                    'continuous_state',
                                    'counterexample'
                                ])

    parser.add_argument('--robust_keyword', default='sample_random', type=str, help='Method for picking some T out of the uncertain environment',
                        choices=[   
                                    'pess',
                                    'mid',
                                    'opt', # i.e., optimistic
                                    'sample_random'
                                ])

    args = parser.parse_args()

    N = args.N
    S = args.S
    A = args.A
    B = args.B
    budget = B
    reward_bound = args.reward_bound
    seed=args.seed
    data = args.data
    home_dir = args.home_dir
    exp_name=args.exp_name
    gamma = args.gamma

    opt_in_rate = 1
    num_atoms = args.num_atoms
    target_q_update_freq = args.target_q_update_freq

    torch.manual_seed(seed)
    np.random.seed(seed)

    agent_kwargs = {}
    agent_kwargs['steps_per_epoch'] = args.agent_steps
    agent_kwargs['epochs'] = args.agent_epochs
    agent_kwargs['vdn_lr'] = args.agent_vdn_lr
    agent_kwargs['vf_lr'] = args.agent_vf_lr
    agent_kwargs['train_v_iters'] = args.agent_train_vf_iters
    agent_kwargs['tp_transform'] = args.agent_tp_transform
    agent_kwargs['ac_kwargs'] = dict(hidden_sizes=[args.hid]*args.l,
                                     input_feat_dim=args.agent_tp_transform_dims)
    agent_kwargs['gamma'] = args.gamma
    agent_kwargs['noise_level'] = args.noise_level
    agent_kwargs['noise_shape'] = args.noise_shape
    agent_kwargs['n_noisy_arms'] = args.n_noisy_arms
    agent_kwargs['no_comm_epochs'] = args.no_comm_epochs

    env_fn = None

    one_hot_encode = False # by default, states are continuous valued
    non_ohe_obs_dim = 1
    state_norm = 1

    if args.data == 'continuous_state':
        env_fn = lambda : ContinuousStateExampleEnv(N,B,seed, args.noise_level, args.noise_shape)

    if args.data == 'counterexample':
        env_fn = lambda: CounterExampleEnv(N,B,seed, args.noise_level, args.noise_shape)

    if args.data == 'armman':
        env_fn = lambda: ARMMANRobustEnv(N,B,seed, args.noise_level, args.noise_shape)

    if args.data == 'sis':
        env_fn = lambda: SISRobustEnv(N,B,args.pop_size,seed, args.noise_level, args.noise_shape)
        state_norm = args.pop_size


    env = env_fn()

    agent_oracle  = AgentOracle(data, N, S, A, budget, seed, reward_bound,
                             agent_kwargs=agent_kwargs, home_dir=home_dir, exp_name=exp_name,
                             robust_keyword=args.robust_keyword,
                             # sampled_nature_parameter_ranges = sampled_nature_parameter_ranges,
                             pop_size=args.pop_size, one_hot_encode=one_hot_encode, state_norm=state_norm,
                             non_ohe_obs_dim=non_ohe_obs_dim, opt_in_rate=opt_in_rate,
                                target_q_update_freq=target_q_update_freq, num_atoms=num_atoms)

    nature_strategy = None



    add_to_seed = 0
    agent_oracle.best_response([nature_strategy], [1.0], add_to_seed)




