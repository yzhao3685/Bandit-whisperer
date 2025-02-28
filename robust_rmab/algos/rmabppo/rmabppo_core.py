

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

# from numba import jit
from itertools import product
# from code.lp_methods import action_knapsack

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)



class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPVDN(nn.Module):

    def __init__(self, input_dim, num_arms, activation, num_neighbors):
        super().__init__()
        self.q_net = np.zeros(num_arms, dtype=object)
        self.num_arms = num_arms
        for i in range(num_arms):
            self.q_net[i] = mlp([input_dim] + [1] + [1], activation)

    def forward(self, communication_strat):
        # communication_strat should be a vector of length num_arms. entry i indicates whether arm i info can be shared with other arms
        total_q = 0
        for i in range(self.num_arms):
            total_q += torch.squeeze(self.q_net[i](communication_strat[i].reshape(1)), -1) # Critical to ensure Q has right shape.
        return total_q

    def compute_best_action(self):
        opt_actions = torch.zeros(self.num_arms)
        for i in range(self.num_arms):
            if self.q_net[i](torch.ones(1)) > self.q_net[i](torch.zeros(1)):
                opt_actions[i] = 1
        return opt_actions


class MLPDistQCritic(nn.Module): # distributional RL

    def __init__(self, full_obs_dim, act_dim, hidden_sizes, activation, output_dim):
        super().__init__()
        # self.q_net = mlp([full_obs_dim + act_dim] + list(hidden_sizes) + [output_dim], activation)
        self.q_net = mlp([full_obs_dim] + list(hidden_sizes) + [act_dim * output_dim], activation)

    def forward(self, x):
        return torch.squeeze(self.q_net(x), -1) # this may produce negative values for probabilities


class MLPActorCriticRMAB(nn.Module):


    def __init__(self, observation_space, action_space, opt_in_rate=1.0,
                 hidden_sizes=(64,64), input_feat_dim=4, C=None, N=None, B=None,
                 strat_ind=0, one_hot_encode=True, non_ohe_obs_dim=None,
                 state_norm=None, seed=0, num_atoms=21, num_neighbors=5,
                 activation=nn.Tanh,
                 ):
        super().__init__()

        self.feature_arr = np.zeros((N, input_feat_dim))
        self.opt_in = np.ones(N) # assume all arms opt-in at the start
        self.opt_in_rate = opt_in_rate
        self.random_stream = np.random.RandomState()
        self.random_stream.seed(seed)

        # one-hot-encode the states for now
        self.obs_dim = observation_space.shape[0]
        if not one_hot_encode:
            self.obs_dim = non_ohe_obs_dim

        self.act_type = 'd' # for discrete

        self.non_ohe_obs_dim = non_ohe_obs_dim
        self.one_hot_encode = one_hot_encode

        # we will only work with discrete actions
        self.act_dim = action_space.shape[0]

        self.q_list_comm = np.zeros(N,dtype=object)
        self.target_q_list_comm = np.zeros(N,dtype=object)
        self.q_list_new_comm = np.zeros(N,dtype=object)
        self.q_list_control = np.zeros(N, dtype=object)
        self.target_q_list_control = np.zeros(N, dtype=object)
        self.q_list_topline = np.zeros(N, dtype=object)
        self.target_q_list_topline = np.zeros(N, dtype=object)
        self.q_list_oracle = np.zeros(N, dtype=object)
        self.target_q_list_oracle = np.zeros(N, dtype=object)
        self.q_list_oracle_beh = np.zeros(N, dtype=object)
        self.q_list_worst = np.zeros(N, dtype=object)
        self.target_q_list_worst = np.zeros(N, dtype=object)
        self.q_list_worst_beh = np.zeros(N, dtype=object)
        self.q_list_random = np.zeros(N, dtype=object)
        self.target_q_list_random = np.zeros(N, dtype=object)
        self.q_list_random_beh = np.zeros(N, dtype=object)
        self.vdn_net = MLPVDN(1, N, activation, num_neighbors) # for now, each individual network use input dimension 1 (whether this arm share info with others or not)
        self.N = N
        self.C = C
        self.B = B
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.state_norm = state_norm
        self.max_q = -1 # will overwrite

        self.input_feat_dim = input_feat_dim
        self.num_atoms = num_atoms
        for i in range(N):
            self.q_list_comm[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)
            self.target_q_list_comm[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)
            self.q_list_new_comm[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)
            self.q_list_control[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)
            self.target_q_list_control[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)
            self.q_list_topline[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)
            self.target_q_list_topline[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)
            self.q_list_oracle[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)
            self.target_q_list_oracle[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)
            self.q_list_oracle_beh[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)
            self.q_list_random[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)
            self.target_q_list_random[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)
            self.q_list_random_beh[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)
            self.q_list_worst[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)
            self.target_q_list_worst[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)
            self.q_list_worst_beh[i] = MLPDistQCritic(self.obs_dim, self.act_dim, hidden_sizes, activation, self.num_atoms)

        transition_prob_dim = int(N * input_feat_dim)

        self.name = "RMABPPO"
        self.ind = strat_ind

        self.epsilon = 0.3 # in training, choose actions using epsilon-greedy

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def step(self, obs, q_list_choice, take_random_actions):
        with torch.no_grad():
            if not self.one_hot_encode:
                obs = obs/self.state_norm
            a_list = np.zeros(self.N,dtype=int)
            if take_random_actions < 0.5:
                actions = self.act_test_deterministic_q_based_multiaction(obs, q_list_choice)
            else:
                C = self.C; B = self.B; N = self.N
                actions = np.zeros(N, dtype=int)
                current_action_cost = 0
                process_order = self.random_stream.choice(np.arange(N), N, replace=False)
                for arm in process_order:
                    # select an action at random
                    num_valid_actions_left = len(C[C <= B - current_action_cost])
                    p = 1 / (C[C <= B - current_action_cost] + 1)
                    p = p / p.sum()
                    p = None
                    a = self.random_stream.choice(np.arange(num_valid_actions_left), 1, p=p)[0]
                    current_action_cost += C[a]
                    # if the next selection takes us over budget, break
                    if current_action_cost > B:
                        break
                    actions[arm] = a
            a_list = actions

        return a_list

    def act_test(self, obs, method='comm'):
        obs=obs.reshape(-1)
        if method=='comm':
            return self.act_test_deterministic_q_based_multiaction(obs, self.q_list_comm)
        elif method=='control':
            return self.act_test_deterministic_q_based_multiaction(obs, self.q_list_control)
        elif method == 'oracle':
            return self.act_test_deterministic_q_based_multiaction(obs, self.q_list_oracle)
        elif method == 'worst':
            return self.act_test_deterministic_q_based_multiaction(obs, self.q_list_worst)
        elif method == 'randomComm':
            return self.act_test_deterministic_q_based_multiaction(obs, self.q_list_random)
        else: # method=='topline'
            return self.act_test_deterministic_q_based_multiaction(obs, self.q_list_topline)

    # Multi-action implementation
    def act_test_deterministic_q_based_multiaction(self, obs, q_list_choice):
        actions = np.zeros(self.N, dtype=int)
        q_list = np.zeros((self.N, self.act_dim), dtype=float)
        softmax_layer = nn.Softmax(dim=1)
        max_q = self.max_q ; min_q = 0  # found performances goes down as min_q increases
        bin_interval = (max_q - min_q) / (self.num_atoms - 1)  # interval length of each bin in categorical DQN
        with torch.no_grad():
            if not self.one_hot_encode:
                obs = obs / self.state_norm
            for i in range(self.N):
                transition_prob = self.feature_arr[i]
                full_obs = None
                if self.one_hot_encode:
                    ohs = np.zeros(self.obs_dim); ohs[int(obs[i])] = 1; full_obs = ohs
                else:
                    full_obs = obs[i].reshape(1)
                full_obs = torch.as_tensor(full_obs, dtype=torch.float32)
                q_distribution = q_list_choice[i](full_obs)
                q_distribution = q_distribution.reshape((self.act_dim, self.num_atoms))
                q_distribution = softmax_layer(q_distribution).numpy()
                q_list[i] = np.dot(q_distribution, np.arange(self.num_atoms) * bin_interval + min_q)
            # breakpoint()
            q_list = q_list - q_list[:, 0][:, None] # take into account the benefit from pulling
            q_list = np.maximum(q_list, 0) # no pull is the worst. this line may not be necessary as results are always positive
            pi_list = q_list

            # sort each list, then play them in order
            row_maxes = pi_list.max(axis=1)
            row_order = np.argsort(row_maxes)[::-1]

            pi_arg_maxes = np.argsort(pi_list, axis=1)

            actions = np.zeros(self.N, dtype=int)

            budget_spent = 0
            done = False

            while budget_spent < self.B and not done:

                i = 0
                while i < self.N:
                    arm = row_order[i]
                    arm_a = pi_arg_maxes[row_order[i]][-1]
                    if self.opt_in[i] < 0.5:
                        arm_a = 0  # 'no pull' action for opt-out arms.
                    a_cost = self.C[arm_a]

                    # if difference in price takes us over, we have to stop
                    if budget_spent + a_cost - self.C[actions[arm]] > self.B:
                        done = True
                        break
                    else:
                        i += 1
                        # assign action
                        actions[arm] = arm_a
                        # now hide all cheaper actions
                        pi_list[arm, :arm_a + 1] = 0
                        # print(a)

                        budget_spent = sum(self.C[a] for a in actions)

                        # also hide all actions that are now too expensive
                        cost_diff_array = np.zeros(pi_list.shape)
                        # print('actions',actions)
                        for j in range(self.N):
                            cost_diff_array[j] = self.C - self.C[actions[j]]
                        overbudget_action_inds = cost_diff_array > self.B - budget_spent

                        if overbudget_action_inds.any():
                            i = 0
                            pi_list[overbudget_action_inds] = 0
                            row_maxes = pi_list.max(axis=1)
                            row_order = np.argsort(row_maxes)[::-1]

                            pi_arg_maxes = np.argsort(pi_list, axis=1)
                        if not pi_list.sum() > 0:
                            done = True
                            break

                row_maxes = pi_list.max(axis=1)
                row_order = np.argsort(row_maxes)[::-1]

                pi_arg_maxes = np.argsort(pi_list, axis=1)

        return actions

    # def featurize_tp(self, transition_probs, transformation=None, out_dim=4, in_dim=4):
    #     N = transition_probs.shape[0]
    #     output_features = np.zeros((N, out_dim))
    #     np.random.seed(0)  # Set random seed for reproducibility
    #
    #     if transformation == "linear":
    #         transformation_matrix = np.random.rand(in_dim, out_dim)
    #         output_features = np.dot(transition_probs, transformation_matrix)
    #     elif transformation == "nonlinear":
    #         transformation_matrix = np.random.rand(in_dim, out_dim)
    #         output_features = 1 / (1 + np.exp(-np.dot(transition_probs, transformation_matrix)))
    #     else:
    #         output_features[:, :min(in_dim, out_dim)] = transition_probs[:, :min(in_dim, out_dim)]
    #     return output_features
