import numpy as np
import gym
import torch
from scipy.special import comb
from copy import deepcopy
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler


# class ARMMANRobustEnv_old(gym.Env):
#     def __init__(self, N, B, seed, self.noise_level):#, REWARD_BOUND):
#
#
#         S = 3
#         A = 2
#
#         self.N = N
#         self.observation_space = np.arange(S)
#         self.action_space = np.arange(A)
#         self.observation_dimension = 1
#         self.action_dimension = 1
#         self.action_dim_nature = N*A
#         # self.REWARD_BOUND = REWARD_BOUND
#         # self.reward_range = (0, REWARD_BOUND)
#         self.S = S
#         self.A = A
#         self.B = B
#
#         self.percent_A = 0.2
#         self.percent_B = 0.2
#
#         self.current_full_state = np.zeros(N)
#         self.random_stream = np.random.RandomState()
#
#         self.PARAMETER_RANGES = self.get_parameter_ranges(self.N) # this range is pretty big, e.g. [0.3, 0.8]
#         self.PARAMETER_RANGES = self.sample_parameter_ranges() # this range is much smaller, e.g. [0.5, 0.7]
#
#         # make sure to set this whenever environment is created, but do it outside so it always the same
#         # self.sampled_parameter_ranges = None
#         self.param_setting = np.zeros(self.PARAMETER_RANGES.shape[:-1])
#         self.max_reward = 1 # placeholder, will be overwritten by get_experiment()
#         self.num_opt_states = 1 # placeholder, will be overwritten by get_experiment()
#
#         self.seed(seed=seed)
#         self.num_partitions = 50 # this is used in perturbed_R calculations
#         self.T, self.R, self.C, self.perturbed_R = self.get_experiment(N)
#         self.update_T()
#
#         self.tanh = torch.nn.Tanh()
#         self.sigmoid = torch.nn.Sigmoid()
#
#
#     def update_transition_probs(self, arms_to_update):
#         # arms_to_update is 1d array of length N. arms_to_update[i] == 1 if transition prob of arm i needs to be resampled
#         # parameters are [arm_i, arm_state, arm_a]
#         # for each arm, we first sample parameters ranges, then sample from the parameter range.
#         # thus, arms are sampled from the same distribution, as long as they are in rangeA, rangeB, and rangeC with given probabilities
#         for i in range(self.N):
#             if arms_to_update[i] > 0.5:
#                 for j in range(self.PARAMETER_RANGES.shape[1]):
#                     sample_ub = self.PARAMETER_RANGES[i, j, :, 1]
#                     sample_lb = self.PARAMETER_RANGES[i, j, :, 0]
#                     new_params = self.random_stream.uniform(low=sample_lb, high=sample_ub)
#                     self.param_setting[i, j, :] = new_params
#         self.update_T()
#
#
#
#     # new version has one range per state, per action
#     # We will sample ranges from within these to get some extra randomness
#     def get_parameter_ranges(self, N):
#
#         # A - 10 in A
#         rangeA = [
#                     [
#                         [0.0, 1.0],
#                         [0.0, 1.0]
#                     ],
#                     [
#                         [0.5, 1.0], # p deteriorate in absence of intervention
#                         [0.5, 1.0], # p improve on intervention
#                     ],
#                     [
#                         [0.35, 0.85],
#                         [0.35, 0.85]
#                     ]
#
#                 ]
#
#
#         # B - 10 in B
#         rangeB = [
#                     [
#                         [0.0, 1.0],
#                         [0.0, 1.0]
#                     ],
#                     [
#                         [0.35, 0.85], # p deteriorate in absence of intervention
#                         [0.15, 0.65], # p improve on intervention
#                     ],
#                     [
#                         [0.35, 0.85],
#                         [0.35, 0.85]
#                     ]
#
#                 ]
#
#         # C - 30 in C
#         rangeC = [
#                     [
#                         [0.0, 1.0],
#                         [0.0, 1.0]
#                     ],
#                     [
#                         [0.35, 0.85], # p deteriorate in absence of intervention
#                         [0.0, 0.5], # p improve on intervention
#                     ],
#                     [
#                         [0.35, 0.85],
#                         [0.35, 0.85]
#                     ]
#
#                 ]
#
#
#
#         num_A = int(N*self.percent_A)
#         num_B = int(N*self.percent_B)
#         num_C = N  - num_A - num_B
#
#         parameter_ranges = []
#         for i in range(num_A):
#             parameter_ranges.append(rangeA)
#         for i in range(num_B):
#             parameter_ranges.append(rangeB)
#         for i in range(num_C):
#             parameter_ranges.append(rangeC)
#
#         # self.parameter_ranges = np.array(parameter_ranges)
#
#         return np.array(parameter_ranges)
#
#
#     def sample_parameter_ranges(self):
#         draw = self.random_stream.rand(*self.PARAMETER_RANGES.shape)
#         mult_transform = (self.PARAMETER_RANGES.max(axis=-1) - self.PARAMETER_RANGES.min(axis=-1))
#         mult_transform = np.expand_dims(mult_transform, axis=-1)
#         add_transform = self.PARAMETER_RANGES.min(axis=-1)
#         add_transform = np.expand_dims(add_transform, axis=-1)
#
#         draw.sort(axis=-1)
#
#         sampled_ranges = draw*mult_transform + add_transform
#
#         assert self.check_ranges(sampled_ranges, self.PARAMETER_RANGES)
#
#         return sampled_ranges
#
#     def check_ranges(self, sampled, edges):
#         all_good = True
#         for i in range(edges.shape[0]):
#             for j in range(edges.shape[1]):
#                 for k in range(edges.shape[2]):
#                     # lower range must be larger or equal to lower edge
#                     all_good &= (sampled[i,j,k,0] >= edges[i,j,k,0])
#                     # upper range must be smaller or equal to upper edge
#                     all_good &= (sampled[i,j,k,1] <= edges[i,j,k,1])
#                     if not all_good:
#                         print('range ',edges[i,j,k])
#                         print('sample',sampled[i,j,k])
#                         print()
#
#         return all_good
#
#
#
#
#
#     def get_experiment(self, N):
#
#         percent_A = 0.2
#
#         percent_B = 0.2
#
#
#         # States go S, P, L
#         #
#
#         # A - 10 in A
#         tA = np.array([[[0.1, 0.9, 0.0],
#                         [0.1, 0.9, 0.0]],
#
#                         [[0, -1, -1],
#                         [-1, -1, 0]],
#
#                         [[0, 0.4, 0.6],
#                         [0.0, 0.4, 0.6]]
#                         ])
#         # B - 10 in B
#         tB = np.array([[[0.9, 0.1, 0.0],
#                         [0.9, 0.1, 0.0]],
#                         [[0, -1, -1],
#                         [-1, -1, 0]],
#                         [[0, 0.4, 0.6],
#                         [0.0, 0.4, 0.6]]
#                         ])
#
#
#         # C - 30 in C
#         tC = np.array([[[0.1, 0.9, 0.0],
#                         [0.1, 0.9, 0.0]],
#                         [[0, -1, -1],
#                         [-1, -1, 0]],
#                         [[0, 0.4, 0.6],
#                         [0.0, 0.4, 0.6]]
#                         ])
#
#
#
#         num_A = int(N*percent_A)
#         num_B = int(N*percent_B)
#         num_C = N  - num_A - num_B
#
#         T = []
#         for i in range(num_A):
#             T.append(tA)
#         for i in range(num_B):
#             T.append(tB)
#         for i in range(num_C):
#             T.append(tC)
#
#         T = np.array(T)
#         # R = np.array([[1,0.5,0] for _ in range(N)])
#         C = np.array([0, 1])
#
#         R = 1 - np.array([np.linspace(0, 1, self.num_partitions) for _ in range(N)])
#         perturbed_R = 1 - np.array([np.linspace(0, 1, self.num_partitions) for _ in range(N)])
#         noise = np.zeros(perturbed_R.shape)
#         if self.noise_shape == 1:
#             noise = self.random_stream.normal(loc=0, scale=1 * self.noise_level, size=perturbed_R.shape)
#         elif self.noise_shape == 2:
#             noise = self.random_stream.uniform(low=-0.5, high=0.5, size=perturbed_R.shape)
#         perturbed_R = np.clip(perturbed_R + noise, 0, 1)
#
#         self.max_reward = 1
#         self.num_opt_states = 1
#         R[:, :self.num_opt_states] = self.max_reward * np.ones(self.num_opt_states) # optimal states
#         perturbed_R[:, :self.num_opt_states] = self.max_reward * np.ones(self.num_opt_states) # optimal states
#
#         return T, R, C, perturbed_R
#
#     def update_T(self):
#         for arm_i in range(self.N):
#             for arm_a in range(self.param_setting.shape[-1]):
#                 for arm_state in range(3):
#                     # arm_state = int(self.current_full_state[arm_i])
#                     param = self.param_setting[arm_i, arm_state, arm_a] # this requires a second look. check a_nature dim
#
#                     # semi-annoying specific code to make sure we set the right entries for each state
#                     if arm_state == 0:
#                         # in both action cases we'll set the p of changing to state 0
#                         self.T[arm_i,arm_state,arm_a,0] = param
#                         self.T[arm_i,arm_state,arm_a,1] = 1-param
#
#                     elif arm_state == 1:
#                         # if action is 0 set the p of changing to state 2 (worse state)
#                         if arm_a == 0:
#                             self.T[arm_i,arm_state,arm_a,2] = param
#                             self.T[arm_i,arm_state,arm_a,1] = 1-param
#
#                         # if action is 1 set the p of changing to state 0 (better state)
#                         elif arm_a == 1:
#                             self.T[arm_i,arm_state,arm_a,0] = param
#                             self.T[arm_i,arm_state,arm_a,1] = 1-param
#
#                     elif arm_state == 2:
#                         # in both action cases we'll set the p of changing to state 2
#                         self.T[arm_i,arm_state,arm_a,2] = param
#                         self.T[arm_i,arm_state,arm_a,1] = 1-param
#                     else:
#                         raise ValueError('Got incorrect state')
#         # for arm in range(25): # for sanity checks
#         #     for state in range(3):
#         #         for action in range(2):
#         #             gaus_mean = np.arange(3) @ self.T[arm, state, action, :]
#         #             print('state ', state, 'action ', action, 'mean ', gaus_mean)
#         ###### Get next state
#
#
#     def step(self, a_agent, opt_in, perturb_reward=False, indices_to_perturb=[]):
#         next_full_state = np.zeros(self.N, dtype=float)
#         rewards = np.zeros(self.N)
#         for i in range(self.N):
#             # current_arm_state=int(self.current_full_state[i]) # this would round the arm state
#             # next_arm_state=np.argmax(self.random_stream.multinomial(1, self.T[i, current_arm_state, int(a_agent[i]), :]))
#             # next_full_state[i]=next_arm_state
#             # rewards[i] = self.R[i, next_arm_state]
#
#             # the mean of the normal should be expected value of the multinomial dist
#             constant = 1
#             if self.current_full_state[i] == 0:
#                 constant = 0.25
#             elif self.current_full_state[i] == 2:
#                 constant = 1.75
#             transition_probs = self.T[i, int(self.current_full_state[i]), int(a_agent[i]), :]
#             # gaus_mean = np.arange(3) @ transition_probs - 1
#             # transition_probs = self.T[i, 1, int(a_agent[i]), :] # simplified setting, specified by one normal dist
#             assert sum(transition_probs > 0.01) == 2 # else the variance computation no longer works
#             gaus_std = np.prod(transition_probs[transition_probs > 0.01]) # only two entries are positive. so simply variance of bernoulli
#             gaus_mean = np.arange(3) @ transition_probs - constant
#             next_arm_state = self.current_full_state[i] + self.random_stream.normal(loc=gaus_mean, scale=gaus_std)
#             next_full_state[i] = np.clip(next_arm_state, 0, 2) # clip to state range
#
#             # once reach opt states, always stay there. in this armman env, opt state is the lowest
#             if self.current_full_state[i] <= self.num_opt_states / self.num_partitions:
#                 next_full_state[i] = 0
#
#             rewards[i] = self.reward_fun(next_full_state[i], i)
#             if perturb_reward and i in indices_to_perturb: # noisy arms only seed perturbed reward
#                 rewards[i] = self.reward_fun(next_full_state[i], i, perturbed=True)
#
#         self.current_full_state = next_full_state
#         next_full_state = next_full_state.reshape(self.N, self.observation_dimension)
#
#         return next_full_state, rewards, False, None
#
#     def reward_fun(self, state, arm_idx, perturbed=False):
#         # find the partition this state belongs to
#         partition = int(round(self.num_partitions * state / 2))
#         partition = min(partition, self.num_partitions - 1)
#         if perturbed:
#             return self.perturbed_R[arm_idx, partition]
#         else:
#             return 1 - state / 2 # self.R[arm_idx, partition]
#
#
#
#
#     # a_agent should correspond to an action respresented in the transition matrix
#     # a_nature should be a probability in the range specified by self.parameter_ranges
#     def get_T_for_a_nature(self, a_nature_expanded):
#
#         for arm_i in range(a_nature_expanded.shape[0]):
#             for arm_state in range(a_nature_expanded.shape[1]):
#                 for arm_a in range(a_nature_expanded.shape[2]):
#
#                     param = a_nature_expanded[arm_i, arm_state, arm_a]
#
#                     if param < self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 0] or param > self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 1]:
#                         raise ValueError("Nature setting outside allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, arm_state, arm_a]))
#                         # print("Warning! nature action below allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, arm_state, arm_a]))
#                         # print("Setting to lower bound of range...")
#                         # param = self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 0]
#                     # elif
#                     #     print("Warning! nature action above allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, arm_state, arm_a]))
#                     #     print("Setting to upper bound of range...")
#                     #     param = self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 1]
#
#                     # semi-annoying specific code to make sure we set the right entries for each state
#                     if arm_state == 0:
#                         # in both action cases we'll set the p of changing to state 0
#                         self.T[arm_i,arm_state,arm_a,0] = param
#                         self.T[arm_i,arm_state,arm_a,1] = 1-param
#
#                     elif arm_state == 1:
#                         # if action is 0 set the p of changing to state 2 (worse state)
#                         if arm_a == 0:
#                             self.T[arm_i,arm_state,arm_a,2] = param
#                             self.T[arm_i,arm_state,arm_a,1] = 1-param
#
#                         # if action is 1 set the p of changing to state 0 (better state)
#                         elif arm_a == 1:
#                             self.T[arm_i,arm_state,arm_a,0] = param
#                             self.T[arm_i,arm_state,arm_a,1] = 1-param
#
#                     elif arm_state == 2:
#                         # in both action cases we'll set the p of changing to state 2
#                         self.T[arm_i,arm_state,arm_a,2] = param
#                         self.T[arm_i,arm_state,arm_a,1] = 1-param
#                     else:
#                         raise ValueError('Got incorrect state')
#
#         return np.copy(self.T)
#
#
#     # this is easier to attach to environment code
#     def bound_nature_actions(self, a_nature_flat, state=None, reshape=True):
#
#         # num arms by num actions
#         a_nature = a_nature_flat.reshape(self.N, self.T.shape[2])
#
#         a_nature_bounded = np.zeros(a_nature.shape)
#         for arm_i in range(a_nature.shape[0]):
#             for arm_a in range(a_nature.shape[1]):
#
#                 param = a_nature[arm_i,arm_a]
#
#                 arm_state = int(state[arm_i])
#                 lb = self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 0]
#                 ub = self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 1]
#
#                 # print('range',lb, ub)
#                 # print('param in',param)
#                 # print('arm state',arm_state)
#
#                 a_nature_bounded[arm_i,arm_a] = ((self.tanh(torch.as_tensor(param, dtype=torch.float32))+1)/2)*(ub - lb) + lb
#                 # print('param out', a_nature_bounded[arm_i,arm_a])
#                 # print()
#
#         if not reshape:
#             a_nature_bounded = a_nature_bounded.reshape(*a_nature_flat.shape)
#
#         return a_nature_bounded
#
#
#
#
#     def reset(self):
#         # self.current_full_state = np.zeros(self.N, dtype=int)
#         # self.current_full_state = self.random_stream.choice(self.observation_space, self.N)
#         self.current_full_state = self.random_stream.uniform(low=0, high=2, size=self.N)
#         return self.current_full_state.reshape(self.N, self.observation_dimension)
#
#     def reset_random(self):
#         return self.reset()
#
#     def render(self):
#         return None
#
#     def close(self):
#         pass
#
#     def seed(self, seed=None):
#         seed1 = seed
#         if seed1 is not None:
#             self.random_stream.seed(seed=seed1)
#             # print('seeded with',seed1)
#         else:
#             seed1 = np.random.randint(1e9)
#             self.random_stream.seed(seed=seed1)
#
#         return [seed1]


class ARMMANRobustEnv(gym.Env):
    def __init__(self, N, B, seed, noise_level=1, noise_shape=1):
        S = 2
        A = 2
        # N = 3

        self.N = N
        self.observation_space = np.arange(S)
        self.action_space = np.arange(A)
        self.observation_dimension = 1
        self.action_dimension = 1
        self.action_dim_nature = N
        self.S = S
        self.A = A
        self.B = B
        self.num_partitions = 51 # this is not number of atoms. this is how we discretize reward function
        self.max_reward = 1 # placehooder. will be updated by get_environment()

        self.current_full_state = np.zeros(N)
        self.random_stream = np.random.RandomState()

        # self.PARAMETER_RANGES = self.get_parameter_ranges(self.N)

        # features (num_beneficiaries=7668, num_features=44),
        # count_matrices (num_beneficiaries=7668, num_states=2, num_actions=2, num_states=2),

        # make sure to set this whenever environment is created, but do it outside so it always the same
        self.sampled_parameter_ranges = None
        self.seed(seed=seed)
        self.noise_level = noise_level
        self.noise_shape = noise_shape

        # pop_level_count (1, num_states=2, num_actions=2, num_states=2)
        from ..environments.armman_real_data import features, count_matrices, pop_level_count

        random_idxs = self.random_stream.choice(features.shape[0], N, replace=False)

        self.T, self.R, self.C, self.perturbed_R = self.get_experiment(N)
        transition_prior = 1
        self.T = count_matrices[random_idxs] + transition_prior * pop_level_count  # replace T with real data
        self.T /= self.T.sum(
            -1, keepdims=True
        )  # Normalize to get probabilities

        # self.features = features[random_idxs]
        # scaler = MinMaxScaler()
        # scaler.fit(self.features)
        # self.features = scaler.transform(self.features)


        effect_std = np.average(np.prod(self.T[:, :, :, :], axis=3), axis=(1, 2)).reshape((-1,1))
        pull_effect = np.average(self.T[:, :, 1, 1] - self.T[:, :, 0, 1], axis=1).reshape((-1,1))
        self.features = np.hstack((pull_effect / 2, - pull_effect / 2, effect_std, effect_std))
        # no_pull_effect = - self.T[:, 1, 0, 0].reshape((self.N, 1))
        # pull_effect = self.T[:, 0, 1, 1].reshape((self.N, 1))
        # self.features = np.hstack((pull_effect, no_pull_effect, effect_std, effect_std))

        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    # # new version has one range per state, per action
    # # We will sample ranges from within these to get some extra randomness
    # def get_parameter_ranges(self, N):
    #     # A - 10 in A - 0, middle
    #     rangeA = [0, 1]
    #
    #     # B - 10 in B - 1, bottom
    #     rangeB = [0.05, 0.9]
    #     # rangeB = [0.05, 0.65] # nudge the middle a bit lower so RL learns the middle policy exactly
    #
    #     # C - 30 in C - 2, top
    #     rangeC = [0.1, 0.95]
    #     # rangeC = [0.35, 0.95] # nudge the middle a bit higher so RL learns the middle policy exactly
    #
    #     parameter_ranges = []
    #
    #     i = 0
    #     while i < N:
    #         if i % 3 == 0:
    #             parameter_ranges.append(rangeA)
    #         if i % 3 == 1:
    #             parameter_ranges.append(rangeB)
    #         if i % 3 == 2:
    #             parameter_ranges.append(rangeC)
    #         i += 1
    #
    #     # self.parameter_ranges = np.array(parameter_ranges)
    #
    #     return np.array(parameter_ranges)

    def sample_parameter_ranges(self):
        return np.copy(self.PARAMETER_RANGES)

    def get_experiment(self, N):
        # States go S, P, L
        #

        # A - 10 in A
        import os

        if os.getenv("DIST_SHIFT") is None:
            dist_shift = 0.0
        else:
            dist_shift = float(os.environ["DIST_SHIFT"])

        t = np.array(
            [
                [
                    [0.5 + dist_shift, 0.5 - dist_shift],
                    [0.5 + dist_shift, 0.5 - dist_shift],
                ],
                [
                    [1.0 - dist_shift, 0.0 + dist_shift],
                    [0.0, -1.0],
                ],  # only set the param for acting in state 1
            ]
        )

        T = []
        for i in range(N):
            T.append(t)

        T = np.array(T)
        # R = np.array([[0, 1] for _ in range(N)])
        C = np.array([0, 1])

        R = np.array([np.linspace(0, 1, self.num_partitions) for _ in range(N)])
        perturbed_R = np.array([np.linspace(0, 1, self.num_partitions) for _ in range(N)])
        # perturbed_R += np.ones(perturbed_R.shape) * 0.5
        # perturbed_R /= 2 # noisy arms overestimate when state low, and underestimate when state high
        # for n_s in range(len(perturbed_R)): # logit reward function
        #     if n_s !=0 and n_s != len(perturbed_R - 1):
        #         perturbed_R[n_s] = 0.1 * np.log(perturbed_R[n_s] / (1  - perturbed_R[n_s])) + 0.5
        noise = np.zeros(perturbed_R.shape)

        if self.noise_shape == 1:
            noise = self.random_stream.normal(loc=0, scale=1 * self.noise_level, size=perturbed_R.shape)
        elif self.noise_shape == 2:
            noise = self.random_stream.uniform(low=-0.3, high=0.3, size=perturbed_R.shape)
        elif self.noise_shape == 3:
            noise = self.random_stream.uniform(low=-0.5, high=0.5, size=perturbed_R.shape)
        elif self.noise_shape == 4:
            noise = self.random_stream.uniform(low=-0.7, high=0.7, size=perturbed_R.shape)
        elif self.noise_shape == 5:
            coin_flip_prob = 0.3
            coin_flips = np.random.binomial(n=1, p=coin_flip_prob, size=perturbed_R.shape)
            noise_1 = self.random_stream.uniform(low=-0.5, high=0.5, size=perturbed_R.shape)
            noise_2 = self.random_stream.normal(loc=0, scale=1 * self.noise_level, size=perturbed_R.shape)
            noise = coin_flips * noise_1 + (1 - coin_flips) * noise_2
        elif self.noise_shape == 6:
            coin_flip_prob = 0.5
            coin_flips = np.random.binomial(n=1, p=coin_flip_prob, size=perturbed_R.shape)
            noise_1 = self.random_stream.uniform(low=-0.5, high=0.5, size=perturbed_R.shape)
            noise_2 = self.random_stream.normal(loc=0, scale=1 * self.noise_level, size=perturbed_R.shape)
            noise = coin_flips * noise_1 + (1 - coin_flips) * noise_2
        elif self.noise_shape == 7:
            coin_flip_prob = 0.7
            coin_flips = np.random.binomial(n=1, p=coin_flip_prob, size=perturbed_R.shape)
            noise_1 = self.random_stream.uniform(low=-0.5, high=0.5, size=perturbed_R.shape)
            noise_2 = self.random_stream.normal(loc=0, scale=1 * self.noise_level, size=perturbed_R.shape)
            noise = coin_flips * noise_1 + (1 - coin_flips) * noise_2
        perturbed_R = np.clip(perturbed_R + noise, 0, 1)

        self.max_reward = 1
        self.num_opt_states = 1
        R[:, -self.num_opt_states:] = self.max_reward * np.ones(self.num_opt_states) # optimal states
        perturbed_R[:, -self.num_opt_states:] = self.max_reward * np.ones(self.num_opt_states) # optimal states

        return T, R, C, perturbed_R

    # env has only binary actions so random is easy to generate
    def random_agent_action(self):
        actions = np.zeros(self.N)
        choices = self.random_stream.choice(np.arange(self.N), int(self.B), replace=False)
        actions[choices] = 1
        return actions

    # def compute_pull_effect(self):
    #     causal_effect_mean = np.average(self.T[:, :, 1, 1] - self.T[:, :, 0, 1], axis=1)  # on avg, the benefit of pulling
    #     # no_pull_effect =  - self.T[:, 1, 0, 0]
    #     # pull_effect = self.T[:, 0, 1, 1]
    #     effect_std = np.average(np.prod(self.T[:, :, :, :], axis=3), axis=(1,2))  # variance of bernoulli is simply p(1-p)
    #     return causal_effect_mean, effect_std

    def step(self, a_agent, opt_in, perturb_reward=False, indices_to_perturb=[]):
        next_full_state = np.zeros(self.N, dtype=float)
        rewards = np.zeros(self.N)
        causal_effect_mean = np.average(self.T[:, :, 1, 1] - self.T[:, :, 0, 1], axis=1)
        # no_pull_effect = - self.T[:, 1, 0, 0]
        # pull_effect = self.T[:, 0, 1, 1]
        effect_std = np.average(np.prod(self.T[:, :, :, :], axis=3), axis=(1, 2))
        for i in range(self.N):
            # next_arm_state = np.argmax(self.random_stream.multinomial(1, self.T[i, int(self.current_full_state[i]), int(a_agent[i]), :]))
            # next_full_state[i] = next_arm_state
            if  int(a_agent[i]) > 0:
                effect_mean = causal_effect_mean[i] / 2
            else:
                effect_mean = - causal_effect_mean[i] / 2
            next_arm_state = self.current_full_state[i] + self.random_stream.normal(loc=effect_mean, scale=effect_std[i])
            next_arm_state = np.clip(next_arm_state, 0, 1) # clip to state range
            next_full_state[i] = next_arm_state
            # # once reach opt states, always stay there.
            # if self.current_full_state[i] >= 1 - self.num_opt_states / self.num_partitions:
            #     next_full_state[i] = 1

            rewards[i] = self.reward_fun(next_full_state[i], i)
            if perturb_reward and i in indices_to_perturb: # noisy arms only seed perturbed reward
                rewards[i] = self.reward_fun(next_full_state[i], i, perturbed=True)

        self.current_full_state = next_full_state
        next_full_state = next_full_state.reshape(self.N, self.observation_dimension)
        return next_full_state, rewards, False, None

    def reward_fun(self, state, arm_idx, perturbed=False):
        # find the partition this state belongs to
        partition = int(round(self.num_partitions * state))
        partition = min(partition, self.num_partitions - 1)
        if perturbed:
            return self.perturbed_R[arm_idx, partition]
        else:
            return state

    # a_agent should correspond to an action respresented in the transition matrix
    # a_nature should be a probability in the range specified by self.parameter_ranges
    def get_T_for_a_nature(self, a_nature_expanded):
        for arm_i in range(a_nature_expanded.shape[0]):
            param = a_nature_expanded[arm_i]

            if (
                    param < self.sampled_parameter_ranges[arm_i, 0]
                    or param > self.sampled_parameter_ranges[arm_i, 1]
            ):
                raise ValueError(
                    "Nature setting outside allowed param range. Was %s but should be in %s"
                    % (param, self.sampled_parameter_ranges[arm_i])
                )
                # print("Warning! nature action below allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, arm_state, arm_a]))
                # print("Setting to lower bound of range...")
                # param = self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 0]

            self.T[arm_i, 1, 1, 1] = param
            self.T[arm_i, 1, 1, 0] = 1 - param

        return np.copy(self.T)

    # this is easier to attach to environment code
    # RETUNR HERE WHEN DONE
    def bound_nature_actions(self, a_nature_flat, state=None, reshape=True):
        # num arms by num actions
        a_nature = a_nature_flat.reshape(self.N)

        a_nature_bounded = np.zeros(a_nature.shape)
        for arm_i in range(a_nature.shape[0]):
            param = a_nature[arm_i]

            lb = self.sampled_parameter_ranges[arm_i, 0]
            ub = self.sampled_parameter_ranges[arm_i, 1]

            a_nature_bounded[arm_i] = (
                                              (self.tanh(torch.as_tensor(param, dtype=torch.float32)) + 1) / 2
                                      ) * (ub - lb) + lb

        if not reshape:
            a_nature_bounded = a_nature_bounded.reshape(*a_nature_flat.shape)

        return a_nature_bounded

    def reset_random(self):
        return self.reset()

    def reset(self):
        # self.current_full_state = np.zeros(self.N, dtype=int)
        # self.current_full_state = self.random_stream.choice(self.observation_space, self.N)
        self.current_full_state = self.random_stream.uniform(low=[0] * self.N, high=[1] * self.N)
        # sample from [0.33, 1] is worse. sample from [0, 0.33] is also worse
        return self.current_full_state.reshape(self.N, self.observation_dimension)

    def render(self):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        seed1 = seed
        if seed1 is not None:
            self.random_stream.seed(seed=seed1)
            # print('seeded with',seed1)
        else:
            seed1 = np.random.randint(1e9)
            self.random_stream.seed(seed=seed1)

        return [seed1]


class CounterExampleEnv(gym.Env):
    def __init__(self, N, B, seed, noise_level=1, noise_shape=1):

        S = 2 # this is not used unless we want to one-hot-encode states
        A = 2
        # N = 3

        self.N = N
        self.observation_space = np.arange(S)
        self.action_space = np.arange(A)
        self.observation_dimension = 1
        self.action_dimension = 1
        self.action_dim_nature = N
        self.S = S
        self.A = A
        self.B = B

        self.current_full_state = np.zeros(N)
        self.random_stream = np.random.RandomState()

        # make sure to set this whenever environment is created, but do it outside so it always the same
        self.sampled_parameter_ranges = None

        self.seed(seed=seed)
        self.max_reward = 5
        self.noise_level = noise_level
        self.noise_shape = noise_shape
        self.T, self.R, self.C, self.perturbed_R = self.get_experiment(N)

        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def get_experiment(self, N):
        T = np.zeros((N,2))
        C = np.array([0, 1])
        R = []
        perturbed_R = []
        return T, R, C, perturbed_R

    def update_transition_probs(self, arms_to_update):
        num_states = [4,5,6] # randomly choose the number of states
        opt_reward_range = [3,5]
        # slight abuse of input arguments.
        # here "noise_level" argument is used to change num_states
        # here "noise_shape" argument is used to change large reward at optimal states
        if self.noise_level == 2:
            num_states = [5,6,7]
        elif self.noise_level == 3:
            num_states = [6,7,8]
        elif self.noise_level == 4:
            num_state = [4,5,6,7,8]

        # if self.noise_shape == 2:
        #     opt_reward_range = [4,6]
        # elif self.noise_shape == 3:
        #     opt_reward_range = [5,7] # this is bad

        self.max_reward = opt_reward_range[-1] * self.noise_shape

        for i in range(self.N):
            if arms_to_update[i] > 0.5:
                self.T[i, 0] = self.random_stream.choice(num_states, replace=True)
                self.T[i, 1] = self.random_stream.uniform(low=opt_reward_range[0], high=opt_reward_range[1])


    def step(self, a_agent, opt_in, perturb_reward=False, indices_to_perturb=[]):
        ###### Get next state
        next_full_state = np.zeros(self.N, dtype=float)
        rewards = np.zeros(self.N)
        for i in range(self.N):
            current_arm_state = self.current_full_state[i]  # want continuous states. not rounded
            action = int(a_agent[i])
            if current_arm_state == 0 or current_arm_state == 1:
                next_arm_state = current_arm_state # self-loop at optimal state (state 1) and bad state (state 0)
            else:
                next_arm_state = current_arm_state + 1 / self.T[i, 0] if action > 0.5 else 0
            next_arm_state = np.clip(next_arm_state, 0,1)
            next_full_state[i] = next_arm_state

            rewards[i] = self.reward_fun(next_full_state[i], i)
            if perturb_reward and i in indices_to_perturb: # noisy arms only seed perturbed reward
                rewards[i] = self.reward_fun(next_full_state[i], i, perturbed=True)

        self.current_full_state = next_full_state
        next_full_state = next_full_state.reshape(self.N, self.observation_dimension)

        return next_full_state, rewards, False, None

    def reward_fun(self, state, arm_idx, perturbed=False):
        if perturbed:
            if state == 0:
                return self.random_stream.uniform(low=1, high=2)
            elif state == 1:
                return self.T[arm_idx, 1] * self.noise_shape # get optimal reward
            else:
                return 0
        else:
            if state == 0:
                return 0
            elif state == 1:
                return self.T[arm_idx, 1] # get optimal reward
            else:
                return 2


    def reset_random(self):
        return self.reset()

    def reset(self):
        self.current_full_state = np.ones(self.N) / 10 # start at the lowest state above 0
        # self.random_stream.uniform(low=[0] * self.N, high=[1] * self.N)
        return self.current_full_state.reshape(self.N, self.observation_dimension)

    def render(self):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        seed1 = seed
        if seed1 is not None:
            self.random_stream.seed(seed=seed1)
            # print('seeded with',seed1)
        else:
            seed1 = np.random.randint(1e9)
            self.random_stream.seed(seed=seed1)

        return [seed1]


class ContinuousStateExampleEnv(gym.Env):
    def __init__(self, N, B, seed, noise_level=1, noise_shape=1):

        S = 2
        A = 2
        # N = 3

        self.N = N
        self.observation_space = np.arange(S)
        self.action_space = np.arange(A)
        self.observation_dimension = 1
        self.action_dimension = 1
        self.action_dim_nature = N
        self.S = S
        self.A = A
        self.B = B

        self.current_full_state = np.zeros(N)
        self.random_stream = np.random.RandomState()

        # make sure to set this whenever environment is created, but do it outside so it always the same
        self.sampled_parameter_ranges = None

        self.seed(seed=seed)
        self.max_reward = 1
        self.num_partitions = 51 # this is not num of atoms. this is how we chop reward functions into pieces to add noise
        self.noise_level = noise_level
        self.noise_shape = noise_shape
        self.T, self.R, self.C, self.perturbed_R = self.get_experiment(N)

        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def get_experiment(self, N):
        t = np.array([[[0.0, -0.4],
                       [0.0, 0.2]],

                      [[0.0, 0.5],
                       [0.0, 0.2]]
                      ])

        T = []
        for i in range(N):
            T.append(t)
        T = np.array(T)
        C = np.array([0, 1])
        R = np.array([np.linspace(0, 1, self.num_partitions) for _ in range(N)])
        perturbed_R = np.array([np.linspace(0, 1, self.num_partitions) for _ in range(N)])
        noise = np.zeros(perturbed_R.shape)
        if self.noise_shape == 1:
            noise = self.random_stream.normal(loc=0, scale=1 * self.noise_level, size=perturbed_R.shape)
        elif self.noise_shape == 2:
            noise = self.random_stream.uniform(low=-0.3, high=0.3, size=perturbed_R.shape)
        elif self.noise_shape == 3:
            noise = self.random_stream.uniform(low=-0.5, high=0.5, size=perturbed_R.shape)
        elif self.noise_shape == 4:
            noise = self.random_stream.uniform(low=-0.7, high=0.7, size=perturbed_R.shape)
        elif self.noise_shape == 5:
            coin_flip_prob = 0.3
            coin_flips = np.random.binomial(n=1, p=coin_flip_prob, size=perturbed_R.shape)
            noise_1 = self.random_stream.uniform(low=-0.5, high=0.5, size=perturbed_R.shape)
            noise_2 = self.random_stream.normal(loc=0, scale=1 * self.noise_level, size=perturbed_R.shape)
            noise = coin_flips * noise_1 + (1 - coin_flips) * noise_2
        elif self.noise_shape == 6:
            coin_flip_prob = 0.5
            coin_flips = np.random.binomial(n=1, p=coin_flip_prob, size=perturbed_R.shape)
            noise_1 = self.random_stream.uniform(low=-0.5, high=0.5, size=perturbed_R.shape)
            noise_2 = self.random_stream.normal(loc=0, scale=1 * self.noise_level, size=perturbed_R.shape)
            noise = coin_flips * noise_1 + (1 - coin_flips) * noise_2
        elif self.noise_shape == 7:
            coin_flip_prob = 0.7
            coin_flips = np.random.binomial(n=1, p=coin_flip_prob, size=perturbed_R.shape)
            noise_1 = self.random_stream.uniform(low=-0.5, high=0.5, size=perturbed_R.shape)
            noise_2 = self.random_stream.normal(loc=0, scale=1 * self.noise_level, size=perturbed_R.shape)
            noise = coin_flips * noise_1 + (1 - coin_flips) * noise_2
        perturbed_R = np.clip(perturbed_R + noise, 0, 1)


        self.max_reward = 1
        self.num_opt_states = 1
        R[:, -self.num_opt_states:] = self.max_reward * np.ones(self.num_opt_states) # optimal states
        perturbed_R[:, -self.num_opt_states:] = self.max_reward * np.ones(self.num_opt_states) # optimal states

        return T, R, C, perturbed_R

    def update_transition_probs(self, arms_to_update):
        # arms_to_update is 1d array of length N. arms_to_update[i] == 1 if transition prob of arm i needs to be resampled
        # if action==0, then next state = current state + Normal(1st entry, 2nd entry)
        # if action==1, then next state = current state + Normal(3rd entry, 4th entry)
        # sample_ub = [0.2, 0.2, 0.2, 0.2]
        # sample_lb = [-0.2, 0.1, -0.2, 0.1]
        sample_ub = [0.2, 0.2, 0.2, 0.2]
        sample_lb = [-0.2, 0.1, -0.2, 0.1]
        for i in range(self.N):
            if arms_to_update[i] > 0.5:
                new_transition_probs = self.random_stream.uniform(low=sample_lb, high=sample_ub)
                new_transition_probs[0] = - new_transition_probs[2] # no pull effect is exactly the negate of pull
                new_transition_probs[1] = new_transition_probs[3] # same std
                self.T[i, :, :, 1] = new_transition_probs.reshape((2, 2))

    # # env has only binary actions so random is easy to generate
    # def random_agent_action(self):
    #     actions = np.zeros(self.N)
    #     choices = np.random.choice(np.arange(self.N), int(self.B), replace=False)
    #     actions[choices] = 1
    #     return actions

    def step(self, a_agent, opt_in, perturb_reward=False, indices_to_perturb=[]):
        ###### Get next state
        next_full_state = np.zeros(self.N, dtype=float)
        rewards = np.zeros(self.N)
        for i in range(self.N):
            # current_arm_state=int(self.current_full_state[i])
            current_arm_state = self.current_full_state[i]  # want continuous states. not rounded
            action = int(a_agent[i])
            # when action is i, state moves according to N(T[i,0,0], T[i,1,0])
            next_arm_state = current_arm_state + self.random_stream.normal(loc=self.T[i, action, 0, 1], scale=self.T[i, action, 1, 1])
            next_arm_state = np.clip(next_arm_state, 0,1)
            next_full_state[i] = next_arm_state

            # # once reach opt states, always stay there.
            # if self.current_full_state[i] == 1: # >= 1 - self.num_opt_states / self.num_partitions:
            #     next_full_state[i] = 1

            rewards[i] = self.reward_fun(next_full_state[i], i)
            if perturb_reward and i in indices_to_perturb: # noisy arms only seed perturbed reward
                rewards[i] = self.reward_fun(next_full_state[i], i, perturbed=True)

        self.current_full_state = next_full_state
        next_full_state = next_full_state.reshape(self.N, self.observation_dimension)

        return next_full_state, rewards, False, None

    def reward_fun(self, state, arm_idx, perturbed=False):
        # find the partition this state belongs to
        partition = int(round(self.num_partitions * state))
        partition = min(partition, self.num_partitions - 1)
        if perturbed:
            return self.perturbed_R[arm_idx, partition]
        else:
            return state


    def reset_random(self):
        return self.reset()

    def reset(self):
        # self.current_full_state = np.zeros(self.N, dtype=int)
        self.current_full_state = self.random_stream.uniform(low=[0] * self.N, high=[1] * self.N)
        # sample from [0.33, 1] is worse. sample from [0, 0.33] is also worse
        return self.current_full_state.reshape(self.N, self.observation_dimension)

    def render(self):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        seed1 = seed
        if seed1 is not None:
            self.random_stream.seed(seed=seed1)
            # print('seeded with',seed1)
        else:
            seed1 = np.random.randint(1e9)
            self.random_stream.seed(seed=seed1)

        return [seed1]


class FakeT:
    def __init__(self, shape):
        self.shape=shape

class SISRobustEnv(gym.Env):
    def __init__(self, N, B, pop_size, seed, noise_level=1, noise_shape=1):

        S = pop_size+1
        A = 3

        self.N = N
        self.observation_space = np.arange(S)
        self.action_space = np.arange(A)

        self.observation_dimension = 1
        self.action_dimension = 1
        self.action_dim_nature = N*4

        self.S = S
        self.A = A
        self.B = B
        self.pop_size = pop_size

        self.current_full_state = np.zeros(N)
        self.random_stream = np.random.RandomState()

        self.PARAMETER_RANGES = self.get_parameter_ranges(self.N)

        # make sure to set this whenever environment is created, but do it outside so it always the same
        # self.sampled_parameter_ranges = None
        self.sampled_parameter_ranges = self.sample_parameter_ranges() # shape (n_arms, 4, 2)

        # this model only needs its params set once at the beginning
        self.param_setting = np.zeros(self.sampled_parameter_ranges.shape[:-1])


        self.seed(seed=seed)
        self.max_reward = 1 # placeholder. will be updated in get_experiment()
        self.noise_level = noise_level
        self.noise_shape = noise_shape
        self.T, self.R, self.C, self.perturbed_R = self.get_experiment(N)

        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def update_transition_probs(self, arms_to_update):
        # arms_to_update is 1d array of length N. arms_to_update[i] == 1 if transition prob of arm i needs to be resampled
        # sample_ub = self.sampled_parameter_ranges[0,:,1] # all arms are sampled from the same distribution. so they use the same range
        # sample_lb = self.sampled_parameter_ranges[0,:,0]
        for i in range(self.N):
            if arms_to_update[i] > 0.5:
                # ranges are randomly sampled. then random sample arms from the range
                sample_ub = self.sampled_parameter_ranges[i, :, 1]
                sample_lb = self.sampled_parameter_ranges[i, :, 0]
                new_params = self.random_stream.uniform(low=sample_lb, high=sample_ub)
                self.param_setting[i, :] = new_params


    def p_i_s(self, q_t, i, s, pop_size):
        prob = 0
        if s == pop_size:
            if i == 0:
                prob = 1
            else:
                prob = 0
        else:
            binom = comb(s, i)
            prob = binom * q_t**(i)*(1-q_t)**(s - i)

        return prob


    def compute_distro(self, arm, s, a):

        # p(infect | contact), lower is better
        r_t = self.param_setting[arm, 0] #np.random.rand()*(r_t_range[1] - r_t_range[0]) + r_t_range[0]
        # number of contacts for delta_t, lower is better
        lam = self.param_setting[arm, 1] #np.random.rand()*(lam_range[1] - lam_range[0]) + lam_range[0]
        # action effect, larger is better
        a_effect_1 = self.param_setting[arm, 2] #np.random.rand()*(a_effect_1_range[1] - a_effect_1_range[0]) + a_effect_1_range[0]
        # action effect, larger is better
        a_effect_2 = self.param_setting[arm, 3] #np.random.rand()*(a_effect_2_range[1] - a_effect_2_range[0]) + a_effect_2_range[0]
        
        # print('r_t',r_t)
        # print('lam',lam)
        # print('a',a_effect_1)
        # print()

        delta_t = 1

        poisson_param = lam*delta_t

        S = self.S
        A = self.A
        pop_size = self.pop_size

        distro = np.zeros(S,dtype=np.float64)
        
        beta_t = (pop_size - s)/pop_size
        EPS = 1e-7
        
        q_t = None

        if a == 0:
            q_t = 1 - np.e**(-poisson_param * beta_t * r_t) 
        elif a == 1:
            q_t = 1 - np.e**(-poisson_param * beta_t/(a_effect_1) * r_t) 
        elif a == 2:
            q_t = 1 - np.e**(-poisson_param * beta_t * r_t/(a_effect_2)) 

        for sp in range(S):
            # print('s:',s)
            # print('sp:',sp)
            # print('pop_size:',pop_size)
            # print(q_t)
            # print(pop_size - s)
            # print()
            if pop_size - s <= sp and sp <= pop_size:
                # print("Here")
                # print('s:',s)
                # print('sp:',sp)
                num_infected = pop_size - sp
                prob = self.p_i_s(q_t, num_infected, s, pop_size)
                # print(prob)
                # print()
                distro[sp] = prob

        inds = distro < EPS
        distro[inds] = 0
        distro = distro / distro.sum()

        return distro


    # We will sample ranges from within these to get some extra randomness
    def get_parameter_ranges(self, N):

        # make it harder for arms to reach top states
        # decrease a_coeff_1_range and a_coeff_2_range. increase num of contacts (lam) and rate of infection (r_t)
        r_t_range = [0.5, 0.99]
        lam_range = [1, 10]  # people per day
        a_effect_1_range = [0.25, 2]  # multiplicative effect on each parameter
        # a_effect_1_range = [1, 8]  # multiplicative effect on each parameter
        a_effect_2_range = [1, 8]  # multiplicative effect on each parameter
        # r_t_range = [0.5, 0.99]
        # lam_range = [1, 10]  # people per day
        # a_effect_1_range = [1, 10]  # multiplicative effect on each parameter
        # a_effect_2_range = [1, 10]  # multiplicative effect on each parameter


        parameter_ranges = np.array([
            [
                r_t_range, 
                lam_range, 
                a_effect_1_range, 
                a_effect_2_range
            ] for _ in range(N)
        ])


        return parameter_ranges


    def sample_parameter_ranges(self):

        draw = self.random_stream.rand(*self.PARAMETER_RANGES.shape)
        mult_transform = (self.PARAMETER_RANGES.max(axis=-1) - self.PARAMETER_RANGES.min(axis=-1))
        mult_transform = np.expand_dims(mult_transform, axis=-1)
        add_transform = self.PARAMETER_RANGES.min(axis=-1)
        add_transform = np.expand_dims(add_transform, axis=-1)

        draw.sort(axis=-1)

        sampled_ranges = draw*mult_transform + add_transform

        assert self.check_ranges(sampled_ranges, self.PARAMETER_RANGES)

        return sampled_ranges


    def check_ranges(self, sampled, edges):
        all_good = True
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                # lower range must be larger or equal to lower edge
                all_good &= (sampled[i,j,0] >= edges[i,j,0])
                # upper range must be smaller or equal to upper edge
                all_good &= (sampled[i,j,1] <= edges[i,j,1])
                if not all_good:
                    print('range ',edges[i,j])
                    print('sample',sampled[i,j])
                    print()

        return all_good


    def get_experiment(self, N):
        T = FakeT((N,self.S,self.A,self.S))
        R = np.array([np.linspace(0, 1, self.S) for _ in range(N)])
        C = np.array([0, 1, 2])

        perturbed_R = np.array([np.linspace(0, 1, self.S) for _ in range(N)])
        noise = np.zeros(perturbed_R.shape)
        if self.noise_shape == 1:
            noise = self.random_stream.normal(loc=0, scale=1 * self.noise_level, size=perturbed_R.shape)
        elif self.noise_shape == 2:
            noise = self.random_stream.uniform(low=-0.3, high=0.3, size=perturbed_R.shape)
        elif self.noise_shape == 3:
            noise = self.random_stream.uniform(low=-0.5, high=0.5, size=perturbed_R.shape)
        elif self.noise_shape == 4:
            noise = self.random_stream.uniform(low=-0.7, high=0.7, size=perturbed_R.shape)
        elif self.noise_shape == 5:
            coin_flip_prob = 0.3
            coin_flips = np.random.binomial(n=1, p=coin_flip_prob, size=perturbed_R.shape)
            noise_1 = self.random_stream.uniform(low=-0.5, high=0.5, size=perturbed_R.shape)
            noise_2 = self.random_stream.normal(loc=0, scale=1 * self.noise_level, size=perturbed_R.shape)
            noise = coin_flips * noise_1 + (1 - coin_flips) * noise_2
        elif self.noise_shape == 6:
            coin_flip_prob = 0.5
            coin_flips = np.random.binomial(n=1, p=coin_flip_prob, size=perturbed_R.shape)
            noise_1 = self.random_stream.uniform(low=-0.5, high=0.5, size=perturbed_R.shape)
            noise_2 = self.random_stream.normal(loc=0, scale=1 * self.noise_level, size=perturbed_R.shape)
            noise = coin_flips * noise_1 + (1 - coin_flips) * noise_2
        elif self.noise_shape == 7:
            coin_flip_prob = 0.7
            coin_flips = np.random.binomial(n=1, p=coin_flip_prob, size=perturbed_R.shape)
            noise_1 = self.random_stream.uniform(low=-0.5, high=0.5, size=perturbed_R.shape)
            noise_2 = self.random_stream.normal(loc=0, scale=1 * self.noise_level, size=perturbed_R.shape)
            noise = coin_flips * noise_1 + (1 - coin_flips) * noise_2
        perturbed_R = np.clip(perturbed_R + noise, 0, 1)

        self.max_reward = 1
        num_opt_states = 1
        R[:, -num_opt_states:] = self.max_reward * np.ones(num_opt_states) # optimal states
        perturbed_R[:, -num_opt_states:] = self.max_reward * np.ones(num_opt_states) # optimal states

        return T, R, C, perturbed_R


    # # Fast random, inverse weighted, works for multi-action
    # def random_agent_action(self):
    #     actions = np.zeros(self.N,dtype=int)
    #
    #     current_action_cost = 0
    #     process_order = np.random.choice(np.arange(self.N), self.N, replace=False)
    #     for arm in process_order:
    #
    #         # select an action at random
    #         num_valid_actions_left = len(self.C[self.C<=self.B-current_action_cost])
    #         p = 1/(self.C[self.C<=self.B-current_action_cost]+1)
    #         p = p/p.sum()
    #         p = None
    #         a = np.random.choice(np.arange(num_valid_actions_left), 1, p=p)[0]
    #         current_action_cost += self.C[a]
    #         # if the next selection takes us over budget, break
    #         if current_action_cost > self.B:
    #             break
    #
    #         actions[arm] = a
    #
    #     return actions



    def set_params(self, a_nature):
        
        # only set this once
        param_setting = np.zeros(self.sampled_parameter_ranges.shape[:-1])
        for arm_i in range(a_nature.shape[0]):
            for param_i in range(a_nature.shape[1]):
                param = a_nature[arm_i, param_i]

                if param < self.sampled_parameter_ranges[arm_i, param_i, 0]:
                    print("Warning! nature action below allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, param_i]))
                    print("Setting to lower bound of range...")
                    param = self.sampled_parameter_ranges[arm_i, param_i, 0]
                    raise ValueError('bad setting')
                elif param > self.sampled_parameter_ranges[arm_i, param_i, 1]:
                    print("Warning! nature action above allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, param_i]))
                    print("Setting to upper bound of range...")
                    param = self.sampled_parameter_ranges[arm_i, param_i, 1]
                    raise ValueError('bad setting')

                param_setting[arm_i, param_i] = param
        self.param_setting = param_setting
                
                

    # a_agent should correspond to an action respresented in the transition matrix
    # a_nature should be a probability in the range specified by self.parameter_ranges
    def step(self, a_agent, opt_in, perturb_reward=False, indices_to_perturb=[]):
        # self.set_params(a_nature)

        ###### Get next state
        next_full_state = np.zeros(self.N, dtype=int)
        rewards = np.zeros(self.N)
        for i in range(self.N):
            current_arm_state=int(self.current_full_state[i])
            distro = self.compute_distro(i, current_arm_state, int(a_agent[i])) # self.T[i, current_arm_state, int(a_agent[i]), :]

            next_arm_state=np.argmax(self.random_stream.multinomial(1, distro))
            next_full_state[i]=next_arm_state
            rewards[i] = self.R[i, next_arm_state]
            if perturb_reward and i in indices_to_perturb: # noisy arms only seed perturbed reward
                rewards[i] = self.perturbed_R[i, next_arm_state]

        self.current_full_state = next_full_state
        next_full_state = next_full_state.reshape(self.N, self.observation_dimension)

        return next_full_state, rewards, False, None


    # only do this if you are sure the state space is small enough (e.g., less than ~500)
    def get_T_for_a_nature(self, a_nature):

        self.set_params(a_nature)
        T = np.zeros((self.N,self.S,self.A,self.S),dtype=np.float64)
        for arm_i in range(self.N):
            for s in range(self.S):
                for a in range(self.A):
                    T[arm_i, s, a] = self.compute_distro(arm_i, s, a)

        return T


    # this is easier to attach to environment code
    def bound_nature_actions(self, a_nature_flat, state=None, reshape=True):
        
        # num arms by num actions
        a_nature = a_nature_flat.reshape((self.N, self.sampled_parameter_ranges.shape[1]))

        a_nature_bounded = np.zeros(a_nature.shape)
        for arm_i in range(a_nature.shape[0]):
            for param_i in range(a_nature.shape[1]):
                
                param = a_nature[arm_i, param_i]

                lb = self.sampled_parameter_ranges[arm_i, param_i, 0]
                ub = self.sampled_parameter_ranges[arm_i, param_i, 1]

                a_nature_bounded[arm_i, param_i] = ((self.tanh(torch.as_tensor(param, dtype=torch.float32))+1)/2)*(ub - lb) + lb

        if not reshape:
            a_nature_bounded = a_nature_bounded.reshape(*a_nature_flat.shape)

        return a_nature_bounded


    def reset_random(self):
        return self.reset()

    def reset(self):
        # self.current_full_state = np.zeros(self.N, dtype=int)
        # tested this, it's about half as fast as randint
        # self.current_full_state = self.random_stream.choice(self.observation_space, self.N)
        self.current_full_state = self.random_stream.randint(low=0, high=int(self.S * 2 / 3), size=self.N)
        # sample from [0, self.S] is slightly smaller gap. sample from [0, self.S/2] is much smaller gap
        return self.current_full_state.reshape(self.N, self.observation_dimension)

    def render(self):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        seed1 = seed
        if seed1 is not None:
            self.random_stream.seed(seed=seed1)
            # print('seeded with',seed1)
        else:
            seed1 = np.random.randint(1e9)
            self.random_stream.seed(seed=seed1)

        return [seed1]
